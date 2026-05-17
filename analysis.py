#!/usr/bin/env python3
"""Analysis and runtime helpers for the Yao-Lee driver.

This file owns lazy Tenax imports, progress bars, stage timing, entropy-profile
analysis, and alpha-beta phase-scan analysis/classification. Model
construction stays in ``models.py`` and plot rendering stays in
``plot_outputs.py``.
"""

from __future__ import annotations

import time
import argparse
import math
import contextlib
import cProfile
import functools
import importlib
import importlib.metadata as importlib_metadata
import importlib.util
import io
import json
import os
import platform
import pstats
import subprocess
import sys
import tracemalloc
import warnings
from typing import Any, Callable, Dict, List, Sequence, Tuple

try:
    import resource
except Exception:  # pragma: no cover - resource is Unix-only.
    resource = None  # type: ignore[assignment]

warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API\..*",
    category=UserWarning,
    module=r"llvmlite\.binding\.ffi",
)

import numpy as np

ENTROPY_ORDERS = (1, 2, 3, 4)
GROUND_MANIFOLD_ABS_TOL_DEFAULT = 1e-12
GROUND_MANIFOLD_REL_TOL_DEFAULT = 1e-12

PROFILE_ENABLED = False
PROFILE_TIMING = True
PROFILE_MEMORY = True
PROFILE_CPROFILE = False
PROFILE_LINE_HOOKS = False
PROFILE_SCAN_POINTS = True
PROFILE_OUTPUT_JSON = True
PROFILE_OUTPUT_FOLDER = "outputs/profiling"

_PROFILE_PACKAGE_NAMES = (
    "numpy",
    "scipy",
    "quspin",
    "tenpy",
    "tenax",
    "quimb",
    "numba",
    "llvmlite",
)

_PROFILE_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

_PROFILE_STATE: Dict[str, Any] = {
    "enabled": bool(PROFILE_ENABLED),
    "timing": bool(PROFILE_TIMING),
    "memory": bool(PROFILE_MEMORY),
    "cprofile": bool(PROFILE_CPROFILE),
    "line_hooks": bool(PROFILE_LINE_HOOKS),
    "scan_points": bool(PROFILE_SCAN_POINTS),
    "output_json": bool(PROFILE_OUTPUT_JSON),
    "output_folder": str(PROFILE_OUTPUT_FOLDER),
    "run_start": None,
    "run_end": None,
    "stage_events": [],
    "scan_point_events": [],
    "metadata": {},
    "cprofile_profiler": None,
    "cprofile_stats": None,
    "tracemalloc_started_here": False,
    "tracemalloc_peak_mb": None,
    "tracemalloc_current_mb": None,
    "resource_peak_rss_mb": None,
    "finalized": False,
}


def profiling_enabled() -> bool:
    return bool(_PROFILE_STATE.get("enabled", False))


def profile_scan_points_enabled() -> bool:
    return profiling_enabled() and bool(_PROFILE_STATE.get("scan_points", True))


def _profile_bool_attr(args: Any, name: str, default: bool) -> bool:
    return bool(getattr(args, name, default))


def _profile_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return float(number)


def _profile_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _profile_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_profile_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _profile_json_safe(item())
        except Exception:
            pass
    return str(value)


def _profile_package_versions() -> Dict[str, Any]:
    versions: Dict[str, Any] = {}
    for package_name in _PROFILE_PACKAGE_NAMES:
        try:
            versions[package_name] = importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            versions[package_name] = None
        except Exception as exc:
            versions[package_name] = f"unavailable: {exc}"
    return versions


def _profile_module_version(module: Any) -> str | None:
    for attr_name in ("__version__", "version"):
        value = getattr(module, attr_name, None)
        if value is not None and not callable(value):
            return str(value)
    return None


def _profile_import_report(module_name: str) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "module": str(module_name),
        "found": False,
        "imported": False,
        "version": None,
        "module_file": None,
        "error": None,
    }
    try:
        spec = importlib.util.find_spec(module_name)
        report["found"] = spec is not None
    except Exception as exc:
        report["error"] = f"find_spec failed: {exc}"
        return report
    try:
        module = importlib.import_module(module_name)
        report["imported"] = True
        report["module_file"] = getattr(module, "__file__", None)
        report["version"] = _profile_module_version(module)
    except Exception as exc:
        report["error"] = str(exc)
    return report


def _profile_package_audit() -> Dict[str, Any]:
    versions = _profile_package_versions()
    imports: Dict[str, Any] = {}
    for module_name in _PROFILE_PACKAGE_NAMES:
        import_report = _profile_import_report(module_name)
        if versions.get(module_name) is None and import_report.get("version") is not None:
            versions[module_name] = import_report.get("version")
        imports[module_name] = import_report
    return {
        "versions": versions,
        "imports": imports,
    }


def _profile_config_show(module: Any) -> Dict[str, Any]:
    config = getattr(module, "__config__", None)
    if config is None:
        return {"available": False, "reason": "module has no __config__"}
    show_text = None
    try:
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            config.show()
        show_text = stream.getvalue()
    except Exception as exc:
        show_text = f"config.show() failed: {exc}"
    info: Dict[str, Any] = {
        "available": True,
        "show": show_text,
    }
    get_info = getattr(config, "get_info", None)
    if callable(get_info):
        for key in ("blas_opt_info", "lapack_opt_info", "openblas_info", "mkl_info"):
            try:
                info[key] = get_info(key)
            except Exception as exc:
                info[key] = f"unavailable: {exc}"
    return info


def _profile_blas_lapack_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    try:
        info["numpy"] = _profile_config_show(np)
    except Exception as exc:
        info["numpy"] = {"available": False, "error": str(exc)}
    try:
        scipy_module = importlib.import_module("scipy")
        info["scipy"] = _profile_config_show(scipy_module)
    except Exception as exc:
        info["scipy"] = {"available": False, "error": str(exc)}
    return info


def _profile_thread_environment() -> Dict[str, Any]:
    return {name: os.environ.get(name) for name in _PROFILE_THREAD_ENV_VARS}


def _profile_run_subprocess(command: Sequence[str], timeout_seconds: float) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "command": [str(part) for part in command],
        "timeout_seconds": float(timeout_seconds),
        "status": "not_run",
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "timed_out": False,
        "error": None,
    }
    try:
        completed = subprocess.run(
            [str(part) for part in command],
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds),
            check=False,
        )
        result.update(
            {
                "status": "completed",
                "returncode": int(completed.returncode),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
    except subprocess.TimeoutExpired as exc:
        result.update(
            {
                "status": "timeout",
                "timed_out": True,
                "stdout": exc.stdout if isinstance(exc.stdout, str) else "",
                "stderr": exc.stderr if isinstance(exc.stderr, str) else "",
                "error": f"timed out after {float(timeout_seconds):.1f}s",
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "error": str(exc),
            }
        )
    return result


def _profile_write_text_file(folder: str, filename: str, text: str) -> str | None:
    try:
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, filename)
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(text)
        return filepath
    except Exception as exc:
        warnings.warn(f"Failed to write profiling audit file {filename}: {exc}")
        return None


def _profile_write_pip_freeze(folder: str) -> Dict[str, Any]:
    result = _profile_run_subprocess([sys.executable, "-m", "pip", "freeze"], timeout_seconds=60.0)
    output_text = str(result.get("stdout") or "")
    result["stdout_line_count"] = int(len(output_text.splitlines()))
    result["stdout_character_count"] = int(len(output_text))
    if result.get("status") != "completed" or int(result.get("returncode") or 0) != 0:
        output_text = (
            "# pip freeze did not complete successfully.\n"
            f"# status: {result.get('status')}\n"
            f"# returncode: {result.get('returncode')}\n"
            f"# error: {result.get('error')}\n"
            "# stderr:\n"
            f"{result.get('stderr') or ''}\n"
            "# stdout:\n"
            f"{result.get('stdout') or ''}\n"
        )
    filepath = _profile_write_text_file(folder, "pip_freeze.txt", output_text)
    result["output_file"] = filepath
    result["stdout"] = "" if filepath is not None else output_text
    return result


def _profile_quspin_small_ed_validation() -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "status": "not_run",
        "quspin_imported": False,
        "small_ed_tests_passed": False,
        "error": None,
    }
    timing_enabled = bool(_PROFILE_STATE.get("timing", True))
    _PROFILE_STATE["timing"] = False
    try:
        quspin_backend = importlib.import_module("quspin_backend")
        report["quspin_imported"] = True
        available, reason = quspin_backend.quspin_package_available()
        report["quspin_package_available"] = bool(available)
        report["quspin_package_reason"] = reason
        if not bool(available):
            report["status"] = "skipped"
            report["error"] = str(reason or "QuSpin package is not available.")
            return report
        from ed_backend import run_spin_orbital_u1_exact_spectrum
        from models import build_lattice_geometry, build_model_spec

        geometry = build_lattice_geometry(
            "honeycomb",
            length_x=1,
            length_y=2,
            circumference_x=True,
            circumference_y=True,
        )
        model_spec = build_model_spec("1/2", "1/2", "yao_lee", "z")
        alpha = 0.7
        beta = 0.2
        standard_spectrum, _standard_vectors, _basis_list, _basis_map = run_spin_orbital_u1_exact_spectrum(
            geometry=geometry,
            model_spec=model_spec,
            alpha=alpha,
            beta=beta,
            coupling_j=1.0,
            eigenstate_count=1,
            check_ground_state_degeneracy=False,
            show_progress=False,
            sparse_tol=1.0e-11,
            sparse_maxiter=20000,
            use_sz_block=False,
            target_sz2=0,
            use_tau_z_block=True,
            target_tz2=0,
        )
        quspin_spectrum, _quspin_vectors = quspin_backend.run_small_cluster_exact_spectrum(
            geometry=geometry,
            model_spec=model_spec,
            alpha=alpha,
            beta=beta,
            coupling_j=1.0,
            eigenstate_count=1,
            check_ground_state_degeneracy=False,
            external_field_terms=[],
            show_progress=False,
            use_sz_block=False,
            use_tau_z_block=True,
            target_tz2=0,
            use_z2_block=False,
        )
        standard_energy = float(standard_spectrum["ground_state_energy"])
        quspin_energy = float(quspin_spectrum["ground_state_energy"])
        tz_difference = abs(standard_energy - quspin_energy)
        z2_sector_energies: List[float] = []
        for parity in (0, 1):
            z2_spectrum, _z2_vectors = quspin_backend.run_small_cluster_exact_spectrum(
                geometry=geometry,
                model_spec=model_spec,
                alpha=alpha,
                beta=beta,
                coupling_j=1.0,
                eigenstate_count=1,
                check_ground_state_degeneracy=False,
                external_field_terms=[],
                show_progress=False,
                use_sz_block=False,
                use_tau_z_block=True,
                target_tz2=0,
                use_z2_block=True,
                z2_generator="spin_flip",
                z2_target_parity=parity,
            )
            z2_sector_energies.append(float(z2_spectrum["ground_state_energy"]))
        z2_difference = abs(min(z2_sector_energies) - quspin_energy)
        tolerance = 1.0e-8
        report["validation"] = {
            "geometry": {
                "lattice": "honeycomb",
                "length_x": 1,
                "length_y": 2,
                "number_of_sites": int(geometry.number_of_sites),
                "boundary": "pbcX_pbcY",
            },
            "parameters": {
                "alpha": float(alpha),
                "beta": float(beta),
                "coupling_j": 1.0,
                "target_tz2": 0,
            },
            "checks": {
                "quspin_tz_matches_standard_ed": {
                    "status": "passed" if tz_difference <= tolerance else "failed",
                    "standard_tz_ground_energy": standard_energy,
                    "quspin_tz_ground_energy": quspin_energy,
                    "absolute_difference": float(tz_difference),
                    "tolerance": float(tolerance),
                },
                "quspin_tz_z2_parity_min_matches_tz": {
                    "status": "passed" if z2_difference <= tolerance else "failed",
                    "z2_sector_energies": z2_sector_energies,
                    "quspin_tz_ground_energy": quspin_energy,
                    "absolute_difference": float(z2_difference),
                    "tolerance": float(tolerance),
                },
            },
        }
        report["small_ed_tests_passed"] = (
            tz_difference <= tolerance
            and z2_difference <= tolerance
        )
        report["status"] = "passed" if bool(report["small_ed_tests_passed"]) else "failed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
    finally:
        _PROFILE_STATE["timing"] = timing_enabled
    return report


def _profile_tested_environment_suggestion(package_audit: Dict[str, Any]) -> str:
    versions = package_audit.get("versions", {}) if isinstance(package_audit, dict) else {}
    python_pin = ".".join(str(sys.version_info[index]) for index in (0, 1))
    lines = [
        "# Suggested tested environment file",
        "# This is a generated report only. Nothing was installed, pinned, or changed automatically.",
        "# It is emitted only after QuSpin imports and the small ED validation passes in this run.",
        "name: yao-lee-quspin-tested",
        "channels:",
        "  - conda-forge",
        "dependencies:",
        f"  - python={python_pin}",
        "  - pip",
        "  - pip:",
    ]
    for package_name in _PROFILE_PACKAGE_NAMES:
        version = versions.get(package_name)
        if isinstance(version, str) and version and not version.startswith("unavailable"):
            lines.append(f"      - {package_name}=={version}")
    return "\n".join(lines) + "\n"


def _profile_environment_report_text(audit: Dict[str, Any]) -> str:
    lines = [
        "Yao-Lee Profiling Environment Audit",
        "",
        f"Python: {audit.get('python_version')}",
        f"Executable: {audit.get('python_executable')}",
        f"Platform: {audit.get('platform')}",
        f"OS: {audit.get('os')}",
        "",
        "Thread environment variables:",
    ]
    thread_env = audit.get("thread_environment_variables", {})
    if isinstance(thread_env, dict):
        for name in _PROFILE_THREAD_ENV_VARS:
            lines.append(f"  {name}={thread_env.get(name)}")
    lines.extend(["", "Package imports:"])
    packages = audit.get("packages", {})
    imports = packages.get("imports", {}) if isinstance(packages, dict) else {}
    versions = packages.get("versions", {}) if isinstance(packages, dict) else {}
    if isinstance(imports, dict):
        for package_name in _PROFILE_PACKAGE_NAMES:
            import_report = imports.get(package_name, {})
            imported = bool(import_report.get("imported")) if isinstance(import_report, dict) else False
            error = import_report.get("error") if isinstance(import_report, dict) else None
            lines.append(
                f"  {package_name}: version={versions.get(package_name) if isinstance(versions, dict) else None}, "
                f"imported={imported}, error={error}"
            )
    pip_check = audit.get("pip_check", {})
    lines.extend(
        [
            "",
            "pip check:",
            f"  status={pip_check.get('status') if isinstance(pip_check, dict) else None}",
            f"  returncode={pip_check.get('returncode') if isinstance(pip_check, dict) else None}",
        ]
    )
    if isinstance(pip_check, dict) and pip_check.get("stdout"):
        lines.append("  stdout:")
        lines.append(str(pip_check.get("stdout")).rstrip())
    if isinstance(pip_check, dict) and pip_check.get("stderr"):
        lines.append("  stderr:")
        lines.append(str(pip_check.get("stderr")).rstrip())

    quspin_validation = audit.get("quspin_small_ed_validation", {})
    validation_passed = (
        isinstance(quspin_validation, dict)
        and bool(quspin_validation.get("small_ed_tests_passed", False))
    )
    lines.extend(
        [
            "",
            "QuSpin small ED validation:",
            f"  status={quspin_validation.get('status') if isinstance(quspin_validation, dict) else None}",
            f"  passed={validation_passed}",
        ]
    )
    if validation_passed:
        lines.extend(["", _profile_tested_environment_suggestion(packages)])
    else:
        lines.extend(
            [
                "",
                "No tested environment suggestion was emitted because QuSpin import and small ED validation did not both pass.",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _profile_environment_audit(output_folder: str | None = None) -> Dict[str, Any]:
    cached = _PROFILE_STATE.get("environment_audit")
    if isinstance(cached, dict):
        return cached
    folder = str(output_folder or _PROFILE_STATE.get("output_folder") or PROFILE_OUTPUT_FOLDER)
    package_audit = _profile_package_audit()
    audit: Dict[str, Any] = {
        "python_version": sys.version,
        "python_version_info": {
            "major": int(sys.version_info.major),
            "minor": int(sys.version_info.minor),
            "micro": int(sys.version_info.micro),
        },
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": package_audit,
        "blas_lapack": _profile_blas_lapack_info(),
        "thread_environment_variables": _profile_thread_environment(),
        "pip_check": _profile_run_subprocess([sys.executable, "-m", "pip", "check"], timeout_seconds=30.0),
        "pip_freeze": _profile_write_pip_freeze(folder),
        "quspin_small_ed_validation": _profile_quspin_small_ed_validation(),
    }
    suggestion_emitted = bool(
        isinstance(audit.get("quspin_small_ed_validation"), dict)
        and audit["quspin_small_ed_validation"].get("small_ed_tests_passed", False)
    )
    audit["tested_environment_suggestion"] = {
        "emitted": suggestion_emitted,
        "location": "environment_audit.txt" if suggestion_emitted else None,
        "reason": (
            "QuSpin imported and the small ED validation passed."
            if suggestion_emitted
            else "QuSpin import and small ED validation did not both pass; no environment suggestion emitted."
        ),
        "note": "This is a generated suggestion only; no package versions are pinned or changed automatically.",
    }
    report_path = _profile_write_text_file(folder, "environment_audit.txt", _profile_environment_report_text(audit))
    audit["environment_report_file"] = report_path
    _PROFILE_STATE["environment_audit"] = audit
    return audit


def _profile_environment_metadata() -> Dict[str, Any]:
    return {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "platform_python_implementation": platform.python_implementation(),
        "package_versions": _profile_package_versions(),
        "line_profiler_builtin_hook_available": callable(getattr(__import__("builtins"), "profile", None)),
    }


def configure_profiling_from_args(args: Any) -> None:
    """Configure optional profiling from the driver args.

    The default path is a pure no-op: when profile_enabled is false this clears
    any prior in-process records and does not alter outputs.
    """
    configure_profiling(
        enabled=_profile_bool_attr(args, "profile_enabled", PROFILE_ENABLED),
        timing=_profile_bool_attr(args, "profile_timing", PROFILE_TIMING),
        memory=_profile_bool_attr(args, "profile_memory", PROFILE_MEMORY),
        cprofile_enabled=_profile_bool_attr(args, "profile_cprofile", PROFILE_CPROFILE),
        line_hooks=_profile_bool_attr(args, "profile_line_hooks", PROFILE_LINE_HOOKS),
        scan_points=_profile_bool_attr(args, "profile_scan_points", PROFILE_SCAN_POINTS),
        output_json=_profile_bool_attr(args, "profile_output_json", PROFILE_OUTPUT_JSON),
        output_folder=str(getattr(args, "profile_output_folder", PROFILE_OUTPUT_FOLDER)),
    )


def configure_profiling(
    *,
    enabled: bool = PROFILE_ENABLED,
    timing: bool = PROFILE_TIMING,
    memory: bool = PROFILE_MEMORY,
    cprofile_enabled: bool = PROFILE_CPROFILE,
    line_hooks: bool = PROFILE_LINE_HOOKS,
    scan_points: bool = PROFILE_SCAN_POINTS,
    output_json: bool = PROFILE_OUTPUT_JSON,
    output_folder: str = PROFILE_OUTPUT_FOLDER,
) -> None:
    profiler = _PROFILE_STATE.get("cprofile_profiler")
    if profiler is not None:
        try:
            profiler.disable()
        except Exception:
            pass

    _PROFILE_STATE.clear()
    _PROFILE_STATE.update(
        {
            "enabled": bool(enabled),
            "timing": bool(timing),
            "memory": bool(memory),
            "cprofile": bool(cprofile_enabled),
            "line_hooks": bool(line_hooks),
            "scan_points": bool(scan_points),
            "output_json": bool(output_json),
            "output_folder": str(output_folder),
            "run_start": None,
            "run_end": None,
            "stage_events": [],
            "scan_point_events": [],
            "metadata": {},
            "cprofile_profiler": None,
            "cprofile_stats": None,
            "tracemalloc_started_here": False,
            "tracemalloc_peak_mb": None,
            "tracemalloc_current_mb": None,
            "resource_peak_rss_mb": None,
            "finalized": False,
        }
    )
    if not bool(enabled):
        return

    _PROFILE_STATE["run_start"] = time.perf_counter()
    _PROFILE_STATE["metadata"] = {
        "environment": _profile_environment_metadata(),
        "options": {
            "profile_enabled": bool(enabled),
            "profile_timing": bool(timing),
            "profile_memory": bool(memory),
            "profile_cprofile": bool(cprofile_enabled),
            "profile_line_hooks": bool(line_hooks),
            "profile_scan_points": bool(scan_points),
            "profile_output_json": bool(output_json),
            "profile_output_folder": str(output_folder),
        },
    }
    if bool(memory) and not tracemalloc.is_tracing():
        try:
            tracemalloc.start()
            _PROFILE_STATE["tracemalloc_started_here"] = True
        except Exception:
            _PROFILE_STATE["tracemalloc_started_here"] = False
    if bool(cprofile_enabled):
        profile_obj = cProfile.Profile()
        profile_obj.enable()
        _PROFILE_STATE["cprofile_profiler"] = profile_obj


def update_profile_metadata(**metadata: Any) -> None:
    if not profiling_enabled():
        return
    target = _PROFILE_STATE.setdefault("metadata", {})
    for key, value in metadata.items():
        target[key] = value


def _record_profile_stage(name: str, start: float, end: float) -> None:
    if not (
        profiling_enabled()
        and bool(_PROFILE_STATE.get("timing", True))
    ):
        return
    elapsed = float(end - start)
    events = _PROFILE_STATE.setdefault("stage_events", [])
    events.append(
        {
            "stage": str(name),
            "start_perf_counter": float(start),
            "end_perf_counter": float(end),
            "wall_time_seconds": elapsed,
        }
    )


@contextlib.contextmanager
def profile_stage(name: str):
    if not (
        profiling_enabled()
        and bool(_PROFILE_STATE.get("timing", True))
    ):
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        _record_profile_stage(str(name), start, end)


def profiled_function(name: str | None = None):
    """Decorator for optional standard-library timing around a function."""
    def decorate(func: Callable[..., Any]):
        stage_name = str(name or getattr(func, "__qualname__", getattr(func, "__name__", "function")))

        @functools.wraps(func)
        def wrapped(*args: Any, **kwargs: Any):
            with profile_stage(stage_name):
                return func(*args, **kwargs)

        return wrapped

    return decorate


def profile(func: Callable[..., Any] | None = None):
    """@profile-compatible no-op unless a builtins profile hook is active."""
    def decorate(inner: Callable[..., Any]):
        hook = getattr(__import__("builtins"), "profile", None)
        if (
            profiling_enabled()
            and bool(_PROFILE_STATE.get("line_hooks", False))
            and callable(hook)
        ):
            return hook(inner)
        return inner

    if func is None:
        return decorate
    return decorate(func)


def record_scan_point_timing(
    *,
    mode: str,
    alpha: float | None = None,
    beta: float | None = None,
    status: str | None = None,
    wall_time_seconds: float,
    point_index: int | None = None,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "mode": str(mode),
        "wall_time_seconds": float(wall_time_seconds),
    }
    if point_index is not None:
        record["point_index"] = int(point_index)
    if alpha is not None:
        record["alpha"] = float(alpha)
    if beta is not None:
        record["beta"] = float(beta)
        record["lambda"] = float(beta)
    if status is not None:
        record["status"] = str(status)
    if extra:
        for key, value in extra.items():
            record[str(key)] = value
    if profile_scan_points_enabled():
        _PROFILE_STATE.setdefault("scan_point_events", []).append(dict(record))
    return record


def _profile_timing_table(events: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[float]] = {}
    for event in events:
        stage = str(event.get("stage", "unknown"))
        elapsed = _profile_float(event.get("wall_time_seconds"))
        if elapsed is None:
            continue
        grouped.setdefault(stage, []).append(float(elapsed))
    rows: List[Dict[str, Any]] = []
    for stage in sorted(grouped):
        values = grouped[stage]
        total = float(sum(values))
        rows.append(
            {
                "stage": stage,
                "calls": int(len(values)),
                "total_seconds": total,
                "mean_seconds": total / float(max(1, len(values))),
                "min_seconds": float(min(values)),
                "max_seconds": float(max(values)),
            }
        )
    rows.sort(key=lambda row: float(row["total_seconds"]), reverse=True)
    return rows


def _profile_scan_point_summary(events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    table = [dict(event) for event in events]
    by_mode: Dict[str, Dict[str, Any]] = {}
    for event in table:
        mode = str(event.get("mode", "unknown"))
        elapsed = _profile_float(event.get("wall_time_seconds"))
        if elapsed is None:
            continue
        item = by_mode.setdefault(
            mode,
            {
                "points": 0,
                "total_seconds": 0.0,
                "min_seconds": None,
                "max_seconds": None,
            },
        )
        item["points"] = int(item["points"]) + 1
        item["total_seconds"] = float(item["total_seconds"]) + float(elapsed)
        item["min_seconds"] = (
            float(elapsed)
            if item["min_seconds"] is None
            else min(float(item["min_seconds"]), float(elapsed))
        )
        item["max_seconds"] = (
            float(elapsed)
            if item["max_seconds"] is None
            else max(float(item["max_seconds"]), float(elapsed))
        )
    for item in by_mode.values():
        item["mean_seconds"] = float(item["total_seconds"]) / float(max(1, int(item["points"])))
    return {
        "count": int(len(table)),
        "by_mode": by_mode,
        "table": table,
    }


def _first_nested(mapping: Any, paths: Sequence[Sequence[str]]) -> Any:
    if not isinstance(mapping, dict):
        return None
    for path in paths:
        cursor: Any = mapping
        ok = True
        for key in path:
            if not isinstance(cursor, dict) or key not in cursor:
                ok = False
                break
            cursor = cursor[key]
        if ok:
            return cursor
    return None


def _collect_nnz_values(value: Any, output: List[int]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() == "nnz":
                try:
                    if item is not None:
                        output.append(int(item))
                except (TypeError, ValueError):
                    pass
            else:
                _collect_nnz_values(item, output)
    elif isinstance(value, list):
        for item in value:
            _collect_nnz_values(item, output)


def _estimated_dense_memory_mb(dim: Any) -> float | None:
    try:
        dimension = int(dim)
    except (TypeError, ValueError):
        return None
    if dimension < 0:
        return None
    return float(dimension * dimension * 16) / float(1024 * 1024)


def _profile_resource_peak_rss_mb() -> float | None:
    if resource is None:
        return None
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        raw = float(usage.ru_maxrss)
    except Exception:
        return None
    if sys.platform == "darwin":
        return raw / float(1024 * 1024)
    return raw / float(1024)


def profile_metadata_from_summary(summary: Dict[str, Any] | None) -> Dict[str, Any]:
    if not isinstance(summary, dict):
        return {}
    parameters = summary.get("parameters", {}) if isinstance(summary.get("parameters"), dict) else {}
    ed = summary.get("ed", {}) if isinstance(summary.get("ed"), dict) else {}
    ed_spectrum = ed.get("spectrum", {}) if isinstance(ed.get("spectrum"), dict) else {}
    eligibility = ed.get("eligibility", {}) if isinstance(ed.get("eligibility"), dict) else {}
    symmetry = summary.get("model_construction_annotations", {}).get("symmetry", {}) if isinstance(summary.get("model_construction_annotations"), dict) else {}
    precheck = symmetry.get("precheck", {}) if isinstance(symmetry.get("precheck"), dict) else {}
    dimension_report = precheck.get("hilbert_space_dimension", {}) if isinstance(precheck.get("hilbert_space_dimension"), dict) else {}
    nnz_values: List[int] = []
    _collect_nnz_values(ed_spectrum.get("memory_diagnostics"), nnz_values)
    _collect_nnz_values(ed_spectrum, nnz_values)

    full_dim = _first_nested(
        {"ed": ed, "eligibility": eligibility, "dimension_report": dimension_report, "spectrum": ed_spectrum},
        (
            ("ed", "full_hilbert_dimension"),
            ("eligibility", "full_hilbert_dimension"),
            ("dimension_report", "full_hilbert_dimension"),
            ("spectrum", "full_spin_orbital_hilbert_dim"),
        ),
    )
    effective_dim = _first_nested(
        {"ed": ed, "eligibility": eligibility, "spectrum": ed_spectrum},
        (
            ("ed", "effective_hilbert_dimension"),
            ("ed", "hilbert_dimension"),
            ("eligibility", "effective_hilbert_dimension"),
            ("eligibility", "actual_hilbert_dimension"),
            ("spectrum", "hilbert_dimension"),
            ("spectrum", "hilbert_dim"),
        ),
    )
    projected_dim = _first_nested(
        {"ed": ed, "spectrum": ed_spectrum},
        (
            ("ed", "projector_reduced_dimension"),
            ("spectrum", "projector_reduced_dimension"),
            ("spectrum", "projector_solver_dimension"),
        ),
    )
    u1_parent_dim = _first_nested(
        {"ed": ed, "eligibility": eligibility, "spectrum": ed_spectrum},
        (
            ("ed", "u1_parent_hilbert_dimension"),
            ("eligibility", "standard_u1_parent_hilbert_dimension"),
            ("spectrum", "u1_basis_dimension"),
        ),
    )
    return {
        "requested_backend": ed.get("requested_backend", parameters.get("ed_backend")),
        "actual_backend": ed.get("actual_backend", ed.get("ed_backend")),
        "requested_backend_main": parameters.get("backend"),
        "actual_backend_main": summary.get("backend_used"),
        "symmetry_engine": ed.get("symmetry_engine"),
        "requested_symmetry_engine": ed.get("requested_symmetry_engine"),
        "accepted_symmetries": ed.get("accepted_symmetries"),
        "dropped_symmetries": ed.get("dropped_symmetries"),
        "applied_reductions": ed.get("applied_reductions"),
        "hilbert_dimensions": {
            "full_hilbert_dimension": int(full_dim) if full_dim is not None else None,
            "effective_hilbert_dimension": int(effective_dim) if effective_dim is not None else None,
            "u1_parent_hilbert_dimension": int(u1_parent_dim) if u1_parent_dim is not None else None,
        },
        "projected_dimensions": {
            "projector_reduced_dimension": int(projected_dim) if projected_dim is not None else None,
        },
        "sparse_nnz": {
            "max_observed_nnz": max(nnz_values) if nnz_values else None,
            "observed_nnz_values": nnz_values,
        },
        "estimated_dense_memory_mb": {
            "full_hilbert_matrix_complex128": _estimated_dense_memory_mb(full_dim),
            "effective_hilbert_matrix_complex128": _estimated_dense_memory_mb(effective_dim),
            "projected_hilbert_matrix_complex128": _estimated_dense_memory_mb(projected_dim),
        },
    }


def _profile_cprofile_stats() -> str | None:
    profiler = _PROFILE_STATE.get("cprofile_profiler")
    if profiler is None:
        return _PROFILE_STATE.get("cprofile_stats")
    try:
        profiler.disable()
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
        stats.print_stats(30)
        text = stream.getvalue()
        _PROFILE_STATE["cprofile_stats"] = text
        return text
    except Exception as exc:
        text = f"cProfile stats unavailable: {exc}"
        _PROFILE_STATE["cprofile_stats"] = text
        return text


def build_profiling_summary(summary: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if not profiling_enabled():
        return {}
    now = time.perf_counter()
    run_start = _PROFILE_STATE.get("run_start")
    run_end = _PROFILE_STATE.get("run_end") or now
    if run_start is None:
        run_start = now
    if bool(_PROFILE_STATE.get("memory", True)) and tracemalloc.is_tracing():
        try:
            current, peak = tracemalloc.get_traced_memory()
            _PROFILE_STATE["tracemalloc_current_mb"] = float(current) / float(1024 * 1024)
            _PROFILE_STATE["tracemalloc_peak_mb"] = float(peak) / float(1024 * 1024)
        except Exception:
            pass
    _PROFILE_STATE["resource_peak_rss_mb"] = _profile_resource_peak_rss_mb()
    summary_metadata = profile_metadata_from_summary(summary)
    metadata = dict(_PROFILE_STATE.get("metadata", {}))
    metadata["summary"] = summary_metadata
    stage_events = list(_PROFILE_STATE.get("stage_events", []))
    scan_events = list(_PROFILE_STATE.get("scan_point_events", []))
    output: Dict[str, Any] = {
        "enabled": True,
        "wall_time_seconds": float(run_end - float(run_start)),
        "metadata": metadata,
        "stage_timing": {
            "events": stage_events,
            "table": _profile_timing_table(stage_events),
        },
        "scan_point_timing": _profile_scan_point_summary(scan_events),
        "memory": {
            "tracemalloc_current_mb": _PROFILE_STATE.get("tracemalloc_current_mb"),
            "tracemalloc_peak_mb": _PROFILE_STATE.get("tracemalloc_peak_mb"),
            "resource_peak_rss_mb": _PROFILE_STATE.get("resource_peak_rss_mb"),
        },
    }
    if bool(_PROFILE_STATE.get("cprofile", False)):
        output["cprofile"] = {
            "enabled": True,
            "sort": "cumulative",
            "top_entries": 30,
            "stats_text": _profile_cprofile_stats(),
        }
    return output


def write_profiling_summary(profiling_summary: Dict[str, Any], output_folder: str | None = None) -> str | None:
    if not profiling_enabled() or not bool(_PROFILE_STATE.get("output_json", True)):
        return None
    folder = str(output_folder or _PROFILE_STATE.get("output_folder") or PROFILE_OUTPUT_FOLDER)
    try:
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, "profile_summary.json")
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(_profile_json_safe(profiling_summary), file, indent=2, sort_keys=True)
        return filepath
    except Exception as exc:
        warnings.warn(f"Failed to write profiling summary JSON: {exc}")
        return None


def finalize_profiling(
    summary: Dict[str, Any] | None = None,
    output_folder: str | None = None,
    include_environment_audit: bool = True,
) -> Dict[str, Any]:
    if not profiling_enabled():
        return {}
    if _PROFILE_STATE.get("run_end") is None:
        _PROFILE_STATE["run_end"] = time.perf_counter()
    profiling_summary = build_profiling_summary(summary)
    if bool(include_environment_audit):
        audit_folder = str(output_folder or _PROFILE_STATE.get("output_folder") or PROFILE_OUTPUT_FOLDER)
        profiling_summary["environment_audit"] = _profile_environment_audit(audit_folder)
    write_profiling_summary(profiling_summary, output_folder=output_folder)
    _PROFILE_STATE["finalized"] = True
    return profiling_summary


_TENAX_API: Dict[str, Any] | None = None
_TENAX_COMPAT_WARNED = False


def get_tenax_api() -> Dict[str, Any]:
    """Lazy import so --help still works even if Tenax is not installed."""
    global _TENAX_API
    if _TENAX_API is not None:
        return _TENAX_API

    try:
        import tenax  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Tenax is required for DMRG. Install it in your active environment, then rerun."
        ) from exc

    global _TENAX_COMPAT_WARNED
    tenax_version = str(getattr(tenax, "__version__", "0.0.0"))
    numpy_major = int(str(np.__version__).split(".")[0])
    # Known risky combination:
    # Tenax 0.2.x + NumPy >= 2 can fail on complex AutoMPO terms if Tenax is unpatched.
    if tenax_version.startswith("0.2.") and numpy_major >= 2 and not _TENAX_COMPAT_WARNED:
        print(
            "[backend] Warning: Tenax 0.2.x with NumPy >=2 may fail on complex AutoMPO terms "
            "(common error: complex128 -> float64 casting)."
        )
        _TENAX_COMPAT_WARNED = True

    for required in ("DMRGConfig", "build_random_mps", "dmrg"):
        if not hasattr(tenax, required):
            raise RuntimeError(f"Tenax is missing required API '{required}'.")

    _TENAX_API = {
        "DMRGConfig": tenax.DMRGConfig,
        "build_random_mps": tenax.build_random_mps,
        "dmrg": tenax.dmrg,
        "iDMRGConfig": getattr(tenax, "iDMRGConfig", None),
        "idmrg": getattr(tenax, "idmrg", None),
        "build_auto_mpo": getattr(tenax, "build_auto_mpo", None),
        "AutoMPO": getattr(tenax, "AutoMPO", None),
        "expectation_value": getattr(tenax, "expectation_value", None),
        "correlation": getattr(tenax, "correlation", None),
        "expectation": getattr(tenax, "expectation", None),
    }
    return _TENAX_API


_TQDM_IMPORT_FAILED_WARNED = False


def _get_tqdm(enabled: bool):
    global _TQDM_IMPORT_FAILED_WARNED
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm  # type: ignore
        return tqdm
    except Exception:
        if not _TQDM_IMPORT_FAILED_WARNED:
            print("[progress] tqdm is not installed; continuing without progress bars.")
            _TQDM_IMPORT_FAILED_WARNED = True
        return None


def _make_progress_bar(
    enabled: bool,
    total: int,
    desc: str,
    unit: str,
    leave: bool = False,
):
    tqdm = _get_tqdm(enabled)
    if tqdm is None:
        return None
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        leave=leave,
        file=sys.stdout,
        mininterval=0.5,
        smoothing=0.1,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )


def _start_stage(name: str, enabled: bool) -> float:
    if enabled:
        print(f"[stage] {name} started")
    return time.perf_counter()


def _end_stage(name: str, stage_start: float, enabled: bool) -> float:
    stage_end = time.perf_counter()
    elapsed = stage_end - stage_start
    _record_profile_stage(name, stage_start, stage_end)
    if enabled:
        print(f"[stage] {name} finished in {elapsed:.2f}s")
    return elapsed


def _validation_global_one_site_operator(model_spec: Any, n_sites: int, op_name: str):
    from ed_backend import build_global_operator_cache_for_model, kron_all
    import scipy.sparse as sparse

    op_cache = build_global_operator_cache_for_model(model_spec)
    local_op = op_cache[str(op_name)]
    ident = op_cache["Id"]
    dim = int(ident.shape[0]) ** int(n_sites)
    total = sparse.csr_matrix((dim, dim), dtype=complex)
    for site in range(int(n_sites)):
        pieces = [ident] * int(n_sites)
        pieces[site] = local_op
        total = total + kron_all(pieces)
    return total.tocsr()


def _validation_global_product_operator(model_spec: Any, n_sites: int, local_matrix: np.ndarray):
    from ed_backend import build_global_operator_cache_for_model, kron_all
    import scipy.sparse as sparse

    build_global_operator_cache_for_model(model_spec)
    local = sparse.csr_matrix(np.asarray(local_matrix, dtype=np.complex128))
    return kron_all([local for _ in range(int(n_sites))]).tocsr()


def _validation_commutator_relative_norm(left: Any, right: Any) -> float:
    import scipy.sparse as sparse

    commutator = left @ right - right @ left
    numerator = float(sparse.linalg.norm(commutator))
    denominator = max(1.0, float(sparse.linalg.norm(left)) * float(sparse.linalg.norm(right)))
    return numerator / denominator


def _yao_lee_validation_field_case(case: str) -> Tuple[str, Tuple[float, float, float]]:
    case_key = str(case).strip().lower()
    if case_key == "none":
        return "off", (0.0, 0.0, 0.0)
    if case_key == "hz":
        return "hamiltonian", (0.0, 0.0, 1.0)
    if case_key == "hx":
        return "hamiltonian", (1.0, 0.0, 0.0)
    if case_key == "hy":
        return "hamiltonian", (0.0, 1.0, 0.0)
    if case_key == "h111":
        component = 1.0 / math.sqrt(3.0)
        return "hamiltonian", (component, component, component)
    if case_key == "generic":
        return "hamiltonian", (1.0, 0.5, 0.25)
    if case_key == "perturbation":
        return "perturbation", (1.0, 0.5, 0.25)
    raise ValueError(f"Unknown Yao-Lee validation field case '{case}'.")


def _yao_lee_validation_expected_rotation_conservation(case: str) -> Dict[str, bool]:
    case_key = str(case).strip().lower()
    if case_key in ("none", "perturbation"):
        return {"Rx_pi": True, "Ry_pi": True, "Rz_pi": True}
    if case_key == "hx":
        return {"Rx_pi": True, "Ry_pi": False, "Rz_pi": False}
    if case_key == "hy":
        return {"Rx_pi": False, "Ry_pi": True, "Rz_pi": False}
    if case_key == "hz":
        return {"Rx_pi": False, "Ry_pi": False, "Rz_pi": True}
    return {"Rx_pi": False, "Ry_pi": False, "Rz_pi": False}


def _yao_lee_tz_sector_energy_check(
    geometry: Any,
    model_spec: Any,
    *,
    alpha: float,
    beta: float,
    coupling_j: float,
    external_field_terms: List[Tuple[float, str]],
    full_ground_energy: float,
) -> Dict[str, Any]:
    from ed_backend import run_spin_orbital_u1_exact_spectrum

    sector_rows: List[Dict[str, Any]] = []
    n_sites = int(geometry.number_of_sites)
    for target_tz2 in range(-n_sites, n_sites + 1, 2):
        spectrum, _vectors, _basis_list, _basis_map = run_spin_orbital_u1_exact_spectrum(
            geometry=geometry,
            model_spec=model_spec,
            alpha=float(alpha),
            beta=float(beta),
            coupling_j=float(coupling_j),
            eigenstate_count=1,
            check_ground_state_degeneracy=False,
            external_field_terms=external_field_terms,
            show_progress=False,
            sparse_tol=1.0e-11,
            sparse_maxiter=20000,
            use_sz_block=False,
            target_sz2=0,
            use_tau_z_block=True,
            target_tz2=int(target_tz2),
        )
        sector_rows.append(
            {
                "target_tz2": int(target_tz2),
                "energy": float(spectrum["ground_state_energy"]),
                "hilbert_dim": int(spectrum["hilbert_dim"]),
            }
        )
    minimum_sector = min(sector_rows, key=lambda row: float(row["energy"]))
    difference = abs(float(full_ground_energy) - float(minimum_sector["energy"]))
    return {
        "full_ground_energy": float(full_ground_energy),
        "minimum_tz_sector_energy": float(minimum_sector["energy"]),
        "minimum_tz_sector": int(minimum_sector["target_tz2"]),
        "absolute_difference": float(difference),
        "sector_energies": sector_rows,
    }


def validate_yao_lee_pure_field_111_ed_dmrg(
    *,
    field_strength: float = 1.0,
    length_x: int = 1,
    length_y: int = 2,
    max_bond_dimension: int = 16,
    max_sweeps: int = 16,
    tol: float = 1.0e-8,
    run_dmrg: bool = True,
) -> Dict[str, Any]:
    """Minimal normalized-spin pure-field check for ED and TeNPy DMRG.

    With all interactions off and a normalized [111] Zeeman term
    ``-H/SQRT(3) * (Sx + Sy + Sz)``, a spin-1/2 site has ground energy
    ``-H/2``. The orbital sector is a spectator, so the same value must be
    obtained in the total-Tz sector.
    """
    from ed_backend import run_small_cluster_exact_spectrum
    from models import build_lattice_geometry, build_model_spec, external_field_terms_for_model, external_field_vector

    model_spec = build_model_spec("1/2", "1/2", "yao_lee", "z")
    geometry = build_lattice_geometry(
        "honeycomb",
        length_x=int(length_x),
        length_y=int(length_y),
        circumference_x=False,
        circumference_y=False,
    )
    field_terms = external_field_terms_for_model(
        external_field_vector("111", float(field_strength), 0.0, 0.0, 0.0),
        mu_b=1.0,
        field_sign=-1.0,
        sigma_factor=1.0,
    )
    expected_energy_per_site = -0.5 * float(abs(field_strength))
    ed_spectrum, _vectors = run_small_cluster_exact_spectrum(
        geometry=geometry,
        model_spec=model_spec,
        alpha=0.0,
        beta=0.0,
        coupling_j=0.0,
        eigenstate_count=1,
        check_ground_state_degeneracy=False,
        external_field_terms=field_terms,
        show_progress=False,
        solver="dense",
    )
    n_sites = int(geometry.number_of_sites)
    ed_energy_per_site = float(ed_spectrum["ground_state_energy"]) / float(n_sites)
    result: Dict[str, Any] = {
        "status": "completed",
        "test": "pure_field_111",
        "field_strength": float(field_strength),
        "field_terms": [(float(coefficient), str(op_name)) for coefficient, op_name in field_terms],
        "expected_energy_per_site": float(expected_energy_per_site),
        "ed_energy_per_site": float(ed_energy_per_site),
        "ed_abs_error": float(abs(ed_energy_per_site - expected_energy_per_site)),
        "ed_passed": bool(abs(ed_energy_per_site - expected_energy_per_site) <= float(tol)),
        "geometry": {
            "lattice": "honeycomb",
            "length_x": int(length_x),
            "length_y": int(length_y),
            "number_of_sites": int(n_sites),
            "boundary": "obcX_obcY",
        },
    }
    if not bool(run_dmrg):
        result["dmrg_status"] = "not_requested"
        result["passed"] = bool(result["ed_passed"])
        return result
    try:
        from tenpy_backend import run_cylindrical_dmrg

        _psi, _mpo, dmrg_info = run_cylindrical_dmrg(
            geometry=geometry,
            alpha=0.0,
            beta=0.0,
            coupling_j=0.0,
            max_bond_dimension=int(max_bond_dimension),
            max_sweeps=int(max_sweeps),
            truncation_cutoff=1.0e-12,
            external_field_terms=field_terms,
            symmetry_reductions={
                "use_sz_block": False,
                "use_tau_z_block": True,
                "use_z2_block": False,
                "target_tz2": 0,
                "allow_dense_fallback": True,
            },
            compute_phase_observables=False,
            show_progress=False,
        )
        dmrg_energy_per_site = float(dmrg_info["E"]) / float(n_sites)
        result.update(
            {
                "dmrg_status": "completed",
                "dmrg_energy_per_site": float(dmrg_energy_per_site),
                "dmrg_abs_error": float(abs(dmrg_energy_per_site - expected_energy_per_site)),
                "ed_dmrg_abs_difference": float(abs(ed_energy_per_site - dmrg_energy_per_site)),
                "dmrg_symmetry_mode": str(dmrg_info.get("symmetry_mode", "unknown")),
                "dmrg_passed": bool(abs(dmrg_energy_per_site - expected_energy_per_site) <= float(tol)),
            }
        )
    except Exception as exc:
        result.update(
            {
                "dmrg_status": "failed",
                "dmrg_error": str(exc),
                "dmrg_passed": False,
            }
        )
    result["passed"] = bool(result.get("ed_passed", False) and result.get("dmrg_passed", False))
    return result


def validate_yao_lee_symmetry_case(
    case: str,
    *,
    length_x: int = 1,
    length_y: int = 2,
    alpha: float = 0.7,
    beta: float = 0.2,
    coupling_j: float = 1.0,
    tol: float = 1.0e-12,
) -> Dict[str, Any]:
    """Validate shared Yao-Lee symmetry rules on a tiny explicit ED cluster."""
    import scipy.linalg
    from ed_backend import build_exact_hamiltonian
    from models import (
        build_lattice_geometry,
        build_model_spec,
        build_site_ops,
        classify_external_field,
        external_field_terms_for_model,
        normalize_requested_symmetry_reductions,
    )

    model_spec = build_model_spec("1/2", "1/2", "yao_lee", "z")
    geometry = build_lattice_geometry(
        "honeycomb",
        length_x=int(length_x),
        length_y=int(length_y),
        circumference_x=True,
        circumference_y=True,
    )
    treatment, field_vector = _yao_lee_validation_field_case(case)
    field_terms = (
        external_field_terms_for_model(field_vector, mu_b=1.0, field_sign=-1.0, sigma_factor=1.0)
        if treatment == "hamiltonian"
        else []
    )
    hamiltonian = build_exact_hamiltonian(
        geometry,
        model_spec,
        alpha=float(alpha),
        beta=float(beta),
        coupling_j=float(coupling_j),
        external_field_terms=field_terms,
        show_progress=False,
    ).tocsr()
    full_ground_energy = float(np.linalg.eigvalsh(hamiltonian.toarray())[0])
    energy_check = _yao_lee_tz_sector_energy_check(
        geometry,
        model_spec,
        alpha=float(alpha),
        beta=float(beta),
        coupling_j=float(coupling_j),
        external_field_terms=field_terms,
        full_ground_energy=full_ground_energy,
    )
    n_sites = int(geometry.number_of_sites)
    site_ops = build_site_ops(model_spec)
    totals = {
        "Sz_tot": _validation_global_one_site_operator(model_spec, n_sites, "Sz"),
        "Tz_tot": _validation_global_one_site_operator(model_spec, n_sites, "Tz"),
    }
    rotations = {
        "Rx_pi": _validation_global_product_operator(
            model_spec, n_sites, scipy.linalg.expm(-1.0j * math.pi * site_ops["Sx"])
        ),
        "Ry_pi": _validation_global_product_operator(
            model_spec, n_sites, scipy.linalg.expm(-1.0j * math.pi * site_ops["Sy"])
        ),
        "Rz_pi": _validation_global_product_operator(
            model_spec, n_sites, scipy.linalg.expm(-1.0j * math.pi * site_ops["Sz"])
        ),
    }
    commutators = {
        name: _validation_commutator_relative_norm(hamiltonian, operator)
        for name, operator in {**totals, **rotations}.items()
    }
    expected_rotations = _yao_lee_validation_expected_rotation_conservation(case)
    checks = {
        "Tz_conserved": commutators["Tz_tot"] <= float(tol),
        "Sz_not_conserved": commutators["Sz_tot"] > float(tol),
    }
    for name, expected_conserved in expected_rotations.items():
        observed = commutators[name] <= float(tol)
        checks[f"{name}_{'conserved' if expected_conserved else 'broken'}"] = observed is expected_conserved

    normalized_auto = normalize_requested_symmetry_reductions(
        ["auto"],
        model_spec,
        treatment,
        field_vector,
        backend="validation",
        strict=True,
        allow_dense_fallback=True,
    )
    normalized_sz = normalize_requested_symmetry_reductions(
        ["sz"],
        model_spec,
        treatment,
        field_vector,
        backend="validation",
        strict=True,
        allow_dense_fallback=True,
    )
    normalized_tz_z2 = normalize_requested_symmetry_reductions(
        ["tz", "z2"],
        model_spec,
        treatment,
        field_vector,
        backend="validation",
        strict=True,
        allow_dense_fallback=True,
    )
    checks["auto_selects_tz"] = normalized_auto.get("safe_reductions") == ["tz"]
    checks["sz_is_not_accepted"] = not bool(normalized_sz.get("use_sz_block", False))
    checks["tz_z2_keeps_tz"] = bool(normalized_tz_z2.get("use_tau_z_block", False))
    checks["tz_z2_drops_unimplemented_z2"] = not bool(normalized_tz_z2.get("use_z2_block", False))
    checks["full_ed_matches_minimum_tz_sector"] = energy_check["absolute_difference"] <= 1.0e-8
    optional_backend_checks: Dict[str, Any] = {}

    selected_tz0 = next(
        (row for row in energy_check["sector_energies"] if int(row["target_tz2"]) == 0),
        None,
    )
    if selected_tz0 is not None:
        try:
            import quspin_backend

            quspin_spectrum, _quspin_vectors = quspin_backend.run_small_cluster_exact_spectrum(
                geometry=geometry,
                model_spec=model_spec,
                alpha=float(alpha),
                beta=float(beta),
                coupling_j=float(coupling_j),
                eigenstate_count=1,
                check_ground_state_degeneracy=False,
                external_field_terms=field_terms,
                show_progress=False,
                use_sz_block=False,
                use_tau_z_block=True,
                target_tz2=0,
                use_z2_block=False,
            )
            difference = abs(float(quspin_spectrum["ground_state_energy"]) - float(selected_tz0["energy"]))
            optional_backend_checks["quspin_tz_matches_standard_ed"] = {
                "status": "passed" if difference <= 1.0e-8 else "failed",
                "absolute_difference": float(difference),
            }
            checks["quspin_tz_matches_standard_ed"] = difference <= 1.0e-8
            if not field_terms:
                z2_sector_energies = []
                for parity in (0, 1):
                    z2_spectrum, _z2_vectors = quspin_backend.run_small_cluster_exact_spectrum(
                        geometry=geometry,
                        model_spec=model_spec,
                        alpha=float(alpha),
                        beta=float(beta),
                        coupling_j=float(coupling_j),
                        eigenstate_count=1,
                        check_ground_state_degeneracy=False,
                        external_field_terms=field_terms,
                        show_progress=False,
                        use_sz_block=False,
                        use_tau_z_block=True,
                        target_tz2=0,
                        use_z2_block=True,
                        z2_generator="spin_flip",
                        z2_target_parity=parity,
                    )
                    z2_sector_energies.append(float(z2_spectrum["ground_state_energy"]))
                z2_min_difference = abs(min(z2_sector_energies) - float(quspin_spectrum["ground_state_energy"]))
                optional_backend_checks["quspin_tz_z2_parity_min_matches_tz"] = {
                    "status": "passed" if z2_min_difference <= 1.0e-8 else "failed",
                    "z2_sector_energies": [float(value) for value in z2_sector_energies],
                    "tz_ground_energy": float(quspin_spectrum["ground_state_energy"]),
                    "absolute_difference": float(z2_min_difference),
                }
                checks["quspin_tz_z2_parity_min_matches_tz"] = z2_min_difference <= 1.0e-8
        except Exception as exc:
            optional_backend_checks["quspin_tz_matches_standard_ed"] = {
                "status": "skipped",
                "reason": str(exc),
            }

    try:
        from tenpy_backend import YaoLeeModel

        tenpy_model = YaoLeeModel(
            geometry,
            alpha=float(alpha),
            beta=float(beta),
            coupling_j=float(coupling_j),
            external_field_terms=field_terms,
            symmetry_reductions=normalized_auto,
        )
        tenpy_ok = str(getattr(tenpy_model, "symmetry_mode", "none")) == "u1_tz"
        optional_backend_checks["tenpy_finite_model_real_u1_tz"] = {
            "status": "passed" if tenpy_ok else "failed",
            "symmetry_mode": str(getattr(tenpy_model, "symmetry_mode", "none")),
        }
        checks["tenpy_finite_model_real_u1_tz"] = bool(tenpy_ok)
        tenpy_idmrg_model = YaoLeeModel(
            geometry,
            alpha=float(alpha),
            beta=float(beta),
            coupling_j=float(coupling_j),
            bc_MPS="infinite",
            infinite_x=True,
            external_field_terms=field_terms,
            symmetry_reductions=normalized_auto,
        )
        tenpy_idmrg_ok = str(getattr(tenpy_idmrg_model, "symmetry_mode", "none")) == "u1_tz"
        optional_backend_checks["tenpy_idmrg_model_real_u1_tz"] = {
            "status": "passed" if tenpy_idmrg_ok else "failed",
            "symmetry_mode": str(getattr(tenpy_idmrg_model, "symmetry_mode", "none")),
            "target_sector": getattr(tenpy_idmrg_model, "target_tz2", None),
        }
        checks["tenpy_idmrg_model_real_u1_tz"] = bool(tenpy_idmrg_ok)
    except Exception as exc:
        optional_backend_checks["tenpy_real_u1_tz"] = {
            "status": "skipped",
            "reason": str(exc),
        }

    return {
        "case": str(case),
        "field": classify_external_field(treatment, field_vector),
        "hamiltonian_field_terms": [
            {"coefficient": float(coefficient), "operator": str(op_name)}
            for coefficient, op_name in field_terms
        ],
        "geometry": {
            "lattice": "honeycomb",
            "length_x": int(length_x),
            "length_y": int(length_y),
            "number_of_sites": n_sites,
        },
        "commutator_relative_norms": commutators,
        "normalization": {
            "auto": normalized_auto,
            "sz": normalized_sz,
            "tz_z2": normalized_tz_z2,
        },
        "energy_consistency": energy_check,
        "optional_backend_checks": optional_backend_checks,
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def validate_yao_lee_symmetry_rules(
    cases: Sequence[str] = ("none", "hz", "h111", "generic", "perturbation"),
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run the standard tiny-cluster Yao-Lee symmetry validation suite."""
    results = [validate_yao_lee_symmetry_case(case, **kwargs) for case in cases]
    return {
        "status": "passed" if all(result["passed"] for result in results) else "failed",
        "results": results,
    }


def _geometry_positions_array(geometry: Any) -> np.ndarray:
    if hasattr(geometry, "positions"):
        return np.asarray(geometry.positions, dtype=float)
    if hasattr(geometry, "coordinates"):
        return np.asarray(geometry.coordinates, dtype=float)
    raise AttributeError("Geometry object must provide positions or coordinates.")


def select_geometric_center_site(
    geometry: Any,
    reference_site_idx: int | None = None,
) -> int:
    """Choose the site closest to the geometric center of the finite cluster."""
    n_sites = int(getattr(geometry, "number_of_sites"))
    if n_sites <= 0:
        raise ValueError("Cannot select a reference site for an empty geometry.")
    if reference_site_idx is not None:
        site = int(reference_site_idx)
        if site < 0:
            site = n_sites + site
        if site < 0 or site >= n_sites:
            raise IndexError(f"reference_site_idx={reference_site_idx} is outside [0, {n_sites - 1}].")
        return site

    positions = _geometry_positions_array(geometry)
    if positions.shape[0] != n_sites:
        raise ValueError(
            f"Geometry has number_of_sites={n_sites}, but positions has length {positions.shape[0]}."
        )
    center = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - center[None, :], axis=1)
    return int(np.argmin(distances))


def find_reference_site_idx(
    geometry: Any,
    reference_site_idx: int | None = None,
) -> int:
    """Return the requested reference site, or the site nearest the cluster center."""
    return select_geometric_center_site(geometry, reference_site_idx)


def extract_spin_reference_correlation(
    correlations: Dict[str, np.ndarray],
    reference_site_idx: int,
) -> np.ndarray:
    """Extract C_S[j] = <S_ref . S_j> from component correlation matrices."""
    component_keys = ("Sx_Sx", "Sy_Sy", "Sz_Sz")
    missing = [key for key in component_keys if key not in correlations]
    if missing:
        raise KeyError(f"correlations is missing spin component matrices: {', '.join(missing)}.")

    matrices = [
        np.asarray(correlations[key], dtype=np.complex128)
        for key in component_keys
    ]
    first_shape = matrices[0].shape
    if len(first_shape) != 2 or first_shape[0] != first_shape[1]:
        raise ValueError("Spin correlation component matrices must be square.")
    for key, matrix in zip(component_keys[1:], matrices[1:]):
        if matrix.shape != first_shape:
            raise ValueError(
                f"Spin correlation component '{key}' has shape {matrix.shape}, "
                f"but expected {first_shape}."
            )

    n_sites = int(first_shape[0])
    ref_site = int(reference_site_idx)
    if ref_site < 0:
        ref_site = n_sites + ref_site
    if ref_site < 0 or ref_site >= n_sites:
        raise IndexError(f"reference_site_idx={reference_site_idx} is outside [0, {n_sites - 1}].")

    spin_scalar = matrices[0] + matrices[1] + matrices[2]
    return np.real_if_close(spin_scalar[ref_site, :]).astype(float)


def build_spin_reference_correlation_pattern(
    geometry: Any,
    correlations: Dict[str, np.ndarray],
    reference_site_idx: int | None = None,
) -> Dict[str, Any]:
    """Build the real-space spin reference-site pattern from raw correlations."""
    ref_site = find_reference_site_idx(geometry, reference_site_idx)
    positions = _geometry_positions_array(geometry)
    c_s = extract_spin_reference_correlation(correlations, ref_site)
    return {
        "reference_site_idx": int(ref_site),
        "reference_position": [float(value) for value in positions[ref_site]],
        "C_S": c_s,
        "correlations": {"S": c_s},
        "max_abs_correlation": {"S": float(np.max(np.abs(c_s))) if c_s.size > 0 else 0.0},
        "method": "reference_site_correlation",
        "definition": "C_S[j] = <S_ref . S_j> with ref chosen near the geometric center",
    }


def _scalar_correlation_matrix(
    scalar_correlations: Dict[str, np.ndarray],
    key: str,
    aliases: Sequence[str],
) -> np.ndarray:
    for candidate in (key, *aliases):
        if candidate in scalar_correlations:
            matrix = np.asarray(scalar_correlations[candidate], dtype=np.complex128)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError(f"Scalar correlation '{candidate}' must be a square matrix.")
            return matrix
    alias_text = ", ".join((key, *aliases))
    raise KeyError(f"scalar_correlations must contain one of: {alias_text}.")


def build_reference_site_correlation_patterns(
    geometry: Any,
    scalar_correlations: Dict[str, np.ndarray],
    reference_site_idx: int | None = None,
) -> Dict[str, Any]:
    """Extract real-space pattern rows C[j] = <O_ref O_j>.

    Finite ED/DMRG eigenstates often keep the exact symmetry, so one-point
    order parameters can vanish even in ordered phases. A row of the two-point
    scalar correlation matrix reveals the relative ordering pattern without
    forcing spontaneous symmetry breaking.
    """
    ref_site = select_geometric_center_site(geometry, reference_site_idx)
    positions = _geometry_positions_array(geometry)
    n_sites = int(getattr(geometry, "number_of_sites"))
    if positions.shape[0] != n_sites:
        raise ValueError(
            f"Geometry has number_of_sites={n_sites}, but positions has length {positions.shape[0]}."
        )

    spin_matrix = _scalar_correlation_matrix(scalar_correlations, "S", ("spin_scalar",))
    orbital_matrix = _scalar_correlation_matrix(scalar_correlations, "T", ("orbital_scalar",))
    mixed_matrix: np.ndarray | None = None
    try:
        mixed_matrix = _scalar_correlation_matrix(scalar_correlations, "ST", ("mixed_scalar",))
    except KeyError:
        mixed_matrix = None

    for name, matrix in (("S", spin_matrix), ("T", orbital_matrix)):
        if matrix.shape[0] != n_sites:
            raise ValueError(
                f"Scalar correlation '{name}' has shape {matrix.shape}, but geometry has {n_sites} sites."
            )
    if mixed_matrix is not None and mixed_matrix.shape[0] != n_sites:
        raise ValueError(
            f"Scalar correlation 'ST' has shape {mixed_matrix.shape}, but geometry has {n_sites} sites."
        )

    correlations: Dict[str, np.ndarray] = {
        "S": np.real_if_close(spin_matrix[ref_site, :]).astype(float),
        "T": np.real_if_close(orbital_matrix[ref_site, :]).astype(float),
    }
    if mixed_matrix is not None:
        correlations["ST"] = np.real_if_close(mixed_matrix[ref_site, :]).astype(float)

    max_abs = {
        key: float(np.max(np.abs(values))) if values.size > 0 else 0.0
        for key, values in correlations.items()
    }
    return {
        "reference_site_idx": int(ref_site),
        "reference_position": [float(value) for value in positions[ref_site]],
        "correlations": correlations,
        "max_abs_correlation": max_abs,
        "method": "reference_site_correlation",
        "definition": "C_O[j] = <O_ref O_j> with ref chosen near the geometric center",
    }


def _classical_scalar_correlations(
    spin_vectors: np.ndarray,
    orbital_vectors: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Build scalar S, T, and ST correlation matrices for a classical product state."""
    spin = np.asarray(spin_vectors, dtype=float)
    orbital = np.asarray(orbital_vectors, dtype=float)
    if spin.ndim != 2 or orbital.ndim != 2 or spin.shape != orbital.shape:
        raise ValueError("Classical spin_vectors and orbital_vectors must have matching shape (N, 3).")
    spin_scalar = spin @ spin.T
    orbital_scalar = orbital @ orbital.T
    return {
        "S": spin_scalar,
        "T": orbital_scalar,
        "ST": spin_scalar * orbital_scalar,
    }


def _reference_patterns_or_warning(
    geometry: Any,
    scalar_correlations: Dict[str, np.ndarray],
    reference_site_idx: int | None,
) -> Dict[str, Any]:
    """Return reference-site patterns without failing the whole phase-scan point."""
    try:
        return build_reference_site_correlation_patterns(
            geometry,
            scalar_correlations,
            reference_site_idx=reference_site_idx,
        )
    except Exception as exc:
        return {
            "available": False,
            "warning": f"Failed to extract reference-site patterns: {exc}",
        }


def resolve_low_energy_spectrum(
    eigenvalues: Sequence[float],
    *,
    check_ground_state_degeneracy: bool = True,
    hilbert_dim: int | None = None,
    degeneracy_tolerance_abs: float = GROUND_MANIFOLD_ABS_TOL_DEFAULT,
    degeneracy_tolerance_rel: float = GROUND_MANIFOLD_REL_TOL_DEFAULT,
) -> Dict[str, Any]:
    """Resolve the ground manifold and the first level above it.

    The default tolerance is intentionally strict, using 1e-12 scale, so ED
    gaps are not swallowed into the degeneracy count.
    """

    values = np.sort(np.asarray(eigenvalues, dtype=float))
    if values.size == 0:
        raise ValueError("Cannot resolve an empty low-energy spectrum.")

    e0 = float(values[0])
    raw_second_energy = float(values[1]) if values.size >= 2 else None
    raw_second_gap = (
        float(raw_second_energy - e0)
        if raw_second_energy is not None
        else None
    )

    if not bool(check_ground_state_degeneracy):
        return {
            "ground_state_energy": e0,
            "ground_state_degeneracy_check_enabled": False,
            "ground_state_degeneracy": None,
            "ground_state_degeneracy_tolerance": None,
            "ground_state_degeneracy_is_lower_bound": None,
            "ground_state_degeneracy_status": "not_checked",
            "ground_state_indices": [0],
            "first_excited_index": 1 if values.size >= 2 else None,
            "raw_second_energy": raw_second_energy,
            "raw_second_gap": raw_second_gap,
            "first_excited_energy": raw_second_energy,
            "spectral_gap": raw_second_gap,
            "low_energy_gaps_from_ground": [float(value - e0) for value in values],
            "note": (
                "Ground-state degeneracy check disabled: first_excited_energy is the raw "
                "second ED eigenvalue and may still belong to a degenerate ground manifold."
            ),
        }

    scale = max(1.0, abs(e0))
    degeneracy_tolerance = max(
        float(degeneracy_tolerance_abs),
        float(degeneracy_tolerance_rel) * scale,
    )
    gaps_from_ground = values - e0
    ground_indices = np.flatnonzero(np.abs(gaps_from_ground) <= degeneracy_tolerance)
    if ground_indices.size == 0:
        ground_indices = np.asarray([0], dtype=int)
    above_ground = np.flatnonzero(np.abs(gaps_from_ground) > degeneracy_tolerance)
    first_excited_index = int(above_ground[0]) if above_ground.size > 0 else None
    first_excited_energy = (
        float(values[first_excited_index])
        if first_excited_index is not None
        else None
    )
    spectral_gap = (
        float(first_excited_energy - e0)
        if first_excited_energy is not None
        else None
    )
    returned_count = int(values.size)
    lower_bound = bool(
        above_ground.size == 0
        and hilbert_dim is not None
        and returned_count < int(hilbert_dim)
    )
    return {
        "ground_state_energy": e0,
        "ground_state_degeneracy_check_enabled": True,
        "ground_state_degeneracy": int(ground_indices.size),
        "ground_state_degeneracy_tolerance": float(degeneracy_tolerance),
        "ground_state_degeneracy_absolute_tolerance": float(degeneracy_tolerance_abs),
        "ground_state_degeneracy_relative_tolerance": float(degeneracy_tolerance_rel),
        "ground_state_degeneracy_is_lower_bound": lower_bound,
        "ground_state_degeneracy_status": "lower_bound" if lower_bound else "resolved",
        "ground_state_indices": [int(index) for index in ground_indices],
        "first_excited_index": first_excited_index,
        "raw_second_energy": raw_second_energy,
        "raw_second_gap": raw_second_gap,
        "first_excited_energy": first_excited_energy,
        "spectral_gap": spectral_gap,
        "gap_above_ground_manifold": spectral_gap,
        "low_energy_gaps_from_ground": [float(gap) for gap in gaps_from_ground],
        "note": (
            "Ground-state degeneracy check enabled: first_excited_energy is the first "
            "resolved level above the ground-state manifold using the recorded tolerance."
        ),
    }


def penalty_excited_state_result(
    *,
    status: str,
    reason: str,
    first_excited_energy: float | None = None,
    spectral_gap: float | None = None,
    penalty_weight_used: float | None = None,
    candidate_variance: float | None = None,
    candidate_max_overlap: float | None = None,
) -> Dict[str, Any]:
    return {
        "method": "finite_dmrg_penalty_excited_state",
        "first_excited_energy": (
            float(first_excited_energy)
            if first_excited_energy is not None
            else None
        ),
        "spectral_gap": (
            float(spectral_gap)
            if spectral_gap is not None
            else None
        ),
        "status": str(status),
        "reason": str(reason),
        "penalty_weight_used": (
            float(penalty_weight_used)
            if penalty_weight_used is not None
            else None
        ),
        "candidate_variance": (
            float(candidate_variance)
            if candidate_variance is not None
            else None
        ),
        "candidate_max_overlap": (
            float(candidate_max_overlap)
            if candidate_max_overlap is not None
            else None
        ),
    }


def _canonicalize_and_normalize_mps(mps: Any) -> Any:
    state = mps
    for method_name in ("canonicalize", "right_canonicalize", "left_canonicalize"):
        method = getattr(state, method_name, None)
        if not callable(method):
            continue
        try:
            result = method()
        except TypeError:
            continue
        if result is not None:
            state = result
        break

    normalize = getattr(state, "normalize", None)
    if callable(normalize):
        result = normalize()
        if result is not None:
            state = result
    return state


def _full_mps_overlap(left_mps: Any, right_mps: Any) -> complex:
    overlap = getattr(left_mps, "overlap", None)
    if callable(overlap):
        return complex(overlap(right_mps))
    overlap = getattr(right_mps, "overlap", None)
    if callable(overlap):
        return complex(np.conjugate(overlap(left_mps)))
    inner = getattr(left_mps, "inner", None)
    if callable(inner):
        return complex(inner(right_mps))
    inner = getattr(right_mps, "inner", None)
    if callable(inner):
        return complex(np.conjugate(inner(left_mps)))
    raise RuntimeError(
        "No full-MPS overlap method is available; refusing tensor-wise overlap checks."
    )


def _evaluate_mps_mpo_energy(H_mpo: Any, mps: Any) -> float:
    if hasattr(mps, "expectation_value"):
        value = mps.expectation_value(H_mpo)
        return float(np.real_if_close(value))
    if hasattr(mps, "expectation"):
        value = mps.expectation(H_mpo)
        return float(np.real_if_close(value))
    exp_fn = get_tenax_api().get("expectation")
    if callable(exp_fn):
        value = exp_fn(H_mpo, mps)
        return float(np.real_if_close(value))
    raise RuntimeError(
        "No expectation evaluator found for original-Hamiltonian candidate energy."
    )


def _evaluate_mps_mpo_variance(H_mpo: Any, mps: Any, energy: float) -> float:
    for method_name in ("variance", "energy_variance", "expectation_variance"):
        method = getattr(mps, method_name, None)
        if not callable(method):
            continue
        try:
            value = method(H_mpo)
        except TypeError:
            continue
        return max(0.0, float(np.real_if_close(value)))
    h2_method = getattr(mps, "expectation_value_squared", None)
    if callable(h2_method):
        h2_value = h2_method(H_mpo)
        return max(0.0, float(np.real_if_close(h2_value)) - float(energy) ** 2)
    raise RuntimeError(
        "No MPO variance evaluator is available; refusing to certify a DMRG gap."
    )


def _suggest_penalty_weights(
    ED_gap_hint: float | None,
    penalty_weights: Sequence[float] | None,
) -> List[float]:
    if penalty_weights is not None:
        weights = [float(weight) for weight in penalty_weights if np.isfinite(float(weight))]
        return [weight for weight in weights if weight > 0.0]
    if ED_gap_hint is not None:
        gap = float(ED_gap_hint)
        if np.isfinite(gap) and gap > 0.0:
            return [5.0 * gap, 10.0 * gap, 50.0 * gap]
    return [1.0, 5.0, 10.0, 50.0]


def find_dmrg_excited_state(
    H_mpo: Any,
    ground_mps_list: Sequence[Any],
    E0: float,
    ED_gap_hint: float | None = None,
    penalty_weights: Sequence[float] | None = None,
    overlap_tol: float = 1e-6,
    energy_tol: float = 1e-7,
    variance_tol: float = 1e-7,
    max_attempts: int = 10,
    *,
    required_ground_degeneracy: int | None = None,
    penalty_dmrg_runner: Callable[..., Any] | None = None,
) -> Dict[str, Any]:
    """Analyze a penalty-state finite-DMRG excited-state search."""

    raw_ground_states = [state for state in ground_mps_list if state is not None]
    if required_ground_degeneracy is not None:
        try:
            required_count = int(required_ground_degeneracy)
        except (TypeError, ValueError):
            required_count = 0
        if required_count > 0 and len(raw_ground_states) < required_count:
            return penalty_excited_state_result(
                status="not_found",
                reason="incomplete ground manifold",
            )

    if len(raw_ground_states) == 0:
        return penalty_excited_state_result(
            status="not_found",
            reason="incomplete ground manifold",
        )

    try:
        ground_states = [
            _canonicalize_and_normalize_mps(state)
            for state in raw_ground_states
        ]
    except Exception as exc:
        return penalty_excited_state_result(
            status="not_found",
            reason=f"ground-state MPS canonicalization failed: {exc}",
        )

    weights = _suggest_penalty_weights(ED_gap_hint, penalty_weights)
    if len(weights) == 0:
        return penalty_excited_state_result(
            status="not_found",
            reason="no positive finite penalty weights",
        )

    if penalty_dmrg_runner is None:
        return penalty_excited_state_result(
            status="not_found",
            reason="penalty-state DMRG runner unavailable",
        )

    best_accepted: Dict[str, Any] | None = None
    best_rejected: Dict[str, Any] | None = None
    e0 = float(E0)
    attempts_done = 0
    for weight in weights:
        for attempt_index in range(max(1, int(max_attempts))):
            attempts_done += 1
            try:
                candidate_result = penalty_dmrg_runner(
                    H_mpo=H_mpo,
                    ground_mps_list=ground_states,
                    penalty_weight=float(weight),
                    attempt_index=int(attempt_index),
                )
            except Exception:
                continue

            if isinstance(candidate_result, dict):
                candidate_mps = candidate_result.get("mps")
                if candidate_mps is None:
                    candidate_mps = candidate_result.get("state")
            elif isinstance(candidate_result, (tuple, list)) and len(candidate_result) > 0:
                candidate_mps = candidate_result[0]
            else:
                candidate_mps = candidate_result
            if candidate_mps is None:
                continue

            try:
                candidate_mps = _canonicalize_and_normalize_mps(candidate_mps)
                overlaps = [
                    abs(_full_mps_overlap(ground_state, candidate_mps))
                    for ground_state in ground_states
                ]
                max_overlap = float(max(overlaps)) if overlaps else 0.0
                energy = _evaluate_mps_mpo_energy(H_mpo, candidate_mps)
                variance = _evaluate_mps_mpo_variance(H_mpo, candidate_mps, energy)
            except Exception:
                continue

            record = {
                "energy": float(energy),
                "gap": float(energy - e0),
                "variance": float(variance),
                "max_overlap": float(max_overlap),
                "penalty_weight": float(weight),
            }
            if best_rejected is None or record["energy"] < best_rejected["energy"]:
                best_rejected = record
            if (
                max_overlap < float(overlap_tol)
                and energy > e0 + float(energy_tol)
                and variance < float(variance_tol)
            ):
                if best_accepted is None or energy < best_accepted["energy"]:
                    best_accepted = record

    if best_accepted is None:
        reason = (
            "no distinct low-variance penalty-DMRG candidate"
            if attempts_done > 0
            else "penalty-state DMRG produced no candidates"
        )
        return penalty_excited_state_result(
            status="not_found",
            reason=reason,
            candidate_variance=(
                best_rejected.get("variance") if best_rejected is not None else None
            ),
            candidate_max_overlap=(
                best_rejected.get("max_overlap") if best_rejected is not None else None
            ),
        )

    return penalty_excited_state_result(
        status="found",
        reason="distinct low-variance state found",
        first_excited_energy=float(best_accepted["energy"]),
        spectral_gap=float(best_accepted["gap"]),
        penalty_weight_used=float(best_accepted["penalty_weight"]),
        candidate_variance=float(best_accepted["variance"]),
        candidate_max_overlap=float(best_accepted["max_overlap"]),
    )


def _entropy_from_probabilities(probabilities: np.ndarray, order_n: int) -> float:
    p = np.asarray(probabilities, dtype=float)
    p = p[p > 1e-15]
    if p.size == 0:
        return 0.0
    p = p / np.sum(p)
    if order_n == 1:
        return float(-np.sum(p * np.log(p)))
    return float(np.log(np.sum(p ** order_n)) / (1.0 - float(order_n)))


def _entropy_dict_from_singular_values(singular_values: np.ndarray, orders: Tuple[int, ...]) -> Dict[str, float]:
    sv = np.asarray(singular_values, dtype=float)
    sv = sv[sv > 1e-15]
    if sv.size == 0:
        return {f"S{n}": 0.0 for n in orders}
    probabilities = sv ** 2
    probabilities = probabilities / np.sum(probabilities)
    return {f"S{n}": _entropy_from_probabilities(probabilities, n) for n in orders}


def _summarize_entropy_values(entropies: Dict[str, List[float]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for key, values in entropies.items():
        if len(values) == 0:
            continue
        values_arr = np.asarray(values, dtype=float)
        summary[f"{key}_mean"] = float(np.mean(values_arr))
        summary[f"{key}_max"] = float(np.max(values_arr))
        summary[f"{key}_min"] = float(np.min(values_arr))
    return summary


def _build_entropy_profile(
    method_label: str,
    cuts: List[float],
    total_span: float,
    entropies: Dict[str, List[float]],
    context: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    cuts_norm = [float(cut) / float(total_span) for cut in cuts] if total_span > 0 else [0.0 for _ in cuts]
    profile: Dict[str, Any] = {
        "method": method_label,
        "cuts": [float(cut) for cut in cuts],
        "cuts_normalized": cuts_norm,
        "entropies": {key: [float(value) for value in values] for key, values in entropies.items()},
        "summary": _summarize_entropy_values(entropies),
    }
    if context is not None:
        profile["context"] = context
    return profile


def compute_tenax_finite_mps_entropy_profile(
    mps: Any,
    orders: Tuple[int, ...] = ENTROPY_ORDERS,
    show_progress: bool = True,
) -> Dict[str, Any]:
    mps_work = mps
    if hasattr(mps_work, "compute_singular_values"):
        mps_work = mps_work.compute_singular_values()
    if not hasattr(mps_work, "singular_values"):
        raise RuntimeError("Tenax finite MPS object does not expose singular_values.")

    n_sites = int(mps_work.n_nodes()) if hasattr(mps_work, "n_nodes") else int(len(getattr(mps_work, "tensors", [])))
    n_bonds = max(n_sites - 1, 0)
    entropies = {f"S{n}": [] for n in orders}
    cuts = [float(idx + 1) for idx in range(n_bonds)]

    progress_bar = _make_progress_bar(
        enabled=show_progress,
        total=n_bonds,
        desc="DMRG entropies",
        unit="bond",
        leave=False,
    )
    for bond in range(n_bonds):
        singular_values = mps_work.singular_values[bond]
        if singular_values is None:
            raise RuntimeError(f"Missing singular values on bond {bond}.")
        entropy_values = _entropy_dict_from_singular_values(np.asarray(singular_values), orders)
        for key, value in entropy_values.items():
            entropies[key].append(value)
        if progress_bar is not None:
            progress_bar.update(1)
    if progress_bar is not None:
        progress_bar.close()

    return _build_entropy_profile(
        method_label="DMRG",
        cuts=cuts,
        total_span=float(max(n_sites, 1)),
        entropies=entropies,
        context={"backend": "tenax", "n_sites": n_sites},
    )


def compute_tenpy_finite_mps_entropy_profile(
    psi: Any,
    orders: Tuple[int, ...] = ENTROPY_ORDERS,
    show_progress: bool = True,
) -> Dict[str, Any]:
    if not hasattr(psi, "L") or not hasattr(psi, "entanglement_entropy"):
        raise RuntimeError("TeNPy MPS object does not expose required entropy methods.")

    n_sites = int(psi.L)
    bonds = list(range(1, n_sites))
    entropies = {f"S{n}": [] for n in orders}

    progress_bar = _make_progress_bar(
        enabled=show_progress,
        total=len(orders),
        desc="DMRG entropies",
        unit="order",
        leave=False,
    )
    for order_n in orders:
        values = psi.entanglement_entropy(n=order_n, bonds=bonds)
        entropies[f"S{order_n}"] = [float(value) for value in values]
        if progress_bar is not None:
            progress_bar.update(1)
    if progress_bar is not None:
        progress_bar.close()

    return _build_entropy_profile(
        method_label="DMRG",
        cuts=[float(bond) for bond in bonds],
        total_span=float(max(n_sites, 1)),
        entropies=entropies,
        context={"backend": "tenpy", "n_sites": n_sites},
    )


def compute_ed_entropy_profile_from_state(
    state: np.ndarray,
    n_sites: int,
    local_dim: int,
    orders: Tuple[int, ...] = ENTROPY_ORDERS,
    show_progress: bool = True,
) -> Dict[str, Any]:
    state_vec = np.asarray(state).reshape(-1)
    entropies = {f"S{n}": [] for n in orders}
    cuts = [float(cut) for cut in range(1, n_sites)]

    progress_bar = _make_progress_bar(
        enabled=show_progress,
        total=max(n_sites - 1, 0),
        desc="ED entropies",
        unit="cut",
        leave=False,
    )
    for cut in range(1, n_sites):
        left_dim = int(local_dim ** cut)
        right_dim = int(local_dim ** (n_sites - cut))
        psi_matrix = state_vec.reshape((left_dim, right_dim))
        singular_values = np.linalg.svd(psi_matrix, full_matrices=False, compute_uv=False)
        entropy_values = _entropy_dict_from_singular_values(singular_values, orders)
        for key, value in entropy_values.items():
            entropies[key].append(value)
        if progress_bar is not None:
            progress_bar.update(1)
    if progress_bar is not None:
        progress_bar.close()

    return _build_entropy_profile(
        method_label="ED",
        cuts=cuts,
        total_span=float(max(n_sites, 1)),
        entropies=entropies,
        context={"n_sites": int(n_sites), "local_dim": int(local_dim)},
    )


def compute_tenax_infinite_mps_entropy_profile(
    mps: Any,
    sites_per_idmrg_site: int,
    finite_n_sites: int | None = None,
    orders: Tuple[int, ...] = ENTROPY_ORDERS,
) -> Dict[str, Any]:
    if not hasattr(mps, "singular_values") or not hasattr(mps, "unit_cell_size"):
        raise RuntimeError("Tenax infinite MPS object does not expose singular_values/unit_cell_size.")

    unit_cell_size = int(mps.unit_cell_size)
    if unit_cell_size <= 0:
        raise RuntimeError("Invalid iDMRG unit cell size.")
    original_sites_per_idmrg_site = max(int(sites_per_idmrg_site), 1)

    entropies = {f"S{n}": [] for n in orders}
    for bond in range(unit_cell_size):
        singular_values = np.asarray(mps.singular_values[bond])
        entropy_values = _entropy_dict_from_singular_values(singular_values, orders)
        for key, value in entropy_values.items():
            entropies[key].append(value)

    context: Dict[str, Any] = {
        "unit_cell_size": unit_cell_size,
        "sites_per_idmrg_site": original_sites_per_idmrg_site,
        "unit_cell_original_sites": int(unit_cell_size * original_sites_per_idmrg_site),
    }
    if finite_n_sites is not None:
        finite_site_count = int(finite_n_sites)
        finite_cuts = [
            float(cut)
            for cut in range(original_sites_per_idmrg_site, finite_site_count, original_sites_per_idmrg_site)
        ]
        if finite_site_count > 0 and len(finite_cuts) > 0:
            finite_entropies = {key: [] for key in entropies}
            for cut in finite_cuts:
                bond_index = (int(cut // original_sites_per_idmrg_site) - 1) % unit_cell_size
                for key, values in entropies.items():
                    finite_entropies[key].append(float(values[bond_index]))
            context.update(
                {
                    "finite_n_sites_for_normalized_cuts": finite_site_count,
                    "available_original_cut_spacing": original_sites_per_idmrg_site,
                    "profile_mapping": (
                        "iDMRG bond entropies are repeated over finite-chain cut positions "
                        "that fall between coarse-grained iDMRG sites."
                    ),
                }
            )
            return _build_entropy_profile(
                method_label="iDMRG-x",
                cuts=finite_cuts,
                total_span=float(finite_site_count),
                entropies=finite_entropies,
                context=context,
            )

    cuts = [float(bond + 1) for bond in range(unit_cell_size)]
    return _build_entropy_profile(
        method_label="iDMRG-x",
        cuts=cuts,
        total_span=float(max(unit_cell_size, 1)),
        entropies=entropies,
        context=context,
    )


def _finite_float_or_none(value: Any) -> float | None:
    try:
        number = float(np.real(value))
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def build_zero_temperature_dmrg_reference(
    geometry: Any,
    dmrg_energy: float,
    scalar_correlations: Dict[str, np.ndarray],
    bond_rows: List[Dict[str, Any]],
    structure_factor_rows: List[Dict[str, Any]],
    uniform_observables: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Package finite-DMRG ground-state data as T=0 references for thermal plots."""
    n_sites = int(getattr(geometry, "number_of_sites", 0))
    n_sites_safe = max(1, n_sites)
    bond_list = list(getattr(geometry, "bond_list", []))

    correlations: Dict[str, float] = {"T": 0.0}
    for channel_key, output_key in (
        ("S", "nearest_neighbor_S"),
        ("T", "nearest_neighbor_T"),
        ("ST", "nearest_neighbor_ST"),
    ):
        matrix = scalar_correlations.get(channel_key)
        if matrix is None or len(bond_list) == 0:
            continue
        matrix_array = np.asarray(matrix)
        values = []
        for bond in bond_list:
            value = _finite_float_or_none(matrix_array[int(bond.i), int(bond.j)])
            if value is not None:
                values.append(value)
        if values:
            correlations[output_key] = float(np.mean(np.asarray(values, dtype=float)))

    bond_energy_values = [
        _finite_float_or_none(row.get("O_ij_gamma"))
        for row in bond_rows
        if isinstance(row, dict)
    ]
    bond_energy_values = [value for value in bond_energy_values if value is not None]
    if bond_energy_values:
        correlations["bond_energy_per_site"] = float(np.sum(bond_energy_values) / float(n_sites_safe))

    observables: Dict[str, float] = {
        "T": 0.0,
        "ground_state_energy": float(dmrg_energy),
        "energy_per_site": float(dmrg_energy) / float(n_sites_safe),
    }
    for key in ("spin_z_per_site", "orbital_z_per_site"):
        if uniform_observables and key in uniform_observables:
            value = _finite_float_or_none(uniform_observables.get(key))
            if value is not None:
                observables[key] = value

    structure_rows: List[Dict[str, Any]] = []
    for row in structure_factor_rows:
        if not isinstance(row, dict):
            continue
        ref_row: Dict[str, Any] = {"T": 0.0}
        for key in ("Q_label", "Qx", "Qy", "S(Q)", "T(Q)", "ST(Q)"):
            if key not in row:
                continue
            if key == "Q_label":
                ref_row[key] = row[key]
                continue
            value = _finite_float_or_none(row[key])
            if value is not None:
                ref_row[key] = value
        structure_rows.append(ref_row)

    return {
        "method": "DMRG",
        "temperature": 0.0,
        "note": (
            "Finite-DMRG ground-state reference shown on finite-temperature ED plots; "
            "this is not a finite-temperature DMRG calculation."
        ),
        "observables": observables,
        "correlations": correlations,
        "structure_factors": structure_rows,
    }


# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Phase-scan analysis
# ----------------------------------------------------------------------

DEFAULT_PHASE_CLASSIFIER_THRESHOLDS = {
    "quantum_weak_order": 0.035,
    "classical_weak_order": 0.075,
    "quantum_bond_nematicity": 0.10,
    "classical_bond_nematicity": 0.08,
    "plaquette_flux_target": 1.0,
    "plaquette_flux_tolerance": 0.15,
}


def phase_classifier_thresholds_from_args(args: Any) -> Dict[str, float]:
    """Read classifier thresholds from CLI args when present, otherwise defaults."""
    return {
        "quantum_weak_order": float(
            getattr(args, "phase_scan_quantum_weak_order_threshold", DEFAULT_PHASE_CLASSIFIER_THRESHOLDS["quantum_weak_order"])
        ),
        "classical_weak_order": float(
            getattr(args, "phase_scan_classical_weak_order_threshold", DEFAULT_PHASE_CLASSIFIER_THRESHOLDS["classical_weak_order"])
        ),
        "quantum_bond_nematicity": float(
            getattr(
                args,
                "phase_scan_quantum_nematicity_threshold",
                DEFAULT_PHASE_CLASSIFIER_THRESHOLDS["quantum_bond_nematicity"],
            )
        ),
        "classical_bond_nematicity": float(
            getattr(
                args,
                "phase_scan_classical_nematicity_threshold",
                DEFAULT_PHASE_CLASSIFIER_THRESHOLDS["classical_bond_nematicity"],
            )
        ),
        "plaquette_flux_target": float(
            getattr(
                args,
                "phase_scan_plaquette_flux_target",
                DEFAULT_PHASE_CLASSIFIER_THRESHOLDS["plaquette_flux_target"],
            )
        ),
        "plaquette_flux_tolerance": float(
            getattr(
                args,
                "phase_scan_plaquette_flux_tolerance",
                DEFAULT_PHASE_CLASSIFIER_THRESHOLDS["plaquette_flux_tolerance"],
            )
        ),
    }


def _phase_scan_axis_values(axis_min: float, axis_max: float, points: int) -> List[float]:
    count = int(points)
    if count < 1:
        raise ValueError("Phase-scan grid point counts must be at least 1.")
    if count == 1:
        return [float(axis_min)]
    if float(axis_max) < float(axis_min):
        raise ValueError("Phase-scan axis max must be >= min.")
    return [float(value) for value in np.linspace(float(axis_min), float(axis_max), count)]


def _phase_scan_grid_from_args(args: Any) -> Tuple[List[float], List[float]]:
    return (
        _phase_scan_axis_values(args.phase_scan_alpha_min, args.phase_scan_alpha_max, args.phase_scan_alpha_points),
        _phase_scan_axis_values(args.phase_scan_beta_min, args.phase_scan_beta_max, args.phase_scan_beta_points),
    )


def _random_unit_vectors(rng: np.random.Generator, count: int) -> np.ndarray:
    vectors = rng.normal(size=(int(count), 3))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms <= 1e-15] = 1.0
    return vectors / norms


def _classical_operator_value(op_name: str, site: int, spin_vectors: np.ndarray, orbital_vectors: np.ndarray) -> float:
    axis_map = {"x": 0, "y": 1, "z": 2}
    text = str(op_name)
    if len(text) < 2:
        raise ValueError(f"Unsupported classical operator '{op_name}'.")
    axis = text[-1].lower()
    if axis not in axis_map:
        raise ValueError(f"Unsupported classical operator '{op_name}'.")
    idx = axis_map[axis]
    if text.startswith("S") and not text.startswith("ST"):
        return float(spin_vectors[site, idx])
    if text.startswith("T"):
        return float(orbital_vectors[site, idx])
    if text.startswith("ST"):
        return float(spin_vectors[site, idx] * orbital_vectors[site, idx])
    raise ValueError(f"Unsupported classical operator '{op_name}'.")


def _classical_bond_energy_value(
    bond: Any,
    spin_vectors: np.ndarray,
    orbital_vectors: np.ndarray,
    model_spec: Any,
    alpha: float,
    beta: float,
    coupling_j: float,
    jx: float,
    jy: float,
    jz: float,
) -> float:
    from models import model_terms_for_bond

    if str(getattr(model_spec, "model_family", "")).strip().lower() == "yao_lee":
        axis_map = {"x": 0, "y": 1, "z": 2}
        gamma = str(bond.gamma).strip().lower()
        axis_index = axis_map[gamma]
        spin_dot = float(np.dot(spin_vectors[bond.i], spin_vectors[bond.j]))
        spin_gamma = float(spin_vectors[bond.i, axis_index] * spin_vectors[bond.j, axis_index])
        orbital_dot = float(np.dot(orbital_vectors[bond.i], orbital_vectors[bond.j]))
        return float(
            -float(coupling_j)
            * (
                float(alpha) * spin_dot * orbital_dot
                - float(alpha) * float(beta) * spin_dot
                - 2.0 * spin_gamma * orbital_dot
                + 2.0 * float(beta) * spin_gamma
                - float(beta) * orbital_dot
                + float(beta) * float(beta)
            )
        )

    value = 0.0
    for coefficient, op_name in model_terms_for_bond(
        bond.gamma,
        model_spec,
        alpha,
        beta,
        coupling_j,
        jx=jx,
        jy=jy,
        jz=jz,
    ):
        left = _classical_operator_value(op_name, bond.i, spin_vectors, orbital_vectors)
        right = _classical_operator_value(op_name, bond.j, spin_vectors, orbital_vectors)
        value += float(coefficient) * left * right
    return float(value)


def _classical_total_energy(
    geometry: Any,
    spin_vectors: np.ndarray,
    orbital_vectors: np.ndarray,
    model_spec: Any,
    alpha: float,
    beta: float,
    coupling_j: float,
    jx: float,
    jy: float,
    jz: float,
    external_field_terms: List[Tuple[float, str]] | None = None,
) -> float:
    bond_energy = sum(
        _classical_bond_energy_value(
            bond,
            spin_vectors,
            orbital_vectors,
            model_spec,
            alpha,
            beta,
            coupling_j,
            jx,
            jy,
            jz,
        )
        for bond in geometry.bond_list
    )
    field_energy = 0.0
    if external_field_terms:
        for site in range(int(spin_vectors.shape[0])):
            for coefficient, op_name in external_field_terms:
                field_energy += float(coefficient) * _classical_operator_value(
                    op_name,
                    site,
                    spin_vectors,
                    orbital_vectors,
                )
    return float(bond_energy + field_energy)


def _classical_bond_rows(
    geometry: Any,
    spin_vectors: np.ndarray,
    orbital_vectors: np.ndarray,
    model_spec: Any,
    alpha: float,
    beta: float,
    coupling_j: float,
    jx: float,
    jy: float,
    jz: float,
) -> List[Dict[str, Any]]:
    from models import model_terms_for_bond

    def channel_for_operator(op_name: str) -> str:
        text = str(op_name)
        if text.startswith("ST"):
            return "ST"
        if text.startswith("T"):
            return "T"
        if text.startswith("S"):
            return "S"
        return "total"

    rows: List[Dict[str, Any]] = []
    is_yao_lee_with_orbital = (
        str(getattr(model_spec, "model_family", "")).strip().lower() == "yao_lee"
    )
    axis_map = {"x": 0, "y": 1, "z": 2}
    for bond in geometry.bond_list:
        if is_yao_lee_with_orbital:
            gamma = str(bond.gamma).strip().lower()
            axis_index = axis_map[gamma]
            spin_dot = float(np.dot(spin_vectors[bond.i], spin_vectors[bond.j]))
            spin_gamma = float(spin_vectors[bond.i, axis_index] * spin_vectors[bond.j, axis_index])
            orbital_dot = float(np.dot(orbital_vectors[bond.i], orbital_vectors[bond.j]))
            channel_energies = {
                "ST": -float(coupling_j) * float(alpha) * spin_dot * orbital_dot
                + 2.0 * float(coupling_j) * spin_gamma * orbital_dot,
                "S": float(coupling_j) * float(alpha) * float(beta) * spin_dot
                - 2.0 * float(coupling_j) * float(beta) * spin_gamma,
                "T": float(coupling_j) * float(beta) * orbital_dot,
                "constant": -float(coupling_j) * float(beta) * float(beta),
            }
        else:
            channel_energies: Dict[str, float] = {}
            for coefficient, op_name in model_terms_for_bond(
                bond.gamma,
                model_spec,
                alpha,
                beta,
                coupling_j,
                jx=jx,
                jy=jy,
                jz=jz,
            ):
                left = _classical_operator_value(op_name, bond.i, spin_vectors, orbital_vectors)
                right = _classical_operator_value(op_name, bond.j, spin_vectors, orbital_vectors)
                channel = channel_for_operator(str(op_name))
                channel_energies[channel] = channel_energies.get(channel, 0.0) + float(coefficient) * left * right
        components = [
            {"channel": channel, "energy": float(energy)}
            for channel, energy in channel_energies.items()
        ]
        rows.append(
            {
                "i": int(bond.i),
                "j": int(bond.j),
                "gamma": str(bond.gamma),
                "O_ij_gamma": float(sum(channel_energies.values())),
                "components": components,
                "channel_energies": channel_energies,
            }
        )
    return rows


def _classical_vector_structure_factor(
    vectors: np.ndarray,
    q: np.ndarray,
    positions: np.ndarray,
) -> float:
    phases = np.exp(1.0j * np.dot(np.asarray(positions, dtype=float), np.asarray(q, dtype=float)))
    amplitude = np.sum(vectors * phases[:, None], axis=0)
    return float(np.real(np.vdot(amplitude, amplitude)) / float(max(1, vectors.shape[0])))


def _classical_structure_factor_rows(
    geometry: Any,
    lattice: str,
    spin_vectors: np.ndarray,
    orbital_vectors: np.ndarray,
) -> List[Dict[str, Any]]:
    from models import default_high_symmetry_momenta

    st_vectors = spin_vectors * orbital_vectors
    rows: List[Dict[str, Any]] = []
    for label, q in default_high_symmetry_momenta(lattice).items():
        rows.append(
            {
                "Q_label": str(label),
                "Qx": float(q[0]),
                "Qy": float(q[1]),
                "S(Q)": _classical_vector_structure_factor(spin_vectors, q, geometry.positions),
                "T(Q)": _classical_vector_structure_factor(orbital_vectors, q, geometry.positions),
                "ST(Q)": _classical_vector_structure_factor(st_vectors, q, geometry.positions),
            }
        )
    return rows


def _run_classical_product_ground_state(
    geometry: Any,
    model_spec: Any,
    alpha: float,
    beta: float,
    coupling_j: float,
    jx: float,
    jy: float,
    jz: float,
    *,
    restarts: int,
    sweeps: int,
    initial_temperature: float,
    final_temperature: float,
    initial_step: float,
    final_step: float,
    seed: int,
    external_field_terms: List[Tuple[float, str]] | None = None,
) -> Dict[str, Any]:
    n_sites = int(geometry.number_of_sites)
    restart_count = max(1, int(restarts))
    sweep_count = max(1, int(sweeps))
    spin_length = max(float(getattr(model_spec, "spin_value", 1.0)), 0.0)
    orbital_length = max(float(getattr(model_spec, "orbital_value", 1.0)), 0.0)
    best_energy = float("inf")
    best_spin = None
    best_orbital = None
    rng_master = np.random.default_rng(int(seed))

    for _ in range(restart_count):
        rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
        spin_vectors = spin_length * _random_unit_vectors(rng, n_sites)
        orbital_vectors = orbital_length * _random_unit_vectors(rng, n_sites)
        current_energy = _classical_total_energy(
            geometry,
            spin_vectors,
            orbital_vectors,
            model_spec,
            alpha,
            beta,
            coupling_j,
            jx,
            jy,
            jz,
            external_field_terms,
        )
        local_best_energy = current_energy
        local_best_spin = spin_vectors.copy()
        local_best_orbital = orbital_vectors.copy()

        for sweep in range(sweep_count):
            frac = float(sweep) / float(max(1, sweep_count - 1))
            temperature = float(initial_temperature) * (
                float(final_temperature) / max(float(initial_temperature), 1e-15)
            ) ** frac
            step_size = float(initial_step) * (
                float(final_step) / max(float(initial_step), 1e-15)
            ) ** frac
            for site in rng.permutation(n_sites):
                for target in ("spin", "orbital"):
                    proposed_spin = spin_vectors.copy()
                    proposed_orbital = orbital_vectors.copy()
                    if target == "spin":
                        if spin_length <= 1e-15:
                            continue
                        random_vector = spin_length * _random_unit_vectors(rng, 1)[0]
                        proposal = (1.0 - step_size) * spin_vectors[site] + step_size * random_vector
                        proposed_spin[site] = spin_length * proposal / max(float(np.linalg.norm(proposal)), 1e-15)
                    else:
                        if orbital_length <= 1e-15:
                            continue
                        random_vector = orbital_length * _random_unit_vectors(rng, 1)[0]
                        proposal = (1.0 - step_size) * orbital_vectors[site] + step_size * random_vector
                        proposed_orbital[site] = orbital_length * proposal / max(float(np.linalg.norm(proposal)), 1e-15)
                    proposed_energy = _classical_total_energy(
                        geometry,
                        proposed_spin,
                        proposed_orbital,
                        model_spec,
                        alpha,
                        beta,
                        coupling_j,
                        jx,
                        jy,
                        jz,
                        external_field_terms,
                    )
                    delta = proposed_energy - current_energy
                    accept = delta <= 0.0
                    if (not accept) and temperature > 0.0:
                        accept = rng.random() < float(np.exp(-delta / max(temperature, 1e-15)))
                    if accept:
                        spin_vectors = proposed_spin
                        orbital_vectors = proposed_orbital
                        current_energy = proposed_energy
                        if current_energy < local_best_energy:
                            local_best_energy = current_energy
                            local_best_spin = spin_vectors.copy()
                            local_best_orbital = orbital_vectors.copy()

        if local_best_energy < best_energy:
            best_energy = local_best_energy
            best_spin = local_best_spin
            best_orbital = local_best_orbital

    if best_spin is None or best_orbital is None:
        raise RuntimeError("Classical product minimization did not produce a state.")
    return {
        "energy": float(best_energy),
        "energy_per_site": float(best_energy / float(max(1, n_sites))),
        "spin_vectors": best_spin,
        "orbital_vectors": best_orbital,
    }


def _dominant_structure_channel(structure_rows: List[Dict[str, Any]], channel: str) -> Dict[str, Any]:
    if len(structure_rows) == 0:
        return {"label": "none", "value": 0.0, "family": "none"}
    values = [(str(row["Q_label"]), float(row.get(channel, 0.0))) for row in structure_rows]
    label, value = max(values, key=lambda item: item[1])
    if label.startswith("M"):
        family = "stripy"
    elif label.startswith("K"):
        family = "afm"
    elif label == "Gamma":
        family = "uniform"
    else:
        family = "other"
    return {"label": label, "value": float(value), "family": family}


def _bond_energy_diagnostics(bond_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(bond_rows) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "range": 0.0,
            "nematicity": 0.0,
            "lowest_energy_gamma": "none",
            "by_gamma": {},
        }
    values = np.asarray([float(row["O_ij_gamma"]) for row in bond_rows], dtype=float)
    by_gamma: Dict[str, Dict[str, float]] = {}
    for gamma in sorted({str(row["gamma"]) for row in bond_rows}):
        gamma_values = np.asarray(
            [float(row["O_ij_gamma"]) for row in bond_rows if str(row["gamma"]) == gamma],
            dtype=float,
        )
        by_gamma[gamma] = {
            "mean": float(np.mean(gamma_values)),
            "std": float(np.std(gamma_values)),
            "min": float(np.min(gamma_values)),
            "max": float(np.max(gamma_values)),
        }
    gamma_means = {gamma: item["mean"] for gamma, item in by_gamma.items()}
    lowest_gamma = min(gamma_means.items(), key=lambda item: item[1])[0] if gamma_means else "none"
    mean_range = float(max(gamma_means.values()) - min(gamma_means.values())) if gamma_means else 0.0
    scale = max(float(np.max(np.abs(values))), abs(float(np.mean(values))), 1e-12)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "range": float(np.max(values) - np.min(values)),
        "nematicity": float(mean_range / scale),
        "lowest_energy_gamma": str(lowest_gamma),
        "by_gamma": by_gamma,
    }


def extract_all_plaquette_fluxes(plaquette_flux: Dict[str, Any] | None) -> Dict[str, float]:
    """Extract a JSON-friendly plaquette-index -> W_p map from flux diagnostics."""
    if not isinstance(plaquette_flux, dict):
        return {}
    for map_key in ("all_plaquette_fluxes", "plaquette_flux_map"):
        flux_map = plaquette_flux.get(map_key)
        if isinstance(flux_map, dict):
            output: Dict[str, float] = {}
            for index, value in flux_map.items():
                try:
                    flux_value = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(flux_value):
                    output[str(index)] = flux_value
            if output:
                return output
    plaquette_details = plaquette_flux.get("plaquettes")
    if isinstance(plaquette_details, dict):
        output = {}
        for index, detail in plaquette_details.items():
            if not isinstance(detail, dict):
                continue
            try:
                flux_value = float(detail.get("W_p", detail.get("value")))
            except (TypeError, ValueError):
                continue
            if np.isfinite(flux_value):
                output[str(index)] = flux_value
        return output
    return {}


def _phase_observable_diagnostics(
    structure_rows: List[Dict[str, Any]],
    bond_rows: List[Dict[str, Any]],
    n_sites: int,
    plaquette_flux: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    spin_peak = _dominant_structure_channel(structure_rows, "S(Q)")
    orbital_peak = _dominant_structure_channel(structure_rows, "T(Q)")
    mixed_peak = _dominant_structure_channel(structure_rows, "ST(Q)")
    bond_diag = _bond_energy_diagnostics(bond_rows)
    norm = float(max(1, n_sites))
    diagnostics = {
        "spin_peak": spin_peak,
        "orbital_peak": orbital_peak,
        "mixed_peak": mixed_peak,
        "spin_order_strength": float(spin_peak["value"] / norm),
        "orbital_order_strength": float(orbital_peak["value"] / norm),
        "mixed_order_strength": float(mixed_peak["value"] / norm),
        "bond_energy": bond_diag,
    }
    if isinstance(plaquette_flux, dict):
        diagnostics["plaquette_flux"] = plaquette_flux
        all_fluxes = extract_all_plaquette_fluxes(plaquette_flux)
        if all_fluxes:
            diagnostics["all_plaquette_fluxes"] = all_fluxes
    return diagnostics


def _classify_phase_from_diagnostics(
    diagnostics: Dict[str, Any],
    alpha: float,
    beta: float,
    diagram_kind: str,
    thresholds: Dict[str, float],
) -> str:
    spin_family = str(diagnostics.get("spin_peak", {}).get("family", "none"))
    orbital_family = str(diagnostics.get("orbital_peak", {}).get("family", "none"))
    spin_strength = float(diagnostics.get("spin_order_strength", 0.0))
    orbital_strength = float(diagnostics.get("orbital_order_strength", 0.0))
    nematicity = float(diagnostics.get("bond_energy", {}).get("nematicity", 0.0))
    flux_diag = diagnostics.get("plaquette_flux", {})
    is_quantum = str(diagram_kind) != "classical_product"

    ordered = max(spin_strength, orbital_strength)
    weak_order = ordered < (
        thresholds["quantum_weak_order"]
        if is_quantum
        else thresholds["classical_weak_order"]
    )
    quantum_qsl_methods = (
        "quantum_ed",
        "tenpy_dmrg",
        "tenpy_idmrg",
        "quimb_peps",
        "quimb_ipeps",
        "tenax",
        "tenax_dmrg",
        "quspin",
    )
    flux_is_conserved = False
    if str(diagram_kind) in quantum_qsl_methods:
        try:
            from models import plaquette_flux_close_to_target

            flux_value = (
                flux_diag.get("W_p", flux_diag.get("value"))
                if isinstance(flux_diag, dict)
                else None
            )
            flux_is_conserved = bool(
                isinstance(flux_diag, dict)
                and flux_diag.get("available", False)
                and plaquette_flux_close_to_target(
                    flux_value,
                    target=float(thresholds.get("plaquette_flux_target", 1.0)),
                    tolerance=float(thresholds.get("plaquette_flux_tolerance", 0.15)),
                )
            )
        except Exception:
            flux_is_conserved = False
        if flux_is_conserved and weak_order:
            return "Spin-Orbital Liquid"
    nematic = nematicity > (
        thresholds["quantum_bond_nematicity"]
        if is_quantum
        else thresholds["classical_bond_nematicity"]
    )

    if is_quantum:
        if alpha <= 0.08 and beta >= 0.045 and weak_order:
            # ED/DMRG quantum spin-liquid identification needs the local-flux
            # diagnostic, not only a coordinate-space heuristic. If the flux
            # was measured and is not in the QSL sector, avoid a false QSL tag.
            if str(diagram_kind) in quantum_qsl_methods:
                return "Weak/undetermined"
            return "Spin liquid"
        if alpha <= 0.20 and beta >= 0.06 and nematic:
            return "NP3"
        if nematic or weak_order:
            return "NP1"
    else:
        if alpha <= 0.035 and beta >= 0.02 and weak_order:
            return "Spin liquid"
        if alpha >= 2.0 and beta <= 0.035 and nematic:
            return "NP2"
        if beta <= 0.035 and (nematic or weak_order):
            return "NP1"

    if spin_family == "stripy" or orbital_family == "stripy":
        return "Stripy S / AFO"
    if spin_family in ("afm", "uniform") or orbital_family in ("afm", "uniform"):
        return "AFM / AFO"
    if nematic:
        return "NP1"
    return "Weak/undetermined"


def _phase_scan_symmetry_mode(args: Any) -> str:
    mode = str(getattr(args, "symmetry_mode", "none")).strip().lower()
    aliases = {"u1sz": "u1_sz", "u1-sz": "u1_sz", "u1tz": "u1_tz", "u1-tz": "u1_tz"}
    return aliases.get(mode, mode)


def _phase_scan_reductions(args: Any) -> Tuple[str, ...]:
    value = getattr(args, "symmetry_reductions", None)
    if value is None:
        mode = _phase_scan_symmetry_mode(args)
        if mode == "auto":
            return ("auto",)
        if mode == "u1":
            return ("sz", "tz")
        if mode == "u1_sz":
            return ("sz",)
        if mode == "u1_tz":
            return ("tz",)
        if mode == "z2":
            return ("z2",)
        return ("none",)
    if isinstance(value, (list, tuple, set)):
        raw_items = []
        for item in value:
            raw_items.extend(
                part.strip()
                for part in str(item).replace("+", ",").replace(";", ",").split(",")
                if part.strip()
            )
    else:
        raw_items = [
            item.strip()
            for item in str(value).replace("+", ",").replace(";", ",").split(",")
            if item.strip()
        ]
    aliases = {
        "auto": "auto",
        "none": "none",
        "u1": "u1",
        "u1_sz": "sz",
        "u1-sz": "sz",
        "u1sz": "sz",
        "sz": "sz",
        "spin": "sz",
        "u1_tz": "tz",
        "u1-tz": "tz",
        "u1tz": "tz",
        "tz": "tz",
        "tau_z": "tz",
        "z2": "z2",
        "parity": "z2",
    }
    reductions: List[str] = []
    for raw in raw_items:
        item = aliases.get(str(raw).strip().lower(), str(raw).strip().lower())
        if item == "auto":
            return ("auto",)
        if item == "none":
            return ("none",)
        if item == "u1":
            for reduction in ("sz", "tz"):
                if reduction not in reductions:
                    reductions.append(reduction)
        elif item in ("sz", "tz", "z2") and item not in reductions:
            reductions.append(item)
    return tuple(reductions or ["none"])


def _phase_scan_uses_sz_block(args: Any) -> bool:
    ed_plan = getattr(args, "ed_symmetry_plan", None)
    if isinstance(ed_plan, dict) and ed_plan.get("status") == "resolved":
        return bool(ed_plan.get("use_sz_block", False))
    if hasattr(args, "use_sz_block"):
        return bool(getattr(args, "use_sz_block"))
    reductions = set(_phase_scan_reductions(args))
    return bool("auto" in reductions or "sz" in reductions)


def _phase_scan_uses_tau_z_block(args: Any) -> bool:
    ed_plan = getattr(args, "ed_symmetry_plan", None)
    if isinstance(ed_plan, dict) and ed_plan.get("status") == "resolved":
        return bool(ed_plan.get("use_tau_z_block", False))
    if hasattr(args, "use_tau_z_block"):
        return bool(getattr(args, "use_tau_z_block"))
    reductions = set(_phase_scan_reductions(args))
    return bool("tz" in reductions)


def _sector_dimension_for_spin_half(n_sites: int, target_m2: int) -> int:
    n = int(n_sites)
    numerator = n + int(target_m2)
    if numerator % 2 != 0:
        return 0
    nup = numerator // 2
    if nup < 0 or nup > n:
        return 0
    return int(math.comb(n, nup))


def _phase_scan_spin_orbital_block_dimension(
    n_sites: int,
    use_sz_block: bool,
    target_sz2: int,
    use_tau_z_block: bool,
    target_tz2: int,
) -> int:
    n = int(n_sites)
    spin_dim = _sector_dimension_for_spin_half(n, target_sz2) if bool(use_sz_block) else (1 << n)
    orbital_dim = _sector_dimension_for_spin_half(n, target_tz2) if bool(use_tau_z_block) else (1 << n)
    return int(spin_dim * orbital_dim)


def _phase_scan_ed_plan_requires_standard_projector(ed_plan: Dict[str, Any], model_spec: Any) -> bool:
    if not isinstance(ed_plan, dict) or ed_plan.get("status") != "resolved":
        return False
    effective_engine = str(ed_plan.get("effective_engine", ed_plan.get("engine", "auto"))).strip().lower()
    if effective_engine == "projector":
        effective_engine = "standard_projector"
    if effective_engine != "standard_projector":
        return False
    is_yao_lee_spin_orbital = bool(
        str(getattr(model_spec, "model_family", "")).strip().lower() == "yao_lee"
        and str(getattr(model_spec, "spin_rep", "")).strip() == "1/2"
        and str(getattr(model_spec, "orbital_rep", "")).strip() == "1/2"
    )
    if not is_yao_lee_spin_orbital or not bool(ed_plan.get("use_tau_z_block", False)):
        return False
    projector_z2 = bool(
        ed_plan.get("use_z2_block", False)
        and str(ed_plan.get("z2_kind")) == "spin_pi_z"
    )
    return bool(
        projector_z2
        or ed_plan.get("use_translation_x_block", False)
        or ed_plan.get("use_translation_y_block", False)
        or ed_plan.get("use_c3_block", False)
    )


def _phase_scan_ed_projector_reduction_factor_estimate(ed_plan: Dict[str, Any], geometry: Any) -> int:
    factor = 1
    if bool(ed_plan.get("use_z2_block", False)) and str(ed_plan.get("z2_kind")) == "spin_pi_z":
        factor *= 2
    if bool(ed_plan.get("use_translation_x_block", False)):
        factor *= max(1, int(getattr(geometry, "length_x", 1) or 1))
    if bool(ed_plan.get("use_translation_y_block", False)):
        factor *= max(1, int(getattr(geometry, "length_y", 1) or 1))
    if bool(ed_plan.get("use_c3_block", False)):
        factor *= 3
    return int(max(1, factor))


def _phase_scan_quantum_point(
    geometry: Any,
    model_spec: Any,
    lattice_name: str,
    alpha: float,
    beta: float,
    args: Any,
    hamiltonian_external_field_terms: List[Tuple[float, str]],
    thresholds: Dict[str, float],
    show_progress: bool = True,
) -> Dict[str, Any]:
    from models import all_high_symmetry_structure_factors
    from ed_backend import (
        all_bond_energies,
        build_spin_orbital_scalar_correlations,
        build_spin_orbital_u1_scalar_correlations,
        collect_correlation_matrices_from_ed,
        collect_correlation_matrices_from_spin_orbital_u1_ed,
        plaquette_flux_from_ed_state,
        plaquette_flux_from_spin_orbital_u1_ed_state,
        run_small_cluster_exact_diagonalization,
        run_spin_orbital_projected_exact_spectrum,
        run_spin_orbital_u1_exact_spectrum,
    )

    local_dim = int(model_spec.physical_dim)
    n_sites = int(geometry.number_of_sites)
    full_hilbert_dim = int(local_dim ** n_sites)
    ed_backend_name = str(getattr(args, "ed_backend", "standard")).strip().lower()
    if ed_backend_name == "ed":
        ed_backend_name = "standard"
    use_sz_block = _phase_scan_uses_sz_block(args)
    use_tau_z_block = _phase_scan_uses_tau_z_block(args)
    ed_plan = (
        getattr(args, "ed_symmetry_plan", {})
        if isinstance(getattr(args, "ed_symmetry_plan", None), dict)
        else {}
    )
    requested_ed_backend_name = ed_backend_name
    use_z2_block = bool(ed_plan.get("use_z2_block", getattr(args, "use_z2_block", False)))
    ed_z2_kind = ed_plan.get("z2_kind")
    use_translation_x_block = bool(ed_plan.get("use_translation_x_block", getattr(args, "use_translation_x_block", False)))
    use_translation_y_block = bool(ed_plan.get("use_translation_y_block", getattr(args, "use_translation_y_block", False)))
    use_translation_block = bool(use_translation_x_block or use_translation_y_block)
    use_reflection_block = bool(getattr(args, "use_reflection_block", False))
    requested_translation_block = bool(use_translation_block)
    requested_translation_x_block = bool(use_translation_x_block)
    requested_translation_y_block = bool(use_translation_y_block)
    requested_reflection_block = bool(use_reflection_block)
    reflection_block = int(getattr(args, "reflection_block", 0))
    momentum_x_block = int(getattr(args, "momentum_x_block", 0))
    momentum_y_block = int(getattr(args, "momentum_y_block", 0))
    target_sz2 = int(getattr(args, "u1_target_sz2", 0))
    target_tz2 = int(getattr(args, "u1_target_tz2", 0))
    effective_ed_engine = str(ed_plan.get("effective_engine", ed_plan.get("engine", getattr(args, "ed_symmetry_engine", "auto")))).strip().lower()
    standard_projector_requested_by_plan = _phase_scan_ed_plan_requires_standard_projector(ed_plan, model_spec)
    plan_backend_override_reason = ed_plan.get("backend_override_reason") if isinstance(ed_plan, dict) else None
    ed_backend_override_reason = None
    if effective_ed_engine == "quspin_experimental_c3":
        experimental_report = ed_plan.get("quspin_experimental_c3", {}) if isinstance(ed_plan, dict) else {}
        return {
            "status": "skipped",
            "alpha": float(alpha),
            "beta": float(beta),
            "reason": (
                experimental_report.get("phase_scan_rejection_reason")
                if isinstance(experimental_report, dict) and experimental_report.get("phase_scan_rejection_reason")
                else (
                    "quspin_experimental_c3 is not enabled for phase scans. Pure C3 site maps are not the "
                    "Yao-Lee combined C3; implement user_basis/custom phases or spin-[111] basis encoding "
                    "and validate against standard_projector on N=8 first."
                )
            ),
            "ed_backend": "quspin",
            "requested_backend": requested_ed_backend_name,
            "actual_backend": "quspin",
            "backend_override_reason": (
                ed_plan.get("backend_override_reason") if isinstance(ed_plan, dict) else None
            ),
            "requested_symmetry_engine": str(ed_plan.get("requested_engine", getattr(args, "ed_symmetry_engine", "auto"))),
            "symmetry_engine": "quspin_experimental_c3",
            "requested_symmetries": list(ed_plan.get("requested_symmetries", [])) if isinstance(ed_plan, dict) else [],
            "accepted_symmetries": list(ed_plan.get("accepted_symmetries", [])) if isinstance(ed_plan, dict) else [],
            "dropped_symmetries": list(ed_plan.get("dropped_symmetries", [])) if isinstance(ed_plan, dict) else [],
            "symmetry_reasons": dict(ed_plan.get("reasons", {})) if isinstance(ed_plan, dict) else {},
            "basis_type": "quspin_experimental_c3_unavailable",
            "effective_hilbert_dimension": 0,
            "full_hilbert_dimension": int(full_hilbert_dim),
            "quspin_experimental_c3": experimental_report,
        }
    if effective_ed_engine == "standard_projector" and ed_backend_name == "quspin":
        ed_backend_name = "standard"
        ed_backend_override_reason = (
            str(plan_backend_override_reason)
            if plan_backend_override_reason
            else "Quantum phase-scan ED used the standard projector/U1 route selected by the ED symmetry engine."
        )
    elif effective_ed_engine.startswith("quspin") and ed_backend_name != "quspin":
        ed_backend_name = "quspin"
        ed_backend_override_reason = (
            f"Quantum phase-scan ED used {effective_ed_engine} as a separate native QuSpin path "
            "for its supported subset."
        )
    ed_route_metadata = {
        "requested_backend": requested_ed_backend_name,
        "actual_backend": ed_backend_name,
        "backend_override_reason": ed_backend_override_reason,
        "requested_symmetry_engine": str(ed_plan.get("requested_engine", getattr(args, "ed_symmetry_engine", "auto")))
        if isinstance(ed_plan, dict)
        else str(getattr(args, "ed_symmetry_engine", "auto")),
        "symmetry_engine": effective_ed_engine,
        "requested_symmetries": list(ed_plan.get("requested_symmetries", [])) if isinstance(ed_plan, dict) else [],
        "accepted_symmetries": list(ed_plan.get("accepted_symmetries", [])) if isinstance(ed_plan, dict) else [],
        "dropped_symmetries": list(ed_plan.get("dropped_symmetries", [])) if isinstance(ed_plan, dict) else [],
        "symmetry_reasons": dict(ed_plan.get("reasons", {})) if isinstance(ed_plan, dict) else {},
        "z2_generator_used": ed_plan.get("z2_generator_used") if isinstance(ed_plan, dict) else None,
        "z2_selection_reason": ed_plan.get("z2_selection_reason") if isinstance(ed_plan, dict) else None,
        "quspin_z2_selection_reason": ed_plan.get("quspin_z2_selection_reason") if isinstance(ed_plan, dict) else None,
    }
    field_ops = {
        str(op_name)
        for coefficient, op_name in list(hamiltonian_external_field_terms or [])
        if abs(float(coefficient)) > 1e-14
    }
    model_requested_reductions = {
        str(item).strip().lower()
        for item in (
            getattr(args, "model_symmetry_selection", {}) or {}
        ).get("requested_reductions", [])
    } if isinstance(getattr(args, "model_symmetry_selection", None), dict) else set()
    if ed_backend_name == "quspin" and "z2" in model_requested_reductions and not field_ops and not ed_plan:
        use_z2_block = True
    if ed_backend_name == "quspin":
        if bool(use_sz_block) and bool(field_ops.intersection({"Sx", "Sy"})):
            use_sz_block = False
            use_z2_block = False
        if bool(use_z2_block) and bool(field_ops.intersection({"Sx", "Sy", "Sz"})):
            use_z2_block = False
        if bool(use_z2_block) and ed_z2_kind == "spin_pi_z":
            use_z2_block = False
        quspin_package_available = importlib.util.find_spec("quspin") is not None
        quspin_translation_x_reason = None
        quspin_translation_y_reason = None
        quspin_translation_support_report = None
        if use_translation_block:
            if not quspin_package_available:
                reason = "QuSpin package is not installed, so translation blocks cannot be checked."
                use_translation_x_block = False
                use_translation_y_block = False
                quspin_translation_x_reason = reason if requested_translation_x_block else None
                quspin_translation_y_reason = reason if requested_translation_y_block else None
            else:
                try:
                    import quspin_backend as quspin_validation_backend

                    support = quspin_validation_backend.quspin_translation_block_support(geometry)
                    quspin_translation_support_report = support
                    x_support = support.get("x", {})
                    y_support = support.get("y", {})
                    use_translation_x_block = bool(
                        requested_translation_x_block and x_support.get("supported", False)
                    )
                    use_translation_y_block = bool(
                        requested_translation_y_block and y_support.get("supported", False)
                    )
                    quspin_translation_x_reason = x_support.get("reason") if requested_translation_x_block else None
                    quspin_translation_y_reason = y_support.get("reason") if requested_translation_y_block else None
                except Exception as exc:
                    reason = str(exc)
                    use_translation_x_block = False
                    use_translation_y_block = False
                    quspin_translation_x_reason = reason if requested_translation_x_block else None
                    quspin_translation_y_reason = reason if requested_translation_y_block else None
        use_translation_block = bool(use_translation_x_block or use_translation_y_block)
        if bool(use_tau_z_block) and bool(use_translation_block):
            if use_translation_x_block:
                quspin_translation_x_reason = (
                    "Dropped for QuSpin native ED: translations must act on spin and orbital together as "
                    "one fused physical-site operation; the current tensor-basis path keeps Tz instead."
                )
            if use_translation_y_block:
                quspin_translation_y_reason = (
                    "Dropped for QuSpin native ED: translations must act on spin and orbital together as "
                    "one fused physical-site operation; the current tensor-basis path keeps Tz instead."
                )
            use_translation_x_block = False
            use_translation_y_block = False
            use_translation_block = False
        quspin_translation_reason = {
            "x": quspin_translation_x_reason,
            "y": quspin_translation_y_reason,
        }
        quspin_reflection_reason = None
        if requested_reflection_block or reflection_block != 0:
            quspin_reflection_reason = (
                "QuSpin reflection/C3 blocks are not applied for the bond-directional Yao-Lee Hamiltonian; "
                "they can permute x/y/z bond types unless a gauge map is implemented."
            )
        use_reflection_block = False
        reflection_block = 0
        compatible = (
            quspin_package_available
            and
            str(getattr(model_spec, "spin_rep", "")) == "1/2"
            and str(getattr(model_spec, "orbital_rep", "")) == "1/2"
            and str(getattr(model_spec, "model_family", "")) == "yao_lee"
            and str(getattr(model_spec, "ising_axis", "")) == "z"
            and _phase_scan_spin_orbital_block_dimension(
                n_sites,
                use_sz_block,
                target_sz2,
                use_tau_z_block,
                target_tz2,
            ) > 0
        )
        hilbert_dim = _phase_scan_spin_orbital_block_dimension(
            n_sites,
            use_sz_block,
            target_sz2,
            use_tau_z_block,
            target_tz2,
        )
        basis_type = (
            "quspin_tensor_spin_z2_orbital_tz"
            if use_z2_block and use_tau_z_block
            else (
                "quspin_tensor_spin_z2_orbital_full"
                if use_z2_block
                else (
                    "quspin_tensor_"
                    f"spin_{'u1_block' if use_sz_block else 'full'}_"
                    f"orbital_{'u1_block' if use_tau_z_block else 'full'}"
                )
            )
        )
        pre_quspin_hilbert_dim = int(hilbert_dim)
        quspin_basis_build_reason = None
        if compatible:
            try:
                import quspin_backend as quspin_basis_backend

                preflight_basis = quspin_basis_backend.build_quspin_yao_lee_basis(
                    n_sites,
                    geometry=geometry,
                    use_sz_block=use_sz_block,
                    target_sz2=target_sz2,
                    use_tau_z_block=use_tau_z_block,
                    target_tz2=target_tz2,
                    use_z2_block=use_z2_block,
                    z2_generator="spin_flip" if use_z2_block else None,
                    z2_target_parity=int(getattr(args, "z2_target_parity", 0)),
                    use_translation_block=use_translation_block,
                    use_translation_x_block=use_translation_x_block,
                    use_translation_y_block=use_translation_y_block,
                    momentum_block_1=momentum_x_block,
                    momentum_block_2=momentum_y_block,
                    momentum_x_block=momentum_x_block,
                    momentum_y_block=momentum_y_block,
                    use_reflection_block=False,
                    reflection_block=0,
                )
                hilbert_dim = int(preflight_basis.Ns)
            except Exception as exc:
                compatible = False
                quspin_basis_build_reason = f"Failed to build the requested QuSpin reduced basis: {exc}"
        if not compatible:
            return {
                "status": "skipped",
                "alpha": float(alpha),
                "beta": float(beta),
                "reason": (
                    quspin_basis_build_reason
                    if quspin_basis_build_reason is not None
                    else
                    "QuSpin ED phase scan requires reachable shared U1 target sectors, "
                    "the quspin Python package, "
                    "spin_rep=1/2, orbital_rep=1/2, model_family=yao_lee, "
                    "and ising_axis=z."
                    " Reflection/C3 blocks are forbidden; Tz can be combined with spin-flip Z2, "
                    "but not with the current 2D translation blocks."
                ),
                "ed_backend": "quspin",
                **ed_route_metadata,
                "basis_type": basis_type,
                "effective_hilbert_dimension": int(hilbert_dim),
                "pre_quspin_hilbert_dimension_estimate": int(pre_quspin_hilbert_dim),
                "full_hilbert_dimension": int(full_hilbert_dim),
                "use_sz_conserved": bool(use_sz_block),
                "use_sz_block": bool(use_sz_block),
                "quspin_package_available": bool(quspin_package_available),
                "quspin_requested_translation_block": bool(requested_translation_block),
                "quspin_requested_translation_x_block": bool(requested_translation_x_block),
                "quspin_requested_translation_y_block": bool(requested_translation_y_block),
                "quspin_use_translation_block": bool(use_translation_block),
                "quspin_use_translation_x_block": bool(use_translation_x_block),
                "quspin_use_translation_y_block": bool(use_translation_y_block),
                "quspin_translation_reason": quspin_translation_reason,
                "quspin_translation_x_reason": quspin_translation_x_reason,
                "quspin_translation_y_reason": quspin_translation_y_reason,
                "quspin_translation_support": quspin_translation_support_report,
                "quspin_requested_reflection_block": bool(requested_reflection_block),
                "quspin_use_reflection_block": False,
                "quspin_reflection_reason": quspin_reflection_reason,
            }
        if n_sites > int(args.phase_scan_ed_max_sites):
            return {
                "status": "skipped",
                "alpha": float(alpha),
                "beta": float(beta),
                "reason": f"Quantum phase scan limited to {int(args.phase_scan_ed_max_sites)} sites.",
                "ed_backend": "quspin",
                **ed_route_metadata,
                "basis_type": basis_type,
                "effective_hilbert_dimension": int(hilbert_dim),
                "pre_quspin_hilbert_dimension_estimate": int(pre_quspin_hilbert_dim),
                "full_hilbert_dimension": int(full_hilbert_dim),
                "use_sz_conserved": bool(use_sz_block),
                "use_sz_block": bool(use_sz_block),
            }
        if hilbert_dim > int(args.phase_scan_ed_max_hilbert_dim):
            return {
                "status": "skipped",
                "alpha": float(alpha),
                "beta": float(beta),
                "reason": (
                    f"Quantum phase scan {basis_type} Hilbert dimension {hilbert_dim} exceeds "
                    f"{int(args.phase_scan_ed_max_hilbert_dim)}."
                ),
                "ed_backend": "quspin",
                **ed_route_metadata,
                "basis_type": basis_type,
                "effective_hilbert_dimension": int(hilbert_dim),
                "pre_quspin_hilbert_dimension_estimate": int(pre_quspin_hilbert_dim),
                "full_hilbert_dimension": int(full_hilbert_dim),
                "use_sz_conserved": bool(use_sz_block),
                "use_sz_block": bool(use_sz_block),
            }

        import quspin_backend as quspin_ed_backend

        spectrum, vectors = quspin_ed_backend.run_small_cluster_exact_spectrum(
            geometry=geometry,
            model_spec=model_spec,
            alpha=alpha,
            beta=beta,
            coupling_j=args.coupling_j,
            eigenstate_count=max(1, int(getattr(args, "ed_max_eigenstates", 2))),
            check_ground_state_degeneracy=False,
            jx=args.jx,
            jy=args.jy,
            jz=args.jz,
            external_field_terms=hamiltonian_external_field_terms,
            show_progress=show_progress,
            solver=getattr(args, "ed_solver", "auto"),
            sparse_tol=float(getattr(args, "ed_sparse_tol", 0.0)),
            sparse_maxiter=(
                int(getattr(args, "ed_sparse_maxiter", 0))
                if int(getattr(args, "ed_sparse_maxiter", 0)) > 0
                else None
            ),
            use_sz_block=use_sz_block,
            target_sz2=target_sz2,
            use_tau_z_block=use_tau_z_block,
            target_tz2=target_tz2,
            use_z2_block=use_z2_block,
            z2_generator="spin_flip" if use_z2_block else None,
            z2_target_parity=int(getattr(args, "z2_target_parity", 0)),
            use_translation_block=use_translation_block,
            use_translation_x_block=use_translation_x_block,
            use_translation_y_block=use_translation_y_block,
            momentum_block_1=momentum_x_block,
            momentum_block_2=momentum_y_block,
            momentum_x_block=momentum_x_block,
            momentum_y_block=momentum_y_block,
            use_reflection_block=use_reflection_block,
            reflection_block=reflection_block,
            check_symm=bool(getattr(args, "quspin_check_symmetries", False)),
            check_herm=bool(getattr(args, "quspin_check_hermiticity", False)),
            check_pcon=bool(getattr(args, "quspin_check_particle_conservation", False)),
        )
        basis_use_sz_block = bool(spectrum.get("use_sz_block", use_sz_block))
        basis_target_sz2 = int(spectrum.get("target_sz2", target_sz2))
        basis_use_z2_block = bool(spectrum.get("use_z2_block", use_z2_block))
        basis = quspin_ed_backend.build_quspin_yao_lee_basis(
            n_sites,
            geometry=geometry,
            use_sz_block=basis_use_sz_block,
            target_sz2=basis_target_sz2,
            use_tau_z_block=use_tau_z_block,
            target_tz2=target_tz2,
            use_z2_block=basis_use_z2_block,
            z2_target_parity=int(getattr(args, "z2_target_parity", 0)),
            use_translation_block=use_translation_block,
            use_translation_x_block=use_translation_x_block,
            use_translation_y_block=use_translation_y_block,
            momentum_block_1=momentum_x_block,
            momentum_block_2=momentum_y_block,
            momentum_x_block=momentum_x_block,
            momentum_y_block=momentum_y_block,
            use_reflection_block=use_reflection_block,
            reflection_block=reflection_block,
        )
        with profile_stage("observables"):
            energy = float(spectrum["ground_state_energy"])
            state = vectors[:, 0]
            scalar_correlations = quspin_ed_backend.build_spin_orbital_scalar_correlations(
                basis,
                state,
                n_sites,
            )
            bond_rows = quspin_ed_backend.all_bond_energies(
                geometry,
                scalar_correlations,
                alpha,
                beta,
                args.coupling_j,
            )
            structure_rows = quspin_ed_backend.all_high_symmetry_structure_factors(
                scalar_correlations,
                geometry,
            )
            plaquette_flux = spectrum.get("plaquette_flux") if isinstance(spectrum, dict) else None
            if plaquette_flux is None:
                try:
                    plaquette_flux = quspin_ed_backend.compute_plaquette_flux(
                        basis,
                        state,
                        geometry,
                        plaquette_center_idx=None,
                    )
                except Exception as exc:
                    plaquette_flux = {"available": False, "warning": str(exc)}
            diagnostics = _phase_observable_diagnostics(
                structure_rows,
                bond_rows,
                geometry.number_of_sites,
                plaquette_flux=plaquette_flux,
            )
            phase_label = _classify_phase_from_diagnostics(diagnostics, alpha, beta, "quantum_ed", thresholds)
        return {
            "status": "completed",
            "alpha": float(alpha),
            "beta": float(beta),
            "ed_backend": "quspin",
            **ed_route_metadata,
            "basis_type": basis_type,
            "effective_hilbert_dimension": int(spectrum.get("hilbert_dimension", hilbert_dim)),
            "pre_quspin_hilbert_dimension_estimate": int(pre_quspin_hilbert_dim),
            "full_hilbert_dimension": int(full_hilbert_dim),
            "use_sz_conserved": bool(use_sz_block),
            "use_sz_block": bool(use_sz_block),
            "use_tau_z_block": bool(use_tau_z_block),
            "use_z2_block": bool(spectrum.get("use_z2_block", use_z2_block)),
            "z2_kind": spectrum.get("z2_kind"),
            "ed_symmetry_plan": ed_plan if ed_plan else None,
            "selected_target_sz2": int(spectrum.get("target_sz2", target_sz2)),
            "quspin_effective_use_z2_block": bool(spectrum.get("use_z2_block", use_z2_block)),
            "quspin_sz_sector_scan": spectrum.get("sz_sector_scan"),
            "quspin_package_available": bool(quspin_package_available),
            "quspin_requested_translation_block": bool(requested_translation_block),
            "quspin_requested_translation_x_block": bool(requested_translation_x_block),
            "quspin_requested_translation_y_block": bool(requested_translation_y_block),
            "quspin_use_translation_block": bool(use_translation_block),
            "quspin_use_translation_x_block": bool(use_translation_x_block),
            "quspin_use_translation_y_block": bool(use_translation_y_block),
            "quspin_translation_reason": quspin_translation_reason,
            "quspin_translation_x_reason": quspin_translation_x_reason,
            "quspin_translation_y_reason": quspin_translation_y_reason,
            "quspin_requested_reflection_block": bool(requested_reflection_block),
            "quspin_use_reflection_block": False,
            "quspin_reflection_reason": quspin_reflection_reason,
            "phase_label": phase_label,
            "energy": float(energy),
            "energy_per_site": float(energy / float(max(1, geometry.number_of_sites))),
            "diagnostics": diagnostics,
            "plaquette_flux": plaquette_flux,
            "all_plaquette_fluxes": extract_all_plaquette_fluxes(plaquette_flux),
            "structure_factors": structure_rows,
            "bond_energies": bond_rows,
        }

    use_sz_conserved_requested = bool(use_sz_block)
    use_sz_conserved = (
        use_sz_conserved_requested
        and str(getattr(model_spec, "spin_rep", "")) == "1/2"
        and str(getattr(model_spec, "orbital_rep", "")) == "1/2"
        and _sector_dimension_for_spin_half(n_sites, target_sz2) > 0
    )
    if use_sz_conserved_requested and not use_sz_conserved and str(getattr(model_spec, "orbital_rep", "")) != "0":
        return {
            "status": "skipped",
            "alpha": float(alpha),
            "beta": float(beta),
            "reason": "Standard Sz-block ED requires a reachable target 2*Sz sector with spin_rep=1/2 and orbital_rep=1/2.",
            "ed_backend": "standard",
            **ed_route_metadata,
            "use_sz_conserved_requested": bool(use_sz_conserved_requested),
        }
    use_tz_conserved = bool(
        use_tau_z_block
        and not use_sz_conserved
        and str(getattr(model_spec, "spin_rep", "")) == "1/2"
        and str(getattr(model_spec, "orbital_rep", "")) == "1/2"
        and _sector_dimension_for_spin_half(n_sites, target_tz2) > 0
    )
    standard_projector_requested = bool(standard_projector_requested_by_plan and use_tz_conserved)
    spectrum: Dict[str, Any] = {}
    if use_tz_conserved:
        u1_parent_dim = int((1 << n_sites) * _sector_dimension_for_spin_half(n_sites, target_tz2))
        if standard_projector_requested:
            projector_factor = _phase_scan_ed_projector_reduction_factor_estimate(ed_plan, geometry)
            hilbert_dim = int(max(1, int(u1_parent_dim) // max(1, projector_factor)))
            basis_type = "bitwise_spin_orbital_tz_projector_block"
        else:
            projector_factor = 1
            hilbert_dim = int(u1_parent_dim)
            basis_type = "bitwise_spin_orbital_total_tz_block"
    elif use_sz_conserved:
        u1_parent_dim = None
        projector_factor = 1
        hilbert_dim = int(_sector_dimension_for_spin_half(n_sites, target_sz2) * (1 << n_sites))
        basis_type = "bitwise_spin_orbital_total_sz_block"
    else:
        u1_parent_dim = None
        projector_factor = 1
        hilbert_dim = full_hilbert_dim
        basis_type = "legacy_full_tensor_product"
    if int(geometry.number_of_sites) > int(args.phase_scan_ed_max_sites):
        return {
            "status": "skipped",
            "alpha": float(alpha),
            "beta": float(beta),
            "reason": f"Quantum phase scan limited to {int(args.phase_scan_ed_max_sites)} sites.",
            "ed_backend": "standard",
            **ed_route_metadata,
            "basis_type": basis_type,
            "effective_hilbert_dimension": int(hilbert_dim),
            "full_hilbert_dimension": int(full_hilbert_dim),
            "use_sz_conserved": bool(use_sz_conserved),
            "use_tau_z_conserved": bool(use_tz_conserved),
        }
    if hilbert_dim > int(args.phase_scan_ed_max_hilbert_dim):
        return {
            "status": "skipped",
            "alpha": float(alpha),
            "beta": float(beta),
            "reason": (
                f"Quantum phase scan {basis_type} Hilbert dimension {hilbert_dim} exceeds "
                f"{int(args.phase_scan_ed_max_hilbert_dim)}."
            ),
            "ed_backend": "standard",
            **ed_route_metadata,
            "basis_type": basis_type,
            "effective_hilbert_dimension": int(hilbert_dim),
            "full_hilbert_dimension": int(full_hilbert_dim),
            "use_sz_conserved": bool(use_sz_conserved),
            "use_tau_z_conserved": bool(use_tz_conserved),
        }
    if use_tz_conserved:
        from ed_backend import all_bond_energies_sz_conserved

        if standard_projector_requested:
            spectrum, vectors, basis_list, basis_map = run_spin_orbital_projected_exact_spectrum(
                geometry=geometry,
                model_spec=model_spec,
                alpha=alpha,
                beta=beta,
                coupling_j=args.coupling_j,
                eigenstate_count=max(1, int(getattr(args, "ed_max_eigenstates", 2))),
                check_ground_state_degeneracy=False,
                jx=args.jx,
                jy=args.jy,
                jz=args.jz,
                external_field_terms=hamiltonian_external_field_terms,
                show_progress=show_progress,
                sparse_tol=float(getattr(args, "ed_sparse_tol", 0.0)),
                sparse_maxiter=(
                    int(getattr(args, "ed_sparse_maxiter", 0))
                    if int(getattr(args, "ed_sparse_maxiter", 0)) > 0
                    else None
                ),
                target_tz2=target_tz2,
                use_spin_pi_z=bool(use_z2_block and str(ed_z2_kind) == "spin_pi_z"),
                z2_target_parity=int(getattr(args, "z2_target_parity", 0)),
                use_translation_x=bool(ed_plan.get("use_translation_x_block", False)),
                use_translation_y=bool(ed_plan.get("use_translation_y_block", False)),
                momentum_x=int(ed_plan.get("momentum_x_block", momentum_x_block)),
                momentum_y=int(ed_plan.get("momentum_y_block", momentum_y_block)),
                use_combined_c3=bool(ed_plan.get("use_c3_block", False)),
                c3_q_blocks=str(ed_plan.get("c3_q_blocks", getattr(args, "ed_c3_q_blocks", "all"))),
                strict_projector_memory=False,
                allow_drop_c3_on_memory=True,
            )
        else:
            spectrum, vectors, basis_list, basis_map = run_spin_orbital_u1_exact_spectrum(
                geometry=geometry,
                model_spec=model_spec,
                alpha=alpha,
                beta=beta,
                coupling_j=args.coupling_j,
                eigenstate_count=max(1, int(getattr(args, "ed_max_eigenstates", 2))),
                check_ground_state_degeneracy=False,
                jx=args.jx,
                jy=args.jy,
                jz=args.jz,
                external_field_terms=hamiltonian_external_field_terms,
                show_progress=show_progress,
                sparse_tol=float(getattr(args, "ed_sparse_tol", 0.0)),
                sparse_maxiter=(
                    int(getattr(args, "ed_sparse_maxiter", 0))
                    if int(getattr(args, "ed_sparse_maxiter", 0)) > 0
                    else None
                ),
                use_sz_block=False,
                target_sz2=target_sz2,
                use_tau_z_block=True,
                target_tz2=target_tz2,
            )
        with profile_stage("observables"):
            energy = float(spectrum["ground_state_energy"])
            state = vectors[:, 0]
            correlations = collect_correlation_matrices_from_spin_orbital_u1_ed(
                geometry,
                state,
                basis_list,
                basis_map,
                show_progress=show_progress,
            )
            scalar_correlations = build_spin_orbital_u1_scalar_correlations(correlations)
            bond_rows = all_bond_energies_sz_conserved(
                geometry,
                correlations,
                alpha,
                beta,
                args.coupling_j,
                show_progress=show_progress,
                progress_desc="Tz-ED bond energies",
            )
            structure_rows = all_high_symmetry_structure_factors(scalar_correlations, geometry, lattice=lattice_name, show_progress=show_progress)
            try:
                plaquette_flux = plaquette_flux_from_spin_orbital_u1_ed_state(
                    geometry,
                    state,
                    basis_list,
                    basis_map,
                    plaquette_center_idx=None,
                )
            except Exception as exc:
                plaquette_flux = {"available": False, "warning": str(exc)}
    elif use_sz_conserved:
        from ed_backend import (
            all_bond_energies_sz_conserved,
            build_sz_conserved_scalar_correlations,
            collect_correlation_matrices_from_sz_conserved_ed,
            plaquette_flux_from_sz_conserved_ed_state,
            run_sz_conserved_exact_spectrum,
        )

        spectrum, vectors, basis_list, basis_map = run_sz_conserved_exact_spectrum(
            geometry=geometry,
            alpha=alpha,
            beta=beta,
            coupling_j=args.coupling_j,
            eigenstate_count=3,
            check_ground_state_degeneracy=False,
            external_field_terms=hamiltonian_external_field_terms,
            show_progress=show_progress,
            sparse_tol=float(getattr(args, "ed_sparse_tol", 0.0)),
            sparse_maxiter=(
                int(getattr(args, "ed_sparse_maxiter", 0))
                if int(getattr(args, "ed_sparse_maxiter", 0)) > 0
                else None
            ),
            target_sz2=target_sz2,
        )
        with profile_stage("observables"):
            energy = float(spectrum["ground_state_energy"])
            state = vectors[:, 0]
            correlations = collect_correlation_matrices_from_sz_conserved_ed(
                geometry,
                state,
                basis_list,
                basis_map,
                show_progress=show_progress,
            )
            scalar_correlations = build_sz_conserved_scalar_correlations(correlations)
            bond_rows = all_bond_energies_sz_conserved(
                geometry,
                correlations,
                alpha,
                beta,
                args.coupling_j,
                show_progress=show_progress,
            )
            try:
                plaquette_flux = plaquette_flux_from_sz_conserved_ed_state(
                    geometry,
                    state,
                    basis_list,
                    basis_map,
                    plaquette_center_idx=None,
                )
            except Exception as exc:
                plaquette_flux = {"available": False, "warning": str(exc)}
    else:
        energy, state = run_small_cluster_exact_diagonalization(
            geometry=geometry,
            model_spec=model_spec,
            alpha=alpha,
            beta=beta,
            coupling_j=args.coupling_j,
            jx=args.jx,
            jy=args.jy,
            jz=args.jz,
            external_field_terms=hamiltonian_external_field_terms,
            show_progress=show_progress,
            solver=getattr(args, "ed_solver", "auto"),
            sparse_tol=float(getattr(args, "ed_sparse_tol", 0.0)),
            sparse_maxiter=(
                int(getattr(args, "ed_sparse_maxiter", 0))
                if int(getattr(args, "ed_sparse_maxiter", 0)) > 0
                else None
            ),
        )
        with profile_stage("observables"):
            correlations = collect_correlation_matrices_from_ed(
                geometry,
                state,
                model_spec=model_spec,
                show_progress=show_progress,
            )
            scalar_correlations = build_spin_orbital_scalar_correlations(correlations)
            bond_rows = all_bond_energies(
                geometry,
                correlations,
                model_spec,
                alpha,
                beta,
                args.coupling_j,
                jx=args.jx,
                jy=args.jy,
                jz=args.jz,
                show_progress=show_progress,
            )
            try:
                plaquette_flux = plaquette_flux_from_ed_state(
                    geometry,
                    state,
                    model_spec,
                    plaquette_center_idx=None,
                )
            except Exception as exc:
                plaquette_flux = {"available": False, "warning": str(exc)}
    with profile_stage("observables"):
        structure_rows = all_high_symmetry_structure_factors(
            scalar_correlations,
            geometry,
            lattice=lattice_name,
            show_progress=show_progress,
        )
        diagnostics = _phase_observable_diagnostics(
            structure_rows,
            bond_rows,
            geometry.number_of_sites,
            plaquette_flux=plaquette_flux,
        )
        phase_label = _classify_phase_from_diagnostics(diagnostics, alpha, beta, "quantum_ed", thresholds)
    return {
        "status": "completed",
        "alpha": float(alpha),
        "beta": float(beta),
        "ed_backend": "standard",
        **ed_route_metadata,
        "requested_ed_backend": requested_ed_backend_name,
        "ed_backend_override_reason": ed_backend_override_reason,
        "basis_type": basis_type,
        "effective_hilbert_dimension": int(hilbert_dim),
        "u1_parent_hilbert_dimension": int(u1_parent_dim) if u1_parent_dim is not None else None,
        "projector_reduction_factor_estimate": int(projector_factor),
        "projector_strategy": spectrum.get("projector_strategy") if isinstance(spectrum, dict) else None,
        "memory_estimate_MB": spectrum.get("memory_estimate_MB") if isinstance(spectrum, dict) else None,
        "dropped_symmetries": (
            list(ed_route_metadata.get("dropped_symmetries", []))
            + (list(spectrum.get("dropped_symmetries", [])) if isinstance(spectrum, dict) else [])
        ),
        "drop_reasons": spectrum.get("drop_reasons", {}) if isinstance(spectrum, dict) else {},
        "projector_reduced_dimension": (
            int(spectrum.get("projector_reduced_dimension"))
            if isinstance(spectrum, dict) and spectrum.get("projector_reduced_dimension") is not None
            else None
        ),
        "full_hilbert_dimension": int(full_hilbert_dim),
        "use_sz_conserved": bool(use_sz_conserved),
        "use_tau_z_conserved": bool(use_tz_conserved),
        "use_z2_conserved": bool(isinstance(spectrum, dict) and spectrum.get("use_z2_block", False)),
        "use_translation_x_conserved": bool(isinstance(spectrum, dict) and spectrum.get("use_translation_x_block", False)),
        "use_translation_y_conserved": bool(isinstance(spectrum, dict) and spectrum.get("use_translation_y_block", False)),
        "use_c3_conserved": bool(isinstance(spectrum, dict) and spectrum.get("use_c3_block", False)),
        "commutator_norms": spectrum.get("commutator_norms", {}) if isinstance(spectrum, dict) else {},
        "selected_c3_q": spectrum.get("selected_c3_q") if isinstance(spectrum, dict) else None,
        "c3_sector_energies": spectrum.get("c3_sector_energies") if isinstance(spectrum, dict) else None,
        "ed_symmetry_plan": ed_plan if ed_plan else None,
        "phase_label": phase_label,
        "energy": float(energy),
        "energy_per_site": float(energy / float(max(1, geometry.number_of_sites))),
        "diagnostics": diagnostics,
        "plaquette_flux": plaquette_flux,
        "all_plaquette_fluxes": extract_all_plaquette_fluxes(plaquette_flux),
        "structure_factors": structure_rows,
        "bond_energies": bond_rows,
    }


def _phase_scan_classical_point(
    geometry: Any,
    model_spec: Any,
    lattice_name: str,
    alpha: float,
    beta: float,
    args: Any,
    point_index: int,
    hamiltonian_external_field_terms: List[Tuple[float, str]],
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    result = _run_classical_product_ground_state(
        geometry=geometry,
        model_spec=model_spec,
        alpha=alpha,
        beta=beta,
        coupling_j=args.coupling_j,
        jx=args.jx,
        jy=args.jy,
        jz=args.jz,
        restarts=args.phase_scan_classical_restarts,
        sweeps=args.phase_scan_classical_sweeps,
        initial_temperature=args.phase_scan_classical_initial_temperature,
        final_temperature=args.phase_scan_classical_final_temperature,
        initial_step=args.phase_scan_classical_initial_step,
        final_step=args.phase_scan_classical_final_step,
        seed=int(args.phase_scan_random_seed) + int(point_index),
        external_field_terms=hamiltonian_external_field_terms,
    )
    spin_vectors = np.asarray(result["spin_vectors"], dtype=float)
    orbital_vectors = np.asarray(result["orbital_vectors"], dtype=float)
    with profile_stage("observables"):
        structure_rows = _classical_structure_factor_rows(geometry, lattice_name, spin_vectors, orbital_vectors)
        bond_rows = _classical_bond_rows(
            geometry,
            spin_vectors,
            orbital_vectors,
            model_spec,
            alpha,
            beta,
            args.coupling_j,
            args.jx,
            args.jy,
            args.jz,
        )
        diagnostics = _phase_observable_diagnostics(structure_rows, bond_rows, geometry.number_of_sites)
        phase_label = _classify_phase_from_diagnostics(diagnostics, alpha, beta, "classical_product", thresholds)
    return {
        "status": "completed",
        "alpha": float(alpha),
        "beta": float(beta),
        "phase_label": phase_label,
        "energy": float(result["energy"]),
        "energy_per_site": float(result["energy_per_site"]),
        "diagnostics": diagnostics,
        "structure_factors": structure_rows,
        "bond_energies": bond_rows,
        "spin_vectors": spin_vectors.tolist(),
        "orbital_vectors": orbital_vectors.tolist(),
    }


def _phase_scan_quantum_global_skip(
    geometry: Any,
    model_spec: Any,
    args: Any,
    hamiltonian_external_field_terms: List[Tuple[float, str]] | None = None,
) -> Dict[str, Any] | None:
    local_dim = int(model_spec.physical_dim)
    n_sites = int(geometry.number_of_sites)
    full_hilbert_dim = int(local_dim ** n_sites)
    external_field_terms = list(hamiltonian_external_field_terms or [])
    ed_backend_name = str(getattr(args, "ed_backend", "standard")).strip().lower()
    if ed_backend_name == "ed":
        ed_backend_name = "standard"
    ed_plan = (
        getattr(args, "ed_symmetry_plan", {})
        if isinstance(getattr(args, "ed_symmetry_plan", None), dict)
        else {}
    )
    effective_ed_engine = str(ed_plan.get("effective_engine", ed_plan.get("engine", getattr(args, "ed_symmetry_engine", "auto")))).strip().lower()
    requested_ed_backend_name = ed_backend_name
    if effective_ed_engine == "standard_projector" and ed_backend_name == "quspin":
        ed_backend_name = "standard"
    elif effective_ed_engine.startswith("quspin") and ed_backend_name != "quspin":
        ed_backend_name = "quspin"
    actual_ed_backend_name = ed_backend_name
    ed_route_metadata = {
        "requested_backend": requested_ed_backend_name,
        "actual_backend": actual_ed_backend_name,
        "backend_override_reason": ed_plan.get("backend_override_reason") if isinstance(ed_plan, dict) else None,
        "requested_symmetry_engine": str(ed_plan.get("requested_engine", getattr(args, "ed_symmetry_engine", "auto")))
        if isinstance(ed_plan, dict)
        else str(getattr(args, "ed_symmetry_engine", "auto")),
        "symmetry_engine": effective_ed_engine,
        "requested_symmetries": list(ed_plan.get("requested_symmetries", [])) if isinstance(ed_plan, dict) else [],
        "accepted_symmetries": list(ed_plan.get("accepted_symmetries", [])) if isinstance(ed_plan, dict) else [],
        "dropped_symmetries": list(ed_plan.get("dropped_symmetries", [])) if isinstance(ed_plan, dict) else [],
        "symmetry_reasons": dict(ed_plan.get("reasons", {})) if isinstance(ed_plan, dict) else {},
        "z2_generator_used": ed_plan.get("z2_generator_used") if isinstance(ed_plan, dict) else None,
        "z2_selection_reason": ed_plan.get("z2_selection_reason") if isinstance(ed_plan, dict) else None,
        "quspin_z2_selection_reason": ed_plan.get("quspin_z2_selection_reason") if isinstance(ed_plan, dict) else None,
    }
    if effective_ed_engine == "quspin_experimental_c3":
        experimental_report = ed_plan.get("quspin_experimental_c3", {}) if isinstance(ed_plan, dict) else {}
        return {
            "status": "skipped",
            "reason": (
                experimental_report.get("phase_scan_rejection_reason")
                if isinstance(experimental_report, dict) and experimental_report.get("phase_scan_rejection_reason")
                else (
                    "Quantum phase scan skipped because quspin_experimental_c3 is not implemented; "
                    "pure C3 maps are not physical for Yao-Lee, and validation belongs in tests."
                )
            ),
            "ed_backend": "quspin",
            **ed_route_metadata,
            "actual_backend": "quspin",
            "effective_hilbert_dimension": 0,
            "full_hilbert_dimension": int(full_hilbert_dim),
            "quspin_experimental_c3": experimental_report,
        }
    use_sz_block = _phase_scan_uses_sz_block(args)
    use_tau_z_block = _phase_scan_uses_tau_z_block(args)
    use_z2_block = bool(getattr(args, "use_z2_block", False))
    use_translation_x_block = bool(getattr(args, "use_translation_x_block", False))
    use_translation_y_block = bool(getattr(args, "use_translation_y_block", False))
    use_translation_block = bool(use_translation_x_block or use_translation_y_block)
    use_reflection_block = bool(getattr(args, "use_reflection_block", False))
    requested_translation_block = bool(use_translation_block)
    requested_translation_x_block = bool(use_translation_x_block)
    requested_translation_y_block = bool(use_translation_y_block)
    requested_reflection_block = bool(use_reflection_block)
    reflection_block = int(getattr(args, "reflection_block", 0))
    target_sz2 = int(getattr(args, "u1_target_sz2", 0))
    target_tz2 = int(getattr(args, "u1_target_tz2", 0))
    field_ops = {
        str(op_name)
        for coefficient, op_name in list(external_field_terms or [])
        if abs(float(coefficient)) > 1e-14
    }
    model_requested_reductions = {
        str(item).strip().lower()
        for item in (
            getattr(args, "model_symmetry_selection", {}) or {}
        ).get("requested_reductions", [])
    } if isinstance(getattr(args, "model_symmetry_selection", None), dict) else set()
    if ed_backend_name == "quspin" and "z2" in model_requested_reductions and not field_ops:
        use_z2_block = True

    if ed_backend_name == "quspin":
        if bool(use_sz_block) and bool(field_ops.intersection({"Sx", "Sy"})):
            use_sz_block = False
            use_z2_block = False
        if bool(use_z2_block) and bool(field_ops.intersection({"Sx", "Sy", "Sz"})):
            use_z2_block = False
        quspin_package_available = importlib.util.find_spec("quspin") is not None
        quspin_translation_x_reason = None
        quspin_translation_y_reason = None
        quspin_translation_support_report = None
        if use_translation_block:
            if not quspin_package_available:
                reason = "QuSpin package is not installed, so translation blocks cannot be checked."
                use_translation_x_block = False
                use_translation_y_block = False
                quspin_translation_x_reason = reason if requested_translation_x_block else None
                quspin_translation_y_reason = reason if requested_translation_y_block else None
            else:
                try:
                    import quspin_backend as quspin_validation_backend

                    support = quspin_validation_backend.quspin_translation_block_support(geometry)
                    quspin_translation_support_report = support
                    x_support = support.get("x", {})
                    y_support = support.get("y", {})
                    use_translation_x_block = bool(
                        requested_translation_x_block and x_support.get("supported", False)
                    )
                    use_translation_y_block = bool(
                        requested_translation_y_block and y_support.get("supported", False)
                    )
                    quspin_translation_x_reason = x_support.get("reason") if requested_translation_x_block else None
                    quspin_translation_y_reason = y_support.get("reason") if requested_translation_y_block else None
                except Exception as exc:
                    reason = str(exc)
                    use_translation_x_block = False
                    use_translation_y_block = False
                    quspin_translation_x_reason = reason if requested_translation_x_block else None
                    quspin_translation_y_reason = reason if requested_translation_y_block else None
        use_translation_block = bool(use_translation_x_block or use_translation_y_block)
        quspin_translation_reason = {
            "x": quspin_translation_x_reason,
            "y": quspin_translation_y_reason,
        }
        quspin_reflection_reason = None
        if requested_reflection_block or reflection_block != 0:
            quspin_reflection_reason = (
                "QuSpin reflection/C3 blocks are not applied for the bond-directional Yao-Lee Hamiltonian; "
                "they can permute x/y/z bond types unless a gauge map is implemented."
            )
        use_reflection_block = False
        reflection_block = 0
        hilbert_dim = _phase_scan_spin_orbital_block_dimension(
            n_sites,
            use_sz_block,
            target_sz2,
            use_tau_z_block,
            target_tz2,
        )
        basis_type = (
            "quspin_tensor_spin_z2_orbital_tz"
            if use_z2_block and use_tau_z_block
            else (
                "quspin_tensor_spin_z2_orbital_full"
                if use_z2_block
                else (
                    "quspin_tensor_"
                    f"spin_{'u1_block' if use_sz_block else 'full'}_"
                    f"orbital_{'u1_block' if use_tau_z_block else 'full'}"
                )
            )
        )
        compatible = (
            quspin_package_available
            and str(getattr(model_spec, "spin_rep", "")) == "1/2"
            and str(getattr(model_spec, "orbital_rep", "")) == "1/2"
            and str(getattr(model_spec, "model_family", "")) == "yao_lee"
            and str(getattr(model_spec, "ising_axis", "")) == "z"
            and int(hilbert_dim) > 0
        )
        pre_quspin_hilbert_dim = int(hilbert_dim)
        quspin_basis_build_reason = None
        if compatible:
            try:
                import quspin_backend as quspin_basis_backend

                preflight_basis = quspin_basis_backend.build_quspin_yao_lee_basis(
                    n_sites,
                    geometry=geometry,
                    use_sz_block=use_sz_block,
                    target_sz2=target_sz2,
                    use_tau_z_block=use_tau_z_block,
                    target_tz2=target_tz2,
                    use_z2_block=use_z2_block,
                    z2_generator="spin_flip" if use_z2_block else None,
                    z2_target_parity=int(getattr(args, "z2_target_parity", 0)),
                    use_translation_block=use_translation_block,
                    use_translation_x_block=use_translation_x_block,
                    use_translation_y_block=use_translation_y_block,
                    momentum_block_1=int(getattr(args, "momentum_x_block", 0)),
                    momentum_block_2=int(getattr(args, "momentum_y_block", 0)),
                    momentum_x_block=int(getattr(args, "momentum_x_block", 0)),
                    momentum_y_block=int(getattr(args, "momentum_y_block", 0)),
                    use_reflection_block=False,
                    reflection_block=0,
                )
                hilbert_dim = int(preflight_basis.Ns)
            except Exception as exc:
                compatible = False
                quspin_basis_build_reason = f"Failed to build the requested QuSpin reduced basis: {exc}"
        common = {
            "ed_backend": "quspin",
            **ed_route_metadata,
            "actual_backend": "quspin",
            "basis_type": basis_type,
            "effective_hilbert_dimension": int(hilbert_dim),
            "pre_quspin_hilbert_dimension_estimate": int(pre_quspin_hilbert_dim),
            "full_hilbert_dimension": int(full_hilbert_dim),
            "use_sz_conserved": bool(use_sz_block),
            "use_sz_block": bool(use_sz_block),
            "use_tau_z_block": bool(use_tau_z_block),
            "use_z2_block": bool(use_z2_block),
            "z2_kind": "spin_flip" if use_z2_block else None,
            "use_sz_conserved_requested": bool(use_sz_block),
            "quspin_package_available": bool(quspin_package_available),
            "quspin_requested_translation_block": bool(requested_translation_block),
            "quspin_requested_translation_x_block": bool(requested_translation_x_block),
            "quspin_requested_translation_y_block": bool(requested_translation_y_block),
            "quspin_use_translation_block": bool(use_translation_block),
            "quspin_use_translation_x_block": bool(use_translation_x_block),
            "quspin_use_translation_y_block": bool(use_translation_y_block),
            "quspin_translation_reason": quspin_translation_reason,
            "quspin_translation_x_reason": quspin_translation_x_reason,
            "quspin_translation_y_reason": quspin_translation_y_reason,
            "quspin_translation_support": quspin_translation_support_report,
            "quspin_requested_reflection_block": bool(requested_reflection_block),
            "quspin_use_reflection_block": False,
            "quspin_reflection_reason": quspin_reflection_reason,
        }
        if not compatible:
            return {
                **common,
                "reason": (
                    quspin_basis_build_reason
                    if quspin_basis_build_reason is not None
                    else
                    "QuSpin ED phase scan requires reachable shared U1 target sectors, "
                    "the quspin Python package, "
                    "spin_rep=1/2, orbital_rep=1/2, model_family=yao_lee, "
                    "and ising_axis=z. "
                    "Reflection/C3 blocks are forbidden; Tz can be combined with spin-flip Z2, "
                    "but not with the current 2D translation blocks."
                ),
            }
        if n_sites > int(args.phase_scan_ed_max_sites):
            return {
                **common,
                "reason": f"Quantum phase scan limited to {int(args.phase_scan_ed_max_sites)} sites.",
            }
        if hilbert_dim > int(args.phase_scan_ed_max_hilbert_dim):
            return {
                **common,
                "reason": (
                    f"Quantum phase scan {basis_type} Hilbert dimension {hilbert_dim} exceeds "
                    f"{int(args.phase_scan_ed_max_hilbert_dim)}."
                ),
            }
        return None

    use_sz_conserved_requested = bool(use_sz_block)
    use_sz_conserved = (
        use_sz_conserved_requested
        and str(getattr(model_spec, "spin_rep", "")) == "1/2"
        and str(getattr(model_spec, "orbital_rep", "")) == "1/2"
        and _sector_dimension_for_spin_half(n_sites, target_sz2) > 0
    )
    if use_sz_conserved:
        hilbert_dim = int(_sector_dimension_for_spin_half(n_sites, target_sz2) * (1 << n_sites))
        basis_type = "bitwise_spin_orbital_total_sz_block"
    elif bool(use_tau_z_block) and str(getattr(model_spec, "spin_rep", "")) == "1/2" and str(getattr(model_spec, "orbital_rep", "")) == "1/2":
        tz_dim = _sector_dimension_for_spin_half(n_sites, target_tz2)
        hilbert_dim = int((1 << n_sites) * tz_dim) if tz_dim > 0 else 0
        basis_type = "bitwise_spin_orbital_total_tz_block"
    else:
        hilbert_dim = full_hilbert_dim
        basis_type = "legacy_full_tensor_product"

    common = {
        "ed_backend": "standard",
        **ed_route_metadata,
        "actual_backend": "standard",
        "basis_type": basis_type,
        "effective_hilbert_dimension": int(hilbert_dim),
        "full_hilbert_dimension": int(full_hilbert_dim),
        "use_sz_conserved": bool(use_sz_conserved),
        "use_tau_z_conserved": bool(basis_type == "bitwise_spin_orbital_total_tz_block" and hilbert_dim > 0),
        "use_sz_conserved_requested": bool(use_sz_conserved_requested),
    }
    if use_sz_conserved_requested and not use_sz_conserved and str(getattr(model_spec, "orbital_rep", "")) != "0":
        return {
            **common,
            "reason": "Standard Sz-block ED requires a reachable target 2*Sz sector with spin_rep=1/2 and orbital_rep=1/2.",
        }
    if n_sites > int(args.phase_scan_ed_max_sites):
        return {
            **common,
            "reason": f"Quantum phase scan limited to {int(args.phase_scan_ed_max_sites)} sites.",
        }
    if hilbert_dim > int(args.phase_scan_ed_max_hilbert_dim):
        return {
            **common,
            "reason": (
                f"Quantum phase scan {basis_type} Hilbert dimension {hilbert_dim} exceeds "
                f"{int(args.phase_scan_ed_max_hilbert_dim)}."
            ),
        }
    return None


def run_alpha_beta_phase_scan(
    geometry: Any,
    model_spec: Any,
    lattice_name: str,
    args: Any,
    hamiltonian_external_field_terms: List[Tuple[float, str]],
    show_progress: bool,
) -> Dict[str, Any]:
    alphas, betas = _phase_scan_grid_from_args(args)
    thresholds = phase_classifier_thresholds_from_args(args)
    raw_mode = str(getattr(args, "phase_scan_mode", "quantum_ed")).strip().lower()
    if raw_mode in ("both", "all"):
        modes = ["quantum_ed", "classical_product"]
    elif raw_mode in ("classical", "classical_product"):
        modes = ["classical_product"]
    elif raw_mode in ("quantum", "methods", "quantum_ed", "ed", "exact", "exact_diagonalization"):
        modes = ["quantum_ed"]
    else:
        modes = [raw_mode]
    total_points = len(alphas) * len(betas)
    quantum_ed_backend = str(getattr(args, "ed_backend", "standard")).strip().lower()
    if quantum_ed_backend == "ed":
        quantum_ed_backend = "standard"
    quantum_ed_use_sz_block = _phase_scan_uses_sz_block(args)
    quantum_ed_use_tau_z_block = _phase_scan_uses_tau_z_block(args)
    quantum_ed_reductions = _phase_scan_reductions(args)
    output: Dict[str, Any] = {
        "status": "completed",
        "mode": str(args.phase_scan_mode),
        "grid": {
            "alpha_min": float(min(alphas)),
            "alpha_max": float(max(alphas)),
            "alpha_points": int(len(alphas)),
            "alpha_values": alphas,
            "beta_min": float(min(betas)),
            "beta_max": float(max(betas)),
            "beta_points": int(len(betas)),
            "beta_values": betas,
        },
        "solver_controls": {
            "quantum_ed_max_sites": int(args.phase_scan_ed_max_sites),
            "quantum_ed_max_hilbert_dimension": int(args.phase_scan_ed_max_hilbert_dim),
            "quantum_ed_backend": quantum_ed_backend,
            "quantum_ed_solver": str(getattr(args, "ed_solver", "auto")),
            "quantum_ed_use_sz_block": bool(quantum_ed_use_sz_block),
            "quantum_ed_use_tau_z_block": bool(quantum_ed_use_tau_z_block),
            "quantum_ed_use_z2_block": bool(getattr(args, "use_z2_block", False)),
            "quantum_ed_use_translation_x_block": bool(getattr(args, "use_translation_x_block", False)),
            "quantum_ed_use_translation_y_block": bool(getattr(args, "use_translation_y_block", False)),
            "symmetry_reductions": list(quantum_ed_reductions),
            "symmetry_mode": str(getattr(args, "symmetry_mode", "none")),
            "target_sz2": int(getattr(args, "u1_target_sz2", 0)),
            "target_tz2": int(getattr(args, "u1_target_tz2", 0)),
            "z2_target_parity": int(getattr(args, "z2_target_parity", 0)) % 2,
            "quantum_ed_sparse_tol": float(getattr(args, "ed_sparse_tol", 0.0)),
            "quantum_ed_sparse_maxiter": (
                int(getattr(args, "ed_sparse_maxiter", 0))
                if int(getattr(args, "ed_sparse_maxiter", 0)) > 0
                else None
            ),
            "classical_restarts": int(args.phase_scan_classical_restarts),
            "classical_sweeps": int(args.phase_scan_classical_sweeps),
            "classical_initial_temperature": float(args.phase_scan_classical_initial_temperature),
            "classical_final_temperature": float(args.phase_scan_classical_final_temperature),
            "classical_initial_step": float(args.phase_scan_classical_initial_step),
            "classical_final_step": float(args.phase_scan_classical_final_step),
            "random_seed": int(args.phase_scan_random_seed),
        },
        "classifier_thresholds": thresholds,
        "classifier_note": (
            "Phase labels are finite-size, observable-based assignments from dominant "
            "spin/orbital structure factors, plaquette flux, and bond-energy nematicity. "
            "They are saved with diagnostics for reproducibility and should be checked "
            "against larger clusters or denser grids before quoting thermodynamic boundaries."
        ),
    }
    progress_bar = _make_progress_bar(
        enabled=show_progress,
        total=total_points * len(modes),
        desc="phase scan",
        unit="point",
        leave=False,
    )
    scan_profile_rows: List[Dict[str, Any]] = []
    scan_points_profiled = profile_scan_points_enabled()

    def _scan_point_profile_extra(row: Dict[str, Any]) -> Dict[str, Any]:
        extra: Dict[str, Any] = {}
        flux = row.get("plaquette_flux")
        if not isinstance(flux, dict):
            diagnostics = row.get("diagnostics")
            if isinstance(diagnostics, dict):
                flux = diagnostics.get("plaquette_flux")
        if isinstance(flux, dict):
            flux_value = flux.get("W_p", flux.get("value"))
            try:
                if flux_value is not None:
                    extra["W_p"] = float(flux_value)
            except (TypeError, ValueError):
                pass
        if row.get("effective_hilbert_dimension") is not None:
            try:
                extra["effective_hilbert_dimension"] = int(row["effective_hilbert_dimension"])
            except (TypeError, ValueError):
                pass
        if row.get("actual_backend") is not None:
            extra["actual_backend"] = row.get("actual_backend")
        if row.get("symmetry_engine") is not None:
            extra["symmetry_engine"] = row.get("symmetry_engine")
        return extra

    point_index = 0
    for mode in modes:
        if mode == "quantum_ed":
            skip_info = _phase_scan_quantum_global_skip(
                geometry,
                model_spec,
                args,
                hamiltonian_external_field_terms,
            )
            if skip_info is not None:
                output[mode] = {
                    "status": "skipped",
                    "reason": str(skip_info["reason"]),
                    "rows": [],
                    "completed_points": 0,
                    "failed_points": 0,
                    "skipped_points": int(total_points),
                    **skip_info,
                    "note": (
                        "Quantum ED phase scan was skipped once at solver setup; "
                        "per-alpha/beta skipped rows are intentionally omitted."
                    ),
                }
                if progress_bar is not None:
                    progress_bar.update(total_points)
                point_index += int(total_points)
                continue

        rows: List[Dict[str, Any]] = []
        for beta in betas:
            for alpha in alphas:
                point_start = time.perf_counter() if scan_points_profiled else None
                try:
                    if mode == "quantum_ed":
                        row = _phase_scan_quantum_point(
                            geometry,
                            model_spec,
                            lattice_name,
                            alpha,
                            beta,
                            args,
                            hamiltonian_external_field_terms,
                            thresholds,
                            show_progress=False,
                        )
                    else:
                        row = _phase_scan_classical_point(
                            geometry,
                            model_spec,
                            lattice_name,
                            alpha,
                            beta,
                            args,
                            point_index,
                            hamiltonian_external_field_terms,
                            thresholds,
                        )
                except Exception as exc:
                    row = {
                        "status": "failed",
                        "alpha": float(alpha),
                        "beta": float(beta),
                        "error": str(exc),
                    }
                if scan_points_profiled and point_start is not None:
                    point_elapsed = time.perf_counter() - point_start
                    point_profile = record_scan_point_timing(
                        mode=mode,
                        alpha=float(alpha),
                        beta=float(beta),
                        status=str(row.get("status", "unknown")),
                        wall_time_seconds=float(point_elapsed),
                        point_index=int(point_index),
                        extra=_scan_point_profile_extra(row),
                    )
                    row.setdefault("profiling", {})
                    if isinstance(row["profiling"], dict):
                        row["profiling"]["wall_time_seconds"] = float(point_elapsed)
                    scan_profile_rows.append(point_profile)
                rows.append(row)
                point_index += 1
                if progress_bar is not None:
                    progress_bar.update(1)
        failed_count = int(sum(1 for row in rows if row.get("status") == "failed"))
        output[mode] = {
            "status": "completed_with_warnings" if failed_count > 0 else "completed",
            "rows": rows,
            "completed_points": int(sum(1 for row in rows if row.get("status") == "completed")),
            "failed_points": failed_count,
            "skipped_points": int(sum(1 for row in rows if row.get("status") == "skipped")),
        }
    if progress_bar is not None:
        progress_bar.close()
    failed_points = sum(
        int(mode_data.get("failed_points", 0))
        for mode_data in output.values()
        if isinstance(mode_data, dict)
    )
    if failed_points > 0:
        output["status"] = "completed_with_warnings"
    if scan_profile_rows:
        output["profiling"] = {
            "scan_point_timing": _profile_scan_point_summary(scan_profile_rows)
        }
    return output


# ----------------------------------------------------------------------
# Opt-in ED symmetry benchmark matrix
# ----------------------------------------------------------------------

def _benchmark_write_json(filepath: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(_profile_json_safe(payload), file, indent=2, sort_keys=True)


def _benchmark_collect_nnz(value: Any, output: List[int]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() == "nnz":
                try:
                    if item is not None:
                        output.append(int(item))
                except Exception:
                    pass
            else:
                _benchmark_collect_nnz(item, output)
    elif isinstance(value, list):
        for item in value:
            _benchmark_collect_nnz(item, output)


def _benchmark_dense_memory_mb(dim: Any) -> float | None:
    try:
        dimension = int(dim)
    except Exception:
        return None
    return float(dimension * dimension * 16) / float(1024 * 1024)


def _benchmark_stage_table(profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    stage_timing = profile.get("stage_timing", {}) if isinstance(profile, dict) else {}
    table = stage_timing.get("table", []) if isinstance(stage_timing, dict) else []
    return list(table) if isinstance(table, list) else []


def _benchmark_top_stage(profile: Dict[str, Any]) -> Dict[str, Any] | None:
    excluded = {"spin-orbital projector ED", "spin-orbital U1 ED", "ED diagonalization"}
    table = [
        row
        for row in _benchmark_stage_table(profile)
        if str(row.get("stage")) not in excluded
    ]
    if not table:
        return None
    return max(table, key=lambda row: float(row.get("total_seconds", 0.0) or 0.0))


def _benchmark_spectrum_dimensions(spectrum: Dict[str, Any]) -> Dict[str, Any]:
    parent_dim = (
        spectrum.get("u1_basis_dimension")
        or spectrum.get("hilbert_dimension")
        or spectrum.get("hilbert_dim")
    )
    projected_dim = (
        spectrum.get("projector_reduced_dimension")
        or spectrum.get("reduced_dimension")
        or spectrum.get("projector_solver_dimension")
        or spectrum.get("hilbert_dimension")
        or spectrum.get("hilbert_dim")
    )
    full_dim = spectrum.get("full_spin_orbital_hilbert_dim")
    return {
        "full_hilbert_dimension": int(full_dim) if full_dim is not None else None,
        "parent_hilbert_dimension": int(parent_dim) if parent_dim is not None else None,
        "reduced_or_projected_dimension": int(projected_dim) if projected_dim is not None else None,
    }


def _benchmark_case_summary(
    case: Dict[str, Any],
    spectrum: Dict[str, Any],
    profile: Dict[str, Any],
    elapsed: float,
) -> Dict[str, Any]:
    nnz_values: List[int] = []
    _benchmark_collect_nnz(spectrum.get("memory_diagnostics"), nnz_values)
    _benchmark_collect_nnz(spectrum, nnz_values)
    dims = _benchmark_spectrum_dimensions(spectrum)
    memory_estimates = {
        "full_dense_complex128_mb": _benchmark_dense_memory_mb(dims.get("full_hilbert_dimension")),
        "parent_dense_complex128_mb": _benchmark_dense_memory_mb(dims.get("parent_hilbert_dimension")),
        "projected_dense_complex128_mb": _benchmark_dense_memory_mb(dims.get("reduced_or_projected_dimension")),
    }
    profile_metadata = profile.get("metadata", {}) if isinstance(profile, dict) else {}
    summary_metadata = profile_metadata.get("summary", {}) if isinstance(profile_metadata, dict) else {}
    if isinstance(summary_metadata, dict) and isinstance(summary_metadata.get("estimated_dense_memory_mb"), dict):
        memory_estimates["profile_estimated_dense_memory_mb"] = summary_metadata["estimated_dense_memory_mb"]
    return {
        "case_id": case["id"],
        "label": case["label"],
        "status": "completed",
        "requested_backend": case["ed_backend"],
        "requested_symmetry_engine": case["ed_symmetry_engine"],
        "wall_time_seconds": float(profile.get("wall_time_seconds", elapsed)),
        "measured_elapsed_seconds": float(elapsed),
        "ground_state_energy": spectrum.get("ground_state_energy"),
        "solver_mode": spectrum.get("solver_mode"),
        "basis_type": spectrum.get("basis_type"),
        "dimensions": dims,
        "sparse_nnz": {
            "max_observed_nnz": max(nnz_values) if nnz_values else None,
            "observed_nnz_values": nnz_values,
        },
        "memory_estimates": memory_estimates,
        "stage_timing": _benchmark_stage_table(profile),
        "top_stage": _benchmark_top_stage(profile),
        "cprofile": profile.get("cprofile") if isinstance(profile, dict) else None,
        "spectrum": spectrum,
    }


def _benchmark_cases(include_quspin: bool) -> List[Dict[str, Any]]:
    cases = [
        {
            "id": "A_standard_projector_tz",
            "label": "A. standard_projector, Tz only",
            "ed_backend": "standard",
            "ed_symmetry_engine": "standard_projector",
            "runner": "standard_projector",
            "use_spin_pi_z": False,
            "use_translation": False,
            "use_c3": False,
            "c3_q_blocks": "off",
        },
        {
            "id": "B_standard_projector_tz_z2",
            "label": "B. standard_projector, Tz+Z2",
            "ed_backend": "standard",
            "ed_symmetry_engine": "standard_projector",
            "runner": "standard_projector",
            "use_spin_pi_z": True,
            "use_translation": False,
            "use_c3": False,
            "c3_q_blocks": "off",
        },
        {
            "id": "C_standard_projector_tz_translation",
            "label": "C. standard_projector, Tz+translation",
            "ed_backend": "standard",
            "ed_symmetry_engine": "standard_projector",
            "runner": "standard_projector",
            "use_spin_pi_z": False,
            "use_translation": True,
            "use_c3": False,
            "c3_q_blocks": "off",
        },
        {
            "id": "D_standard_projector_tz_translation_c3_all",
            "label": "D. standard_projector, Tz+translation+C3(q=all)",
            "ed_backend": "standard",
            "ed_symmetry_engine": "standard_projector",
            "runner": "standard_projector",
            "use_spin_pi_z": False,
            "use_translation": True,
            "use_c3": True,
            "c3_q_blocks": "all",
        },
    ]
    if include_quspin:
        cases.extend(
            [
                {
                    "id": "E1_quspin_tz",
                    "label": "E1. QuSpin, Tz only",
                    "ed_backend": "quspin",
                    "ed_symmetry_engine": "quspin_native",
                    "runner": "quspin",
                    "use_z2": False,
                },
                {
                    "id": "E2_quspin_tz_z2",
                    "label": "E2. QuSpin, Tz+spin_flip Z2",
                    "ed_backend": "quspin",
                    "ed_symmetry_engine": "quspin_native",
                    "runner": "quspin",
                    "use_z2": True,
                },
            ]
        )
    return cases


def _run_standard_projector_benchmark_case(
    geometry: Any,
    model_spec: Any,
    case: Dict[str, Any],
    *,
    alpha: float,
    beta: float,
    coupling_j: float,
    sparse_tol: float,
    sparse_maxiter: int | None,
) -> Dict[str, Any]:
    import ed_backend

    spectrum, _vectors, _basis_list, _basis_map = ed_backend.run_spin_orbital_projected_exact_spectrum(
        geometry=geometry,
        model_spec=model_spec,
        alpha=float(alpha),
        beta=float(beta),
        coupling_j=float(coupling_j),
        eigenstate_count=1,
        check_ground_state_degeneracy=False,
        show_progress=False,
        sparse_tol=float(sparse_tol),
        sparse_maxiter=sparse_maxiter,
        target_tz2=0,
        use_spin_pi_z=bool(case.get("use_spin_pi_z", False)),
        z2_target_parity=0,
        use_translation_x=bool(case.get("use_translation", False)),
        use_translation_y=bool(case.get("use_translation", False)),
        momentum_x=0,
        momentum_y=0,
        use_combined_c3=bool(case.get("use_c3", False)),
        c3_q_blocks=str(case.get("c3_q_blocks", "off")),
    )
    return spectrum


def _run_quspin_benchmark_case(
    geometry: Any,
    model_spec: Any,
    case: Dict[str, Any],
    *,
    alpha: float,
    beta: float,
    coupling_j: float,
    sparse_tol: float,
    sparse_maxiter: int | None,
) -> Dict[str, Any]:
    import quspin_backend

    available, reason = quspin_backend.quspin_package_available()
    if not bool(available):
        raise RuntimeError(f"QuSpin is not available: {reason}")
    spectrum, _vectors = quspin_backend.run_small_cluster_exact_spectrum(
        geometry=geometry,
        model_spec=model_spec,
        alpha=float(alpha),
        beta=float(beta),
        coupling_j=float(coupling_j),
        eigenstate_count=1,
        check_ground_state_degeneracy=False,
        external_field_terms=[],
        show_progress=False,
        sparse_tol=float(sparse_tol),
        sparse_maxiter=sparse_maxiter,
        use_sz_block=False,
        use_tau_z_block=True,
        target_tz2=0,
        use_z2_block=bool(case.get("use_z2", False)),
        z2_generator="spin_flip" if bool(case.get("use_z2", False)) else None,
        z2_target_parity=0,
        use_translation_block=False,
    )
    return spectrum


def _run_benchmark_case(
    case: Dict[str, Any],
    geometry: Any,
    model_spec: Any,
    params: Dict[str, Any],
    output_folder: str,
) -> Dict[str, Any]:
    case_folder = os.path.join(output_folder, case["id"])
    configure_profiling(
        enabled=True,
        timing=True,
        memory=True,
        cprofile_enabled=bool(params.get("profile_cprofile", False)),
        line_hooks=False,
        scan_points=False,
        output_json=True,
        output_folder=case_folder,
    )
    update_profile_metadata(
        benchmark_case=case["id"],
        requested_backend=case["ed_backend"],
        requested_ed_backend=case["ed_backend"],
        requested_ed_symmetry_engine=case["ed_symmetry_engine"],
    )
    start = time.perf_counter()
    try:
        if case["runner"] == "standard_projector":
            spectrum = _run_standard_projector_benchmark_case(
                geometry,
                model_spec,
                case,
                alpha=float(params["alpha"]),
                beta=float(params["beta"]),
                coupling_j=float(params["coupling_j"]),
                sparse_tol=float(params["sparse_tol"]),
                sparse_maxiter=params["sparse_maxiter"],
            )
        elif case["runner"] == "quspin":
            spectrum = _run_quspin_benchmark_case(
                geometry,
                model_spec,
                case,
                alpha=float(params["alpha"]),
                beta=float(params["beta"]),
                coupling_j=float(params["coupling_j"]),
                sparse_tol=float(params["sparse_tol"]),
                sparse_maxiter=params["sparse_maxiter"],
            )
        else:
            raise ValueError(f"Unknown benchmark runner: {case['runner']}")
        elapsed = time.perf_counter() - start
        summary = {
            "benchmark_case": case,
            "parameters": dict(params),
            "geometry": {
                "lattice": params["lattice"],
                "length_x": int(params["length_x"]),
                "length_y": int(params["length_y"]),
                "circumference_x": True,
                "circumference_y": True,
                "number_of_sites": int(geometry.number_of_sites),
            },
            "ed": {
                "status": "completed",
                "requested_backend": case["ed_backend"],
                "actual_backend": case["ed_backend"],
                "symmetry_engine": case["ed_symmetry_engine"],
                "spectrum": spectrum,
                **spectrum,
            },
        }
        profile = finalize_profiling(summary, output_folder=case_folder, include_environment_audit=False)
        case_result = _benchmark_case_summary(case, spectrum, profile, elapsed)
    except Exception as exc:
        elapsed = time.perf_counter() - start
        summary = {
            "benchmark_case": case,
            "parameters": dict(params),
            "ed": {
                "status": "failed",
                "requested_backend": case["ed_backend"],
                "symmetry_engine": case["ed_symmetry_engine"],
                "error": str(exc),
            },
        }
        profile = finalize_profiling(summary, output_folder=case_folder, include_environment_audit=False)
        case_result = {
            "case_id": case["id"],
            "label": case["label"],
            "status": "failed",
            "error": str(exc),
            "wall_time_seconds": float(profile.get("wall_time_seconds", elapsed)),
            "measured_elapsed_seconds": float(elapsed),
            "stage_timing": _benchmark_stage_table(profile),
            "top_stage": _benchmark_top_stage(profile),
            "cprofile": profile.get("cprofile") if isinstance(profile, dict) else None,
        }
    _benchmark_write_json(os.path.join(case_folder, "benchmark_case_summary.json"), case_result)
    return case_result


def _benchmark_recommendation_for_stage(stage: str, fraction: float) -> str:
    stage_lower = stage.lower()
    prefix = f"Measured bottleneck: `{stage}` ({fraction:.1%} of case runtime)."
    if "diagonalization" in stage_lower:
        return f"{prefix} Prioritize stronger valid symmetry sectors or fewer requested eigenpairs before tuning builders."
    if "c3" in stage_lower:
        return f"{prefix} Cache/reuse the C3 operator and q-sector projectors for the same geometry/Tz sector."
    if "translation" in stage_lower or "orbit" in stage_lower:
        return f"{prefix} Cache translation orbit/projector columns per geometry, momentum, and Tz target."
    if "hamiltonian" in stage_lower:
        return f"{prefix} Focus on sparse Hamiltonian assembly and reusable sparsity patterns across nearby couplings."
    if "basis" in stage_lower:
        return f"{prefix} Cache the Tz parent basis/basis_map for repeated scans on the same site count and target sector."
    if "standard projector" in stage_lower:
        return f"{prefix} Reuse projector bases when only solver settings or nearby couplings change."
    return f"{prefix} Treat this stage as the first optimization target; no unrelated changes are recommended from this run."


def _write_benchmark_acceleration_report(
    results: List[Dict[str, Any]],
    output_folder: str,
    params: Dict[str, Any],
) -> str:
    completed = [row for row in results if row.get("status") == "completed"]
    ranked = sorted(completed, key=lambda row: float(row.get("wall_time_seconds", float("inf"))))
    baseline = next((row for row in completed if row.get("case_id") == "A_standard_projector_tz"), None)
    baseline_time = float(baseline.get("wall_time_seconds", 0.0)) if baseline else None
    lines: List[str] = [
        "# ED Symmetry Acceleration Benchmark",
        "",
        "This report is generated only from measured benchmark cases. No package versions were changed.",
        "",
        "## Benchmark Setup",
        "",
        f"- Geometry: `{params['lattice']}` Lx={int(params['length_x'])}, Ly={int(params['length_y'])}, pbcX=True, pbcY=True",
        f"- Couplings: alpha={float(params['alpha'])}, beta={float(params['beta'])}, J={float(params['coupling_j'])}",
        f"- cProfile top cumulative functions: {'enabled, top 30' if bool(params.get('profile_cprofile')) else 'disabled'}",
        "",
        "## Ranked Cases",
        "",
        "| Rank | Case | Status | Runtime (s) | Speedup vs A | Parent dim | Projected dim | Max nnz | Top stage |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    rank_by_id = {row["case_id"]: idx + 1 for idx, row in enumerate(ranked)}
    for row in sorted(results, key=lambda item: rank_by_id.get(item.get("case_id"), 10_000)):
        dims = row.get("dimensions", {}) if isinstance(row.get("dimensions"), dict) else {}
        nnz = row.get("sparse_nnz", {}) if isinstance(row.get("sparse_nnz"), dict) else {}
        top = row.get("top_stage", {}) if isinstance(row.get("top_stage"), dict) else {}
        runtime = float(row.get("wall_time_seconds", 0.0) or 0.0)
        speedup = ""
        if baseline_time and runtime > 0.0 and row.get("status") == "completed":
            speedup = f"{baseline_time / runtime:.3f}x"
        lines.append(
            "| {rank} | {case} | {status} | {runtime:.6g} | {speedup} | {parent} | {projected} | {nnz} | {top_stage} |".format(
                rank=rank_by_id.get(row.get("case_id"), "-"),
                case=row.get("label", row.get("case_id")),
                status=row.get("status"),
                runtime=runtime,
                speedup=speedup,
                parent=dims.get("parent_hilbert_dimension", ""),
                projected=dims.get("reduced_or_projected_dimension", ""),
                nnz=nnz.get("max_observed_nnz", ""),
                top_stage=top.get("stage", row.get("error", "")),
            )
        )
    lines.extend(["", "## Measured Bottlenecks", ""])
    for row in completed:
        top = row.get("top_stage", {}) if isinstance(row.get("top_stage"), dict) else {}
        runtime = float(row.get("wall_time_seconds", 0.0) or 0.0)
        top_seconds = float(top.get("total_seconds", 0.0) or 0.0)
        fraction = top_seconds / runtime if runtime > 0.0 else 0.0
        lines.append(f"- **{row.get('label')}**: {_benchmark_recommendation_for_stage(str(top.get('stage', 'unknown')), fraction)}")
    failed = [row for row in results if row.get("status") != "completed"]
    if failed:
        lines.extend(["", "## Failed Or Skipped Cases", ""])
        for row in failed:
            lines.append(f"- **{row.get('label')}**: {row.get('error', 'unknown error')}")
    report_path = os.path.join(output_folder, "acceleration_suggestions.md")
    os.makedirs(output_folder, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines).rstrip() + "\n")
    return report_path


def run_symmetry_benchmark_matrix(
    *,
    output_folder: str = PROFILE_OUTPUT_FOLDER,
    lattice: str = "honeycomb",
    length_x: int = 2,
    length_y: int = 2,
    alpha: float = 1.0,
    beta: float = 0.5,
    coupling_j: float = 1.0,
    sparse_tol: float = 1.0e-10,
    sparse_maxiter: int | None = None,
    profile_cprofile: bool = False,
    include_quspin: bool = True,
    cases: str | Sequence[str] = "all",
) -> Dict[str, Any]:
    """Run the opt-in small ED symmetry benchmark matrix from inside analysis.py."""
    import models

    output_folder = os.path.abspath(os.path.expanduser(str(output_folder)))
    geometry = models.build_lattice_geometry(
        lattice,
        int(length_x),
        length_y=int(length_y),
        circumference_x=True,
        circumference_y=True,
    )
    model_spec = models.build_model_spec("1/2", "1/2", "yao_lee", "z")
    all_cases = _benchmark_cases(include_quspin=bool(include_quspin))
    if isinstance(cases, str) and cases.strip().lower() != "all":
        requested = {item.strip() for item in cases.split(",") if item.strip()}
        all_cases = [case for case in all_cases if case["id"] in requested]
    elif not isinstance(cases, str):
        requested = {str(item) for item in cases}
        all_cases = [case for case in all_cases if case["id"] in requested]
    params: Dict[str, Any] = {
        "output_folder": output_folder,
        "lattice": str(lattice),
        "length_x": int(length_x),
        "length_y": int(length_y),
        "alpha": float(alpha),
        "beta": float(beta),
        "coupling_j": float(coupling_j),
        "sparse_tol": float(sparse_tol),
        "sparse_maxiter": sparse_maxiter,
        "profile_cprofile": bool(profile_cprofile),
        "include_quspin": bool(include_quspin),
        "cases": cases,
    }
    results = [_run_benchmark_case(case, geometry, model_spec, params, output_folder) for case in all_cases]
    matrix_summary = {
        "status": "completed_with_failures" if any(row.get("status") != "completed" for row in results) else "completed",
        "output_folder": output_folder,
        "parameters": params,
        "results": results,
    }
    report_path = _write_benchmark_acceleration_report(results, output_folder, params)
    matrix_summary["acceleration_report"] = report_path
    _benchmark_write_json(os.path.join(output_folder, "benchmark_matrix_summary.json"), matrix_summary)
    return matrix_summary


def symmetry_benchmark_cli_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the small ED symmetry acceleration benchmark matrix.")
    parser.add_argument("--output-folder", default=PROFILE_OUTPUT_FOLDER)
    parser.add_argument("--lattice", default="honeycomb", choices=["honeycomb"])
    parser.add_argument("--length-x", "--length_x", dest="length_x", type=int, default=2)
    parser.add_argument("--length-y", "--length_y", dest="length_y", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--coupling-j", "--coupling_j", dest="coupling_j", type=float, default=1.0)
    parser.add_argument("--sparse-tol", "--sparse_tol", dest="sparse_tol", type=float, default=1.0e-10)
    parser.add_argument("--sparse-maxiter", "--sparse_maxiter", dest="sparse_maxiter", type=int, default=0)
    parser.add_argument("--profile-cprofile", "--profile_cprofile", dest="profile_cprofile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-quspin", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cases", default="all", help="Comma-separated case IDs to run, or all.")
    args = parser.parse_args(argv)
    sparse_maxiter = int(args.sparse_maxiter) if int(args.sparse_maxiter) > 0 else None
    summary = run_symmetry_benchmark_matrix(
        output_folder=args.output_folder,
        lattice=args.lattice,
        length_x=int(args.length_x),
        length_y=int(args.length_y),
        alpha=float(args.alpha),
        beta=float(args.beta),
        coupling_j=float(args.coupling_j),
        sparse_tol=float(args.sparse_tol),
        sparse_maxiter=sparse_maxiter,
        profile_cprofile=bool(args.profile_cprofile),
        include_quspin=bool(args.include_quspin),
        cases=str(args.cases),
    )
    print(f"[benchmark] wrote summary: {os.path.join(summary['output_folder'], 'benchmark_matrix_summary.json')}")
    print(f"[benchmark] wrote report: {summary.get('acceleration_report')}")
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(symmetry_benchmark_cli_main())
