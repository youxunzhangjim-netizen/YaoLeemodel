"""Small checks for named external-field axis presets."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import (  # noqa: E402
    classify_external_field,
    external_field_display_label,
    external_field_filename_label,
    resolve_field_vector,
)


def test_axis_001_uses_shared_strength_as_pure_hz():
    vector = resolve_field_vector("001", 2.0, hx=9.0, hy=9.0, hz=9.0)
    assert vector == (0.0, 0.0, 2.0)
    assert classify_external_field("hamiltonian", vector)["field_class"] == "hz"
    assert external_field_display_label("hamiltonian", "001", vector) == "|H|=2.000, axis=[0,0,1]"
    assert external_field_filename_label("hamiltonian", "001", vector) == "H2p000axis001"


def test_axis_111_keeps_normalized_direction_with_same_strength_option():
    vector = resolve_field_vector("111", 3.0, hx=0.0, hy=0.0, hz=0.0)
    assert abs(sum(component * component for component in vector) ** 0.5 - 3.0) < 1.0e-12
    assert classify_external_field("hamiltonian", vector)["field_class"] == "h111"
    assert external_field_display_label("hamiltonian", "111", vector) == "|H|=3.000, axis=[1,1,1]"
    assert external_field_filename_label("hamiltonian", "111", vector) == "H3p000axis111"


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            suite.addTest(unittest.FunctionTestCase(value))
    return suite


if __name__ == "__main__":
    unittest.main()
