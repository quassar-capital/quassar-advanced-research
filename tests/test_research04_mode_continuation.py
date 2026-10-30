
"""Validation of local gravitational angular mode continuation.

These tests verify finite-dimensional tracking and rejection safeguards.
They do not certify physical Kerr resonances or global mode continuation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

RESEARCH_DIR = Path(__file__).resolve().parents[1] / "research_04"
sys.path.insert(0, str(RESEARCH_DIR))

from angular_solver import gravitational_angular_eigenvalue
from mode_continuation import continue_gravitational_angular_mode


def test_spherical_reference_and_local_continuation():
    result = continue_gravitational_angular_mode(
        2,
        2,
        [0j, 0.001 + 0j],
        truncation_ell=12,
    )

    assert result.spin_weight == -2
    assert result.ell == 2
    assert result.azimuthal_number == 2
    assert len(result.steps) == 2
    assert result.steps[0].eigenvalue == pytest.approx(4.0)
    assert result.steps[0].overlap == pytest.approx(1.0)
    assert result.steps[1].overlap > 0.99
    assert result.steps[1].relative_residual < 1e-12


def test_complex_path_matches_independent_angular_selection():
    target = 0.02 + 0.01j

    tracked = continue_gravitational_angular_mode(
        2,
        2,
        [0j, 0.01 + 0.005j, target],
        truncation_ell=12,
    )

    reference = gravitational_angular_eigenvalue(
        2,
        2,
        target,
        truncation_ell=12,
        convergence_extension=1,
    )

    assert len(tracked.steps) == 3
    assert tracked.steps[-1].eigenvalue == pytest.approx(
        reference.eigenvalue,
        rel=1e-8,
        abs=1e-10,
    )
    assert all(
        step.relative_residual < 1e-12
        for step in tracked.steps
    )


def test_invalid_paths_are_rejected():
    invalid_paths = (
        [0j],
        [0.01 + 0j, 0.02 + 0j],
        [0j, 0j],
        [0j, complex(float("nan"), 0.0)],
    )

    for path in invalid_paths:
        with pytest.raises(ValueError):
            continue_gravitational_angular_mode(
                2,
                2,
                path,
            )


def test_conservative_tracking_rejects_unresolved_steps():
    with pytest.raises(
        ArithmeticError,
        match="ambiguous eigenvector identification",
    ):
        continue_gravitational_angular_mode(
            2,
            2,
            [0j, 0.1 + 0j],
            truncation_ell=12,
            minimum_overlap=0.9999999999,
        )
