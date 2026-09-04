"""Data-free checks for reproducibility and terminal objective helpers."""

import random

import numpy as np
import pytest
import torch

from cecoppo.utils import (
    effective_makespan_for_tri,
    set_seed,
    tri_objective_normalized_terms,
    tri_objective_scalar,
    tri_objective_weighted_scalar,
)


def test_set_seed_reproduces_python_numpy_and_torch_streams():
    set_seed(17)
    first = (random.random(), np.random.random(), torch.rand(3))
    set_seed(17)
    second = (random.random(), np.random.random(), torch.rand(3))

    assert first[0] == second[0]
    assert first[1] == second[1]
    torch.testing.assert_close(first[2], second[2], rtol=0.0, atol=0.0)


def test_normalized_terms_use_explicit_refs_and_clamp_negative_metrics():
    assert tri_objective_normalized_terms(
        20.0,
        10.0,
        50.0,
        ref_makespan_sec=10.0,
        ref_slr=5.0,
        ref_load_balance=100.0,
    ) == pytest.approx((2.0, 2.0, 0.5))
    assert tri_objective_normalized_terms(-1.0, -2.0, -3.0) == (0.0, 0.0, 0.0)


def test_weighted_objective_ignores_negative_weights_and_normalizes_positive_ones():
    score = tri_objective_weighted_scalar(
        20.0,
        10.0,
        50.0,
        -1.0,
        3.0,
        1.0,
        ref_makespan_sec=10.0,
        ref_slr=5.0,
        ref_load_balance=100.0,
    )
    assert score == pytest.approx((3.0 * 2.0 + 1.0 * 0.5) / 4.0)

    fallback = tri_objective_weighted_scalar(
        20.0,
        10.0,
        50.0,
        0.0,
        0.0,
        0.0,
        ref_makespan_sec=10.0,
        ref_slr=5.0,
        ref_load_balance=100.0,
    )
    assert fallback == pytest.approx(
        tri_objective_scalar(
            20.0,
            10.0,
            50.0,
            ref_makespan_sec=10.0,
            ref_slr=5.0,
            ref_load_balance=100.0,
        )
    )


def test_effective_makespan_clips_blend_to_mean_max_range():
    assert effective_makespan_for_tri(80.0, 140.0, -1.0) == 80.0
    assert effective_makespan_for_tri(80.0, 140.0, 0.25) == 95.0
    assert effective_makespan_for_tri(80.0, 140.0, 2.0) == 140.0
