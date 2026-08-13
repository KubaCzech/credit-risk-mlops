import numpy as np
import pandas as pd

from ml.monitoring import (
    MIN_SAMPLE_SIZE,
    PSI_MODERATE_THRESHOLD,
    PSI_SIGNIFICANT_THRESHOLD,
    _reference_bins,
    compute_psi,
    psi_status,
)


class TestReferenceBins:
    def test_uniform_data_gives_roughly_equal_proportions(self):
        values = pd.Series(np.arange(1000))
        reference = _reference_bins(values)
        assert len(reference["bin_edges"]) == 11  # 10 bins -> 11 edges
        assert all(0.08 < p < 0.12 for p in reference["reference_proportions"])

    def test_degenerate_column_collapses_to_fewer_bins(self):
        # matches NumberOfTimes90DaysLate in the real data: ~96% zeros, so most quantile
        # edges land on the same value and dedupe away - see monitoring_reference.json
        values = pd.Series([0] * 950 + [1] * 30 + [2] * 20)
        reference = _reference_bins(values)
        assert len(reference["bin_edges"]) < 11

    def test_proportions_sum_to_one(self):
        values = pd.Series(np.random.default_rng(0).exponential(size=500))
        reference = _reference_bins(values)
        assert sum(reference["reference_proportions"]) == 1.0


class TestComputePSI:
    def test_identical_distribution_scores_near_zero(self):
        reference = _reference_bins(pd.Series(np.arange(1000)))
        psi = compute_psi(reference["bin_edges"], reference["reference_proportions"], np.arange(1000))
        assert psi < 0.01

    def test_shifted_distribution_scores_high(self):
        # reference is uniform 0-999; "current" is everything jammed into the top decile
        reference = _reference_bins(pd.Series(np.arange(1000)))
        current = np.full(200, 999)
        psi = compute_psi(reference["bin_edges"], reference["reference_proportions"], current)
        assert psi > PSI_SIGNIFICANT_THRESHOLD

    def test_moderately_shifted_distribution_lands_between_thresholds(self):
        rng = np.random.default_rng(0)
        reference_values = rng.normal(loc=0, scale=1, size=2000)
        reference = _reference_bins(pd.Series(reference_values))
        # a real but partial shift: same spread, mean nudged by ~0.5 std
        current = rng.normal(loc=0.5, scale=1, size=500)
        psi = compute_psi(reference["bin_edges"], reference["reference_proportions"], current)
        assert PSI_MODERATE_THRESHOLD < psi < PSI_SIGNIFICANT_THRESHOLD


class TestPSIStatus:
    def test_thresholds(self):
        assert psi_status(0.0) == "stable"
        assert psi_status(PSI_MODERATE_THRESHOLD) == "moderate_shift"
        assert psi_status(PSI_SIGNIFICANT_THRESHOLD) == "significant_shift"
        assert psi_status(1.0) == "significant_shift"


def test_min_sample_size_is_a_reasonable_positive_threshold():
    # not a deep property, but a guard against an accidental 0/negative edit that would
    # silently make every drift check "sufficient"
    assert MIN_SAMPLE_SIZE > 0
