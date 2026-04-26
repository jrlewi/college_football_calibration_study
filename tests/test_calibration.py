"""Tests for cfb_calibration.analysis.calibration."""

from __future__ import annotations

import numpy as np
import pytest

from cfb_calibration.analysis.calibration import (
    brier_score,
    calibration_by_segment,
    calibration_summary,
    expected_calibration_error,
    log_loss,
    maximum_calibration_error,
    reliability_diagram_data,
)
import pandas as pd


class TestReliabilityDiagramData:
    def test_output_shape(self, perfectly_calibrated):
        probs, outcomes = perfectly_calibrated
        rel = reliability_diagram_data(probs, outcomes, n_bins=10)
        assert len(rel.bin_centers) == 10
        assert len(rel.mean_predicted) == 10
        assert len(rel.actual_fraction) == 10
        assert len(rel.counts) == 10

    def test_counts_sum_to_n(self, perfectly_calibrated):
        probs, outcomes = perfectly_calibrated
        rel = reliability_diagram_data(probs, outcomes, n_bins=10)
        assert rel.counts.sum() == len(probs)

    def test_calibration_gap_sign(self, overconfident):
        probs, outcomes = overconfident
        rel = reliability_diagram_data(probs, outcomes, n_bins=10)
        # Overconfident model: most gaps should be positive
        valid = ~np.isnan(rel.calibration_gap)
        positive_gaps = (rel.calibration_gap[valid] > 0).sum()
        assert positive_gaps >= valid.sum() * 0.5

    def test_empty_input(self):
        rel = reliability_diagram_data(np.array([]), np.array([]))
        assert rel.counts.sum() == 0
        assert np.all(np.isnan(rel.mean_predicted))

    def test_nan_handling(self):
        probs = np.array([0.3, np.nan, 0.7, 0.5])
        outcomes = np.array([0.0, 1.0, 1.0, np.nan])
        rel = reliability_diagram_data(probs, outcomes, n_bins=5)
        # Only 2 valid predictions (0.3→0, 0.7→1)
        assert rel.counts.sum() == 2


class TestBrierScore:
    def test_perfect_predictions(self):
        probs = np.array([1.0, 0.0, 1.0, 0.0])
        outcomes = np.array([1.0, 0.0, 1.0, 0.0])
        assert brier_score(probs, outcomes) == pytest.approx(0.0)

    def test_worst_predictions(self):
        probs = np.array([0.0, 1.0, 0.0, 1.0])
        outcomes = np.array([1.0, 0.0, 1.0, 0.0])
        assert brier_score(probs, outcomes) == pytest.approx(1.0)

    def test_fifty_fifty(self, all_fifty_fifty):
        probs, outcomes = all_fifty_fifty
        bs = brier_score(probs, outcomes)
        assert 0.2 < bs < 0.3  # ~0.25 for random outcomes

    def test_nan_returns_nan(self):
        probs = np.array([np.nan, np.nan])
        outcomes = np.array([1.0, 0.0])
        assert np.isnan(brier_score(probs, outcomes))

    def test_perfect_calibration_has_low_brier(self, perfectly_calibrated):
        probs, outcomes = perfectly_calibrated
        bs = brier_score(probs, outcomes)
        # Perfectly calibrated uniform predictions → Brier ≈ 1/6 ≈ 0.167
        assert bs < 0.22


class TestLogLoss:
    def test_confident_correct(self):
        probs = np.array([0.99, 0.01])
        outcomes = np.array([1.0, 0.0])
        ll = log_loss(probs, outcomes)
        assert ll < 0.05

    def test_confident_wrong(self):
        probs = np.array([0.99, 0.01])
        outcomes = np.array([0.0, 1.0])
        ll = log_loss(probs, outcomes)
        assert ll > 4.0

    def test_fifty_fifty_log_loss(self, all_fifty_fifty):
        probs, outcomes = all_fifty_fifty
        ll = log_loss(probs, outcomes)
        assert abs(ll - np.log(2)) < 0.02  # ≈ 0.693


class TestECE:
    def test_perfect_calibration_has_low_ece(self, perfectly_calibrated):
        probs, outcomes = perfectly_calibrated
        rel = reliability_diagram_data(probs, outcomes, n_bins=10)
        ece = expected_calibration_error(rel)
        assert ece < 0.02  # Should be very small with 10k samples

    def test_overconfident_has_positive_ece(self, overconfident):
        probs, outcomes = overconfident
        rel = reliability_diagram_data(probs, outcomes, n_bins=10)
        ece = expected_calibration_error(rel)
        assert ece > 0.05

    def test_all_fifty_fifty_ece(self, all_fifty_fifty):
        probs, outcomes = all_fifty_fifty
        rel = reliability_diagram_data(probs, outcomes, n_bins=10)
        ece = expected_calibration_error(rel)
        # All preds at 0.5, outcomes ~50% → near-perfect in that single bin
        assert ece < 0.05


class TestMCE:
    def test_mce_geq_ece(self, overconfident):
        probs, outcomes = overconfident
        rel = reliability_diagram_data(probs, outcomes, n_bins=10)
        ece = expected_calibration_error(rel)
        mce = maximum_calibration_error(rel)
        assert mce >= ece


class TestCalibrationSummary:
    def test_returns_summary_object(self, perfectly_calibrated):
        probs, outcomes = perfectly_calibrated
        cs = calibration_summary(probs, outcomes)
        assert cs.n == len(probs)
        assert cs.brier_score > 0
        assert cs.ece >= 0

    def test_empty_returns_nan_metrics(self):
        cs = calibration_summary(np.array([]), np.array([]))
        assert cs.n == 0
        assert np.isnan(cs.brier_score)
        assert np.isnan(cs.ece)

    def test_to_dataframe(self, perfectly_calibrated):
        probs, outcomes = perfectly_calibrated
        cs = calibration_summary(probs, outcomes)
        df = cs.reliability.to_dataframe()
        assert len(df) == 10
        assert "mean_predicted" in df.columns


class TestCalibrationBySegment:
    def test_segments_returned(self, perfectly_calibrated):
        probs, outcomes = perfectly_calibrated
        df = pd.DataFrame({
            "home_win_prob": probs,
            "actual_home_win": outcomes,
            "quarter": np.random.choice(["Q1", "Q2", "Q3", "Q4"], len(probs)),
        })
        results = calibration_by_segment(df, "quarter", min_samples=10)
        assert set(results.keys()) == {"Q1", "Q2", "Q3", "Q4"}

    def test_min_samples_filter(self, perfectly_calibrated):
        probs, outcomes = perfectly_calibrated
        df = pd.DataFrame({
            "home_win_prob": probs[:100],
            "actual_home_win": outcomes[:100],
            "quarter": ["Q1"] * 90 + ["Q2"] * 10,
        })
        results = calibration_by_segment(df, "quarter", min_samples=20)
        assert "Q1" in results
        assert "Q2" not in results  # only 10 samples, below threshold
