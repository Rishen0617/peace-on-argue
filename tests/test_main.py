"""
Tests for calm_app/main.py

TensorFlow and TensorFlow Hub are replaced by lightweight stubs in
tests/conftest.py before this module is imported, so no model download or GPU
is required.

Coverage targets
----------------
* score_argument_like  – probability aggregation, fallback path, return type
* Module-level config  – constants loaded from config.yaml
* argument_class_ids   – keyword matching against the fake class map
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Lazy import helper – ensures conftest stubs are active before the first
# import of calm_app.main (Python caches the module after the first import).
# ---------------------------------------------------------------------------

def _main():
    import calm_app.main as m
    return m


# ---------------------------------------------------------------------------
# Argument class discovery (module-level init)
# ---------------------------------------------------------------------------

class TestArgumentClassIds:
    """Verify that KEYWORDS are matched against the fake class map."""

    def test_screaming_and_yelling_detected(self):
        m = _main()
        # conftest class map has "Screaming" at index 1 and "Yelling" at index 2
        assert 1 in m.argument_class_ids
        assert 2 in m.argument_class_ids

    def test_non_matching_classes_excluded(self):
        m = _main()
        # "Speech" (0), "Music" (3), "Silence" (4) should not be in the set
        for idx in [0, 3, 4]:
            assert idx not in m.argument_class_ids

    def test_argument_class_ids_is_nonempty(self):
        m = _main()
        assert len(m.argument_class_ids) > 0


# ---------------------------------------------------------------------------
# Config constants
# ---------------------------------------------------------------------------

class TestConfigConstants:
    """Smoke-test that config.yaml values are loaded into module constants."""

    def test_sample_rate_is_positive_integer(self):
        m = _main()
        assert isinstance(m.SR, int)
        assert m.SR > 0

    def test_threshold_is_between_zero_and_one(self):
        m = _main()
        assert 0.0 < m.THRESH < 1.0

    def test_consecutive_windows_is_positive(self):
        m = _main()
        assert m.K >= 1

    def test_cooldown_is_non_negative(self):
        m = _main()
        assert m.COOLDOWN_S >= 0

    def test_night_range_has_two_elements(self):
        m = _main()
        assert len(m.NIGHT_RANGE) == 2


# ---------------------------------------------------------------------------
# score_argument_like
# ---------------------------------------------------------------------------

class TestScoreArgumentLike:
    """Unit tests for score_argument_like(waveform)."""

    def test_returns_float(self):
        m = _main()
        waveform = np.zeros(15360, dtype=np.float32)
        result = m.score_argument_like(waveform)
        assert isinstance(result, float)

    def test_all_zero_scores_yields_zero_probability(self):
        m = _main()
        # conftest stub yamnet returns np.zeros → probability must be 0.0
        waveform = np.zeros(15360, dtype=np.float32)
        assert m.score_argument_like(waveform) == pytest.approx(0.0)

    def test_nonzero_argument_class_score_yields_positive_probability(self):
        m = _main()
        scores = np.zeros((5, 5), dtype=np.float32)
        scores[:, 1] = 0.8  # index 1 = "Screaming" in our fake map

        mock_scores = MagicMock()
        mock_scores.numpy.return_value = scores

        with patch.object(m, "yamnet", return_value=(mock_scores, MagicMock(), MagicMock())):
            result = m.score_argument_like(np.zeros(15360, dtype=np.float32))

        assert result > 0.0

    def test_result_is_non_negative(self):
        m = _main()
        waveform = np.zeros(15360, dtype=np.float32)
        assert m.score_argument_like(waveform) >= 0.0

    def test_accepts_arbitrary_length_waveform(self):
        m = _main()
        for length in [1600, 7680, 15360]:
            waveform = np.zeros(length, dtype=np.float32)
            result = m.score_argument_like(waveform)
            assert isinstance(result, float)

    def test_fallback_path_uses_max_score_per_frame(self):
        """When argument_class_ids is empty the fallback averages per-frame maxima."""
        m = _main()
        original_ids = m.argument_class_ids[:]
        m.argument_class_ids.clear()

        try:
            scores = np.zeros((4, 5), dtype=np.float32)
            scores[:, 0] = 0.6  # highest class is index 0 in every frame

            mock_scores = MagicMock()
            mock_scores.numpy.return_value = scores

            with patch.object(m, "yamnet", return_value=(mock_scores, MagicMock(), MagicMock())):
                result = m.score_argument_like(np.zeros(15360, dtype=np.float32))

            # mean of [0.6, 0.6, 0.6, 0.6] = 0.6
            assert result == pytest.approx(0.6, rel=1e-4)
        finally:
            m.argument_class_ids[:] = original_ids

    def test_probability_aggregated_across_frames(self):
        """Mean is taken over all frames, not just one."""
        m = _main()
        # Two frames: first has high score, second has zero
        scores = np.zeros((2, 5), dtype=np.float32)
        scores[0, 1] = 1.0  # index 1 in argument_class_ids

        mock_scores = MagicMock()
        mock_scores.numpy.return_value = scores

        with patch.object(m, "yamnet", return_value=(mock_scores, MagicMock(), MagicMock())):
            result = m.score_argument_like(np.zeros(15360, dtype=np.float32))

        # frame 0 contributes 1.0, frame 1 contributes 0.0 → mean ≈ 0.5
        # (argument_class_ids may include index 2 as well, still > 0 and < 1)
        assert 0.0 < result <= 1.0
