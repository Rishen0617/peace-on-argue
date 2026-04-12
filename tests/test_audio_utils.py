"""
Tests for calm_app/audio_utils.py

Coverage targets
----------------
* resample_to_16k  – passthrough, upsample, downsample, dtype handling
* AudioStream      – context-manager lifecycle, mono conversion, blocksize
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from calm_app.audio_utils import resample_to_16k


# ---------------------------------------------------------------------------
# resample_to_16k
# ---------------------------------------------------------------------------

class TestResampleTo16k:
    """Unit tests for resample_to_16k(audio, orig_sr, target_sr)."""

    def test_passthrough_when_already_target_rate(self):
        audio = np.array([0.1, 0.2, -0.3], dtype=np.float32)
        result = resample_to_16k(audio, orig_sr=16000)
        np.testing.assert_array_equal(result, audio)

    def test_passthrough_returns_float32(self):
        audio = np.zeros(100, dtype=np.float64)
        result = resample_to_16k(audio, orig_sr=16000)
        assert result.dtype == np.float32

    def test_upsampling_output_is_float32(self):
        audio = np.zeros(800, dtype=np.float32)
        result = resample_to_16k(audio, orig_sr=8000, target_sr=16000)
        assert result.dtype == np.float32

    def test_downsampling_output_is_float32(self):
        audio = np.zeros(3200, dtype=np.float32)
        result = resample_to_16k(audio, orig_sr=32000, target_sr=16000)
        assert result.dtype == np.float32

    def test_upsampling_roughly_doubles_length(self):
        # 800 samples at 8 kHz → ~1600 samples at 16 kHz
        audio = np.zeros(800, dtype=np.float32)
        result = resample_to_16k(audio, orig_sr=8000, target_sr=16000)
        assert len(result) == pytest.approx(1600, rel=0.05)

    def test_downsampling_roughly_halves_length(self):
        # 3200 samples at 32 kHz → ~1600 samples at 16 kHz
        audio = np.zeros(3200, dtype=np.float32)
        result = resample_to_16k(audio, orig_sr=32000, target_sr=16000)
        assert len(result) == pytest.approx(1600, rel=0.05)

    def test_custom_target_sample_rate(self):
        # 1600 samples at 16 kHz → ~800 samples at 8 kHz
        audio = np.zeros(1600, dtype=np.float32)
        result = resample_to_16k(audio, orig_sr=16000, target_sr=8000)
        assert len(result) == pytest.approx(800, rel=0.05)

    def test_float64_input_is_accepted_and_converted(self):
        audio = np.zeros(3200, dtype=np.float64)
        result = resample_to_16k(audio, orig_sr=32000, target_sr=16000)
        assert result.dtype == np.float32

    def test_1d_output(self):
        audio = np.zeros(800, dtype=np.float32)
        result = resample_to_16k(audio, orig_sr=8000, target_sr=16000)
        assert result.ndim == 1


# ---------------------------------------------------------------------------
# AudioStream
# ---------------------------------------------------------------------------

class TestAudioStream:
    """Tests for AudioStream context manager with sounddevice mocked."""

    @patch("calm_app.audio_utils.sd")
    def test_enter_creates_and_starts_inputstream(self, mock_sd):
        from calm_app.audio_utils import AudioStream

        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        with AudioStream(samplerate=16000, block_seconds=0.48):
            mock_sd.InputStream.assert_called_once_with(
                channels=1,
                samplerate=16000,
                blocksize=7680,       # int(0.48 * 16000)
                dtype="float32",
                latency="low",
            )
            mock_stream.start.assert_called_once()

    @patch("calm_app.audio_utils.sd")
    def test_exit_stops_and_closes_stream(self, mock_sd):
        from calm_app.audio_utils import AudioStream

        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        with AudioStream():
            pass

        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()

    @patch("calm_app.audio_utils.sd")
    def test_exit_does_not_raise_when_stream_stop_fails(self, mock_sd):
        from calm_app.audio_utils import AudioStream

        mock_stream = MagicMock()
        mock_stream.stop.side_effect = Exception("device disconnected")
        mock_sd.InputStream.return_value = mock_stream

        # __exit__ must silently absorb the error
        with AudioStream():
            pass

    @patch("calm_app.audio_utils.sd")
    def test_read_block_returns_1d_float32_array(self, mock_sd):
        from calm_app.audio_utils import AudioStream

        mock_stream = MagicMock()
        mock_stream.read.return_value = (
            np.zeros((7680, 1), dtype=np.float32),
            None,
        )
        mock_sd.InputStream.return_value = mock_stream

        with AudioStream(samplerate=16000, block_seconds=0.48) as s:
            block = s.read_block()

        assert block.ndim == 1
        assert block.dtype == np.float32
        assert len(block) == 7680

    @patch("calm_app.audio_utils.sd")
    def test_read_block_averages_stereo_to_mono(self, mock_sd):
        """Left channel = 1.0, right = 0.0 → averaged output = 0.5."""
        from calm_app.audio_utils import AudioStream

        stereo = np.zeros((100, 2), dtype=np.float32)
        stereo[:, 0] = 1.0  # left channel
        stereo[:, 1] = 0.0  # right channel

        mock_stream = MagicMock()
        mock_stream.read.return_value = (stereo, None)
        mock_sd.InputStream.return_value = mock_stream

        with AudioStream(samplerate=16000, block_seconds=0.1) as s:
            block = s.read_block()

        assert block.ndim == 1
        np.testing.assert_allclose(block, 0.5, atol=1e-6)

    @patch("calm_app.audio_utils.sd")
    def test_blocksize_computed_correctly(self, mock_sd):
        from calm_app.audio_utils import AudioStream

        mock_sd.InputStream.return_value = MagicMock()

        s1 = AudioStream(samplerate=16000, block_seconds=0.5)
        assert s1.blocksize == 8000

        s2 = AudioStream(samplerate=44100, block_seconds=1.0)
        assert s2.blocksize == 44100

    @patch("calm_app.audio_utils.sd")
    def test_stream_is_none_before_enter(self, mock_sd):
        from calm_app.audio_utils import AudioStream

        s = AudioStream()
        assert s.stream is None
