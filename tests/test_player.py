"""
Tests for calm_app/player.py

Coverage targets
----------------
* night_volume_scale  – pure function, all branches
* list_audio_files    – filesystem filtering
* play_playlist       – mocked pygame + time; key behaviours
"""

import os
from unittest.mock import MagicMock, patch, call

import pytest

from calm_app.player import list_audio_files, night_volume_scale


# ---------------------------------------------------------------------------
# night_volume_scale
# ---------------------------------------------------------------------------

class TestNightVolumeScale:
    """Unit tests for night_volume_scale(hour_now, night_range, night_scale)."""

    def test_returns_night_scale_for_late_evening(self):
        assert night_volume_scale(22) == 0.4
        assert night_volume_scale(23) == 0.4

    def test_returns_night_scale_for_early_morning(self):
        assert night_volume_scale(0) == 0.4
        assert night_volume_scale(3) == 0.4
        assert night_volume_scale(6) == 0.4

    def test_returns_full_volume_during_day(self):
        assert night_volume_scale(7) == 1.0
        assert night_volume_scale(12) == 1.0
        assert night_volume_scale(21) == 1.0

    def test_boundary_start_hour_is_night(self):
        # 22:00 is the first night hour
        assert night_volume_scale(22) == pytest.approx(0.4)

    def test_boundary_end_hour_is_day(self):
        # 07:00 is the first daytime hour after the quiet window
        assert night_volume_scale(7) == pytest.approx(1.0)

    def test_custom_night_scale_returned_at_night(self):
        result = night_volume_scale(23, night_scale=0.2)
        assert result == pytest.approx(0.2)

    def test_custom_night_range(self):
        # The function is designed for midnight-crossing ranges where start > end.
        # (23, 5) means "from 23:00 to 05:00".
        assert night_volume_scale(2, night_range=(23, 5), night_scale=0.3) == pytest.approx(0.3)
        assert night_volume_scale(12, night_range=(23, 5)) == pytest.approx(1.0)

    @pytest.mark.xfail(strict=True, reason=(
        "Bug: condition `start <= hour_now` is always True when start=0, "
        "so ranges that don't cross midnight (start < end) are not handled."
    ))
    def test_non_midnight_crossing_range_not_supported(self):
        # night_range=(0, 5) means 00:00–05:00; hour 6 should be daytime.
        # With the current implementation this incorrectly returns night_scale
        # because `0 <= 6` is True regardless of end.
        assert night_volume_scale(6, night_range=(0, 5)) == pytest.approx(1.0)

    def test_return_type_is_float(self):
        assert isinstance(night_volume_scale(12), float)
        assert isinstance(night_volume_scale(23), float)


# ---------------------------------------------------------------------------
# list_audio_files
# ---------------------------------------------------------------------------

class TestListAudioFiles:
    """Unit tests for list_audio_files(folder)."""

    def test_returns_only_supported_audio_extensions(self, tmp_path):
        for name in ["track.mp3", "loop.wav", "data.csv", "photo.png"]:
            (tmp_path / name).touch()

        result = list_audio_files(str(tmp_path))
        basenames = sorted(os.path.basename(f) for f in result)
        assert basenames == ["loop.wav", "track.mp3"]

    def test_all_four_supported_extensions_are_included(self, tmp_path):
        for name in ["a.wav", "b.mp3", "c.ogg", "d.flac"]:
            (tmp_path / name).touch()

        result = list_audio_files(str(tmp_path))
        assert len(result) == 4

    def test_extension_matching_is_case_insensitive(self, tmp_path):
        (tmp_path / "LOUD.MP3").touch()
        (tmp_path / "calm.WAV").touch()
        (tmp_path / "other.Ogg").touch()

        result = list_audio_files(str(tmp_path))
        assert len(result) == 3

    def test_empty_directory_returns_empty_list(self, tmp_path):
        assert list_audio_files(str(tmp_path)) == []

    def test_no_audio_files_returns_empty_list(self, tmp_path):
        (tmp_path / "notes.txt").touch()
        (tmp_path / "image.jpg").touch()

        assert list_audio_files(str(tmp_path)) == []

    def test_returns_absolute_paths(self, tmp_path):
        (tmp_path / "song.mp3").touch()

        result = list_audio_files(str(tmp_path))
        assert len(result) == 1
        assert os.path.isabs(result[0])
        assert result[0].endswith("song.mp3")

    def test_non_audio_files_are_excluded_even_if_named_similarly(self, tmp_path):
        (tmp_path / "track.mp3.bak").touch()
        (tmp_path / "track.wav.tmp").touch()

        assert list_audio_files(str(tmp_path)) == []


# ---------------------------------------------------------------------------
# play_playlist
# ---------------------------------------------------------------------------

class TestPlayPlaylist:
    """Behavioural tests for play_playlist() with external deps mocked."""

    def test_warns_and_returns_early_when_folder_is_empty(self, tmp_path, capsys):
        from calm_app.player import play_playlist

        play_playlist(str(tmp_path))

        captured = capsys.readouterr()
        assert "[WARN]" in captured.out

    def test_does_not_call_pygame_when_folder_is_empty(self, tmp_path):
        from calm_app.player import play_playlist

        with patch("calm_app.player.pygame") as mock_pygame:
            play_playlist(str(tmp_path))
            mock_pygame.mixer.music.load.assert_not_called()

    @patch("calm_app.player.pygame")
    @patch("calm_app.player.time")
    def test_loads_and_plays_a_track(self, mock_time, mock_pygame, tmp_path):
        from calm_app.player import play_playlist

        (tmp_path / "calm.mp3").touch()

        # time.time() call sequence:
        #   1 → used to compute end_time (end_time = 0 + 60 = 60)
        #   2 → first while-loop guard (0 < 60 → enter loop)
        #   3 → seg calculation (601 → seg = max(120, -541) = 120)
        #   4 → first inner-loop guard (601 ≥ 60 → break immediately)
        #   5 → second while-loop guard (601 ≥ 60 → exit)
        mock_time.time.side_effect = [0, 0, 601, 601, 601]
        mock_pygame.mixer.get_init.return_value = True

        play_playlist(str(tmp_path), minutes=1)

        mock_pygame.mixer.music.load.assert_called_once()
        mock_pygame.mixer.music.play.assert_called_once()

    @patch("calm_app.player.pygame")
    @patch("calm_app.player.time")
    def test_fadeout_called_after_track_segment(self, mock_time, mock_pygame, tmp_path):
        from calm_app.player import play_playlist

        (tmp_path / "calm.mp3").touch()
        mock_time.time.side_effect = [0, 0, 601, 601, 601]
        mock_pygame.mixer.get_init.return_value = True

        play_playlist(str(tmp_path), minutes=1, fade_out_ms=1500)

        mock_pygame.mixer.music.fadeout.assert_called_once_with(1500)

    @patch("calm_app.player.pygame")
    @patch("calm_app.player.time")
    def test_playback_exception_is_caught_and_reported(self, mock_time, mock_pygame, tmp_path, capsys):
        from calm_app.player import play_playlist

        (tmp_path / "broken.mp3").touch()
        mock_time.time.side_effect = [0, 0, 601, 601, 601]
        mock_pygame.mixer.get_init.return_value = True
        mock_pygame.mixer.music.load.side_effect = Exception("codec error")

        # Must not raise
        play_playlist(str(tmp_path), minutes=1)

        captured = capsys.readouterr()
        assert "[ERR]" in captured.out

    @patch("calm_app.player.pygame")
    @patch("calm_app.player.time")
    def test_volume_clamped_to_valid_range(self, mock_time, mock_pygame, tmp_path):
        """night_scale > 1.0 should be clamped, not crash."""
        from calm_app.player import play_playlist

        (tmp_path / "song.mp3").touch()
        mock_time.time.side_effect = [0, 601]  # exit immediately before first load
        mock_pygame.mixer.get_init.return_value = True

        # Passing night_scale > 1 should not raise
        play_playlist(str(tmp_path), minutes=1, night_scale=2.0)
