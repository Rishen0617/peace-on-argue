"""
Global pytest configuration.

Stubs out heavy external dependencies (TensorFlow, TensorFlow Hub) before any
test can trigger an import of calm_app.main, so the suite runs without GPU
hardware, internet access, or a real YAMNet model download.
"""

import io
import sys
from unittest.mock import MagicMock

import numpy as np

# ---------------------------------------------------------------------------
# Minimal YAMNet class-map CSV with a handful of relevant labels.
# Indices deliberately match the logic exercised in test_main.py.
# ---------------------------------------------------------------------------
_CLASS_MAP_CSV = (
    "index,mid,display_name\n"
    "0,/m/000,Speech\n"
    "1,/m/001,Screaming\n"
    "2,/m/002,Yelling\n"
    "3,/m/003,Music\n"
    "4,/m/004,Silence\n"
)


class _FakeGFile:
    """Minimal stand-in for tf.io.gfile.GFile that returns our fake CSV."""

    def __init__(self, path):
        pass  # ignore path

    def __enter__(self):
        return io.StringIO(_CLASS_MAP_CSV)

    def __exit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# Build a yamnet callable that returns zero scores for all frames/classes.
# Tests that need specific score values can patch calm_app.main.yamnet
# directly via unittest.mock.patch.object.
# ---------------------------------------------------------------------------
_fake_scores = np.zeros((5, 5), dtype=np.float32)  # 5 frames × 5 classes

_mock_yamnet = MagicMock()
_mock_yamnet.return_value = (
    MagicMock(numpy=lambda: _fake_scores.copy()),  # scores tensor
    MagicMock(),                                    # embeddings tensor
    MagicMock(),                                    # spectrogram tensor
)
_mock_yamnet.class_map_path.return_value = MagicMock(
    numpy=lambda: b"/fake/class_map.csv"
)

# ---------------------------------------------------------------------------
# Inject stubs into sys.modules *before* any calm_app module is imported.
# setdefault means we only install a stub if the real library is absent,
# preventing accidental double-stubs in environments where TF is installed.
# ---------------------------------------------------------------------------
_mock_hub = MagicMock()
_mock_hub.load.return_value = _mock_yamnet
sys.modules.setdefault("tensorflow_hub", _mock_hub)

_mock_tf = MagicMock()
_mock_tf.io.gfile.GFile = _FakeGFile
_mock_tf.convert_to_tensor = lambda x, dtype=None: x
sys.modules.setdefault("tensorflow", _mock_tf)

# sounddevice – not available in CI / no-hardware environments
sys.modules.setdefault("sounddevice", MagicMock())

# pygame – requires SDL which is unavailable in headless CI environments
sys.modules.setdefault("pygame", MagicMock())
