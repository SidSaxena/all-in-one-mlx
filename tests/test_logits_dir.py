"""Test that activations and embeddings can be written apart from the JSON."""

import tempfile
from pathlib import Path

import numpy as np

from allin1_mlx.helpers import save_results
from allin1_mlx.typings import AnalysisResult, Segment


def _result():
    return AnalysisResult(
        path=Path("/fake/path/test.wav"),
        bpm=120,
        beats=[0.37, 1.23],
        downbeats=[0.37],
        beat_positions=[1, 2],
        segments=[Segment(start=0.0, end=2.0, label="intro")],
        activations={"beat": np.zeros(4, dtype=np.float32)},
        embeddings=np.zeros((2, 3), dtype=np.float32),
    )


def test_arrays_stay_beside_the_json_by_default():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "struct"
        save_results(_result(), out_dir)

        assert (out_dir / "test.json").is_file()
        assert (out_dir / "test.activ.npz").is_file()
        assert (out_dir / "test.embed.npy").is_file()


def test_logits_dir_redirects_the_arrays_only():
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "struct"
        logits_dir = Path(tmp) / "logits"
        save_results(_result(), out_dir, logits_dir)

        assert (out_dir / "test.json").is_file()
        assert not (out_dir / "test.activ.npz").exists()
        assert not (out_dir / "test.embed.npy").exists()
        assert (logits_dir / "test.activ.npz").is_file()
        assert (logits_dir / "test.embed.npy").is_file()


def test_logits_dir_is_created_when_missing():
    with tempfile.TemporaryDirectory() as tmp:
        logits_dir = Path(tmp) / "nested" / "logits"
        save_results(_result(), Path(tmp) / "struct", logits_dir)

        assert logits_dir.is_dir()
