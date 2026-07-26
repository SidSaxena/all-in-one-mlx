"""Test that the demucs random time shift can be seeded."""

from pathlib import Path

from allin1_mlx.demix import demix


class _Separator:
    samplerate = 44100
    seen = []

    def __init__(self, model, progress, seed=None):
        type(self).seen.append(seed)

    def separate_audio_file(self, path):
        return None, {}


def _patch(monkeypatch):
    import demucs_mlx.api as api

    _Separator.seen = []
    monkeypatch.setattr(api, "Separator", _Separator)
    monkeypatch.setattr(api, "save_audio", lambda audio, dst, sr: None)


def test_seed_reaches_the_separator(monkeypatch, tmp_path):
    _patch(monkeypatch)
    demix([Path("/fake/a.wav")], tmp_path, seed=1234)
    assert _Separator.seen == [1234]


def test_unseeded_by_default(monkeypatch, tmp_path):
    _patch(monkeypatch)
    demix([Path("/fake/a.wav")], tmp_path)
    assert _Separator.seen == [None]
