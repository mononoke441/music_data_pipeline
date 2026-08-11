from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from metadata_merge import merge_record


def test_merge_by_audio_id_preserves_audio_path():
    target = {"audio_id": "track-1"}
    conflicts = {}

    merge_record(
        target,
        {
            "audio_id": "track-1",
            "audio_path": "/data/music.flac",
            "duration": 12.5,
        },
        conflicts,
        "audio_id",
    )

    assert target == {
        "audio_id": "track-1",
        "audio_path": "/data/music.flac",
        "duration": 12.5,
    }
    assert conflicts == {}
