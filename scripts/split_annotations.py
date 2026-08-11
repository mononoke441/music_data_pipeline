#!/usr/bin/env python3
"""Convert a legacy annotated JSONL into the per-audio JSON layout."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterator

from annotation_storage import publish_annotation_records, source_relpath_for_audio
from pipeline_core import iter_jsonl


def converted_records(input_jsonl: str, input_root: str) -> Iterator[Dict[str, Any]]:
    for source in iter_jsonl(input_jsonl):
        record = dict(source)
        if not record.get("source_relpath"):
            audio_path = str(record.get("audio_path") or "")
            if not audio_path:
                raise ValueError("legacy annotation is missing audio_path")
            record["source_relpath"] = source_relpath_for_audio(audio_path, input_root)
        yield record


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-jsonl")
    source.add_argument("--empty", action="store_true")
    parser.add_argument("--input-root")
    parser.add_argument("--result-dir", required=True)
    args = parser.parse_args()

    if args.input_jsonl and not args.input_root:
        parser.error("--input-root is required with --input-jsonl")
    records = (
        converted_records(args.input_jsonl, args.input_root)
        if args.input_jsonl
        else iter(())
    )
    annotations_dir = Path(args.result_dir).expanduser().resolve() / "final" / "annotations"
    counts = publish_annotation_records(records, annotations_dir)
    print(
        f"[annotations] total={counts['total']} song={counts['song']} "
        f"instrumental={counts['instrumental']} dir={annotations_dir}"
    )


if __name__ == "__main__":
    main()
