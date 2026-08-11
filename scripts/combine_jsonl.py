#!/usr/bin/env python3
"""Concatenate JSONL manifests while enforcing unique audio_id values."""

from __future__ import annotations

import argparse

from pipeline_core import iter_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    seen = set()
    records = []
    for path in args.inputs:
        for record in iter_jsonl(path):
            audio_id = str(record.get("audio_id", "")).strip()
            if not audio_id:
                raise ValueError(f"{path}: record is missing audio_id")
            if audio_id in seen:
                raise ValueError(f"duplicate audio_id={audio_id}")
            seen.add(audio_id)
            records.append(record)
    write_jsonl(args.output, records)


if __name__ == "__main__":
    main()
