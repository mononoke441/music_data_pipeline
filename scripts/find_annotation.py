#!/usr/bin/env python3
"""Find one final annotation by the original input path, without hashing audio."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from annotation_storage import annotation_path, source_relpath_for_audio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--path-only", action="store_true")
    args = parser.parse_args()

    source_relpath = source_relpath_for_audio(args.audio, args.input_root)
    path = annotation_path(
        Path(args.result_dir).expanduser().resolve() / "final" / "annotations",
        source_relpath,
    )
    if not path.is_file():
        raise SystemExit(f"annotation not found: {path}")
    if args.path_only:
        print(path)
        return
    sys.stdout.write(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
