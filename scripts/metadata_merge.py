#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 4b — 按 audio_path 合并多个 JSONL 文件中的记录。

多文件中同一 audio_path 的字段会被合并，缺失字段自动补齐，
冲突时保留先出现的值。
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple

from pipeline_progress import count_jsonl, pipeline_tqdm


def iter_jsonl_files(inputs: List[str], recursive: bool = True) -> Iterable[Path]:
    for p in inputs:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"Input not found: {path}")
        if path.is_file():
            if path.suffix.lower() == ".jsonl":
                yield path
            else:
                raise ValueError(f"Not a .jsonl file: {path}")
        else:
            glob_fn = path.rglob if recursive else path.glob
            for f in glob_fn("*.jsonl"):
                if f.is_file():
                    yield f


def safe_json_loads(line: str, src: str, lineno: int) -> Dict[str, Any]:
    try:
        obj = json.loads(line)
    except Exception as e:
        raise ValueError(f"JSON parse error in {src}:{lineno}: {e}")
    if not isinstance(obj, dict):
        raise ValueError(f"JSONL line must be dict in {src}:{lineno}")
    return obj


def is_missing(v: Any) -> bool:
    return v is None


def merge_record(dst: Dict[str, Any], src: Dict[str, Any],
                 conflict_counter: Dict[Tuple[str, str], int],
                 primary_key: str):
    for k, v in src.items():
        if k == primary_key:
            continue
        if k not in dst:
            dst[k] = v
        else:
            if is_missing(dst[k]) and not is_missing(v):
                dst[k] = v
            elif dst[k] != v and not is_missing(v):
                conflict_counter[(k, str(dst[k])[:200])] = \
                    conflict_counter.get((k, str(dst[k])[:200]), 0) + 1


def main():
    ap = argparse.ArgumentParser(
        description="按 audio_path 合并多个 JSONL 文件的记录")
    ap.add_argument("--inputs", "-i", nargs="+", required=True,
                    help="输入路径（.jsonl 文件或目录）")
    ap.add_argument("--output", "-o", required=True, help="输出 jsonl 路径")
    ap.add_argument("--recursive", action="store_true", default=True)
    ap.add_argument("--no-recursive", dest="recursive", action="store_false")
    ap.add_argument("--audio-key", default="audio_path",
                    help="主键字段名（默认 audio_path）")
    ap.add_argument("--encoding", default="utf-8")
    args = ap.parse_args()

    files = sorted(set(iter_jsonl_files(args.inputs, recursive=args.recursive)))
    if not files:
        raise RuntimeError("No .jsonl files found from inputs")

    merged: Dict[str, Dict[str, Any]] = {}
    conflict_counter: Dict[Tuple[str, str], int] = {}
    total_lines = skipped_no_key = 0
    progress = pipeline_tqdm(
        total=sum(count_jsonl(str(path)) for path in files),
        desc="4/7 section context merge",
        unit="row",
    )

    for f in files:
        with f.open("r", encoding=args.encoding) as rf:
            for idx, line in enumerate(rf, start=1):
                line = line.strip()
                if not line:
                    continue
                total_lines += 1
                progress.update(1)
                obj = safe_json_loads(line, str(f), idx)

                key = obj.get(args.audio_key)
                if key is None:
                    skipped_no_key += 1
                    continue
                if not isinstance(key, str):
                    key = str(key)

                if key not in merged:
                    merged[key] = {args.audio_key: key}
                merge_record(merged[key], obj, conflict_counter, args.audio_key)
    progress.close()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding=args.encoding) as wf:
        for key in sorted(merged.keys()):
            wf.write(json.dumps(merged[key], ensure_ascii=False) + "\n")

    print(
        f"[context] rows={total_lines} tracks={len(merged)} "
        f"skipped={skipped_no_key} conflicts={sum(conflict_counter.values())}"
    )


if __name__ == "__main__":
    main()
