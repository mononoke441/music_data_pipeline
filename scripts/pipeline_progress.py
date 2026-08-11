"""Consistent, low-noise progress bars for production pipeline stages."""

from __future__ import annotations

import os
import sys
from typing import Any, Iterable, Optional, TypeVar

from tqdm import tqdm


T = TypeVar("T")


def progress_enabled() -> bool:
    return os.environ.get("PIPELINE_PROGRESS", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


def pipeline_tqdm(
    iterable: Optional[Iterable[T]] = None,
    *,
    total: Optional[int] = None,
    initial: int = 0,
    desc: str,
    unit: str = "item",
    **kwargs: Any,
) -> tqdm[T]:
    """Build one predictable bar even when stderr is piped through ``tee``.

    Tqdm normally disables itself when the runner redirects stderr.  Production
    runs intentionally use ``tee`` for ``pipeline.log``, so progress is enabled
    explicitly and rate-limited to keep that log compact.
    """

    try:
        min_interval = max(
            0.1, float(os.environ.get("PIPELINE_PROGRESS_MIN_INTERVAL", "2.0"))
        )
    except ValueError:
        min_interval = 2.0
    if iterable is None and total is not None and initial >= total:
        if progress_enabled():
            print(
                f"{desc}: 100%|██████████| {total}/{total} [cached]",
                file=sys.stderr,
                flush=True,
            )
        return tqdm(total=0, disable=True)
    return tqdm(
        iterable,
        total=total,
        initial=max(0, initial),
        desc=desc,
        unit=unit,
        mininterval=min_interval,
        maxinterval=max(10.0, min_interval * 5.0),
        dynamic_ncols=True,
        leave=True,
        disable=not progress_enabled(),
        **kwargs,
    )


def count_jsonl(path: str) -> int:
    with open(path, "rb") as stream:
        return sum(1 for line in stream if line.strip())
