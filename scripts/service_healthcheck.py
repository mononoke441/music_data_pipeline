#!/usr/bin/env python3
"""Fail fast unless every configured resident inference service is ready."""

from __future__ import annotations

import argparse

from service_client import ServiceClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service", action="append", required=True, metavar="NAME=URL")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    for value in args.service:
        name, separator, url = value.partition("=")
        if not separator or not name or not url:
            parser.error(f"invalid --service {value!r}; expected NAME=URL")
        health = ServiceClient(url, timeout=args.timeout).healthz()
        print(
            f"[health] {name} ready pid={health.get('pid')} "
            f"model={health.get('model_fingerprint')} "
            f"queue={health.get('queue_depth')}/{health.get('queue_capacity')}"
        )


if __name__ == "__main__":
    main()
