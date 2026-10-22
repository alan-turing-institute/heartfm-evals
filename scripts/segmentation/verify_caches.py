"""Verify segmentation feature caches against their split manifests.

Checks each cache root with ``verify_cache_splits`` and, with several roots, that they
all share one train/val/test split (so backbones are compared like-for-like).

Usage:
    uv run --no-sync python scripts/segmentation/verify_caches.py \\
        feature_cache_segmentation/acdc/*
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from heartfm_evals.caching import (
    CacheSplitError,
    verify_cache_splits,
    verify_caches_consistent,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cache_dirs", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        for d in args.cache_dirs:
            verify_cache_splits(d)
            print(f"OK  {d}")
        verify_caches_consistent(args.cache_dirs)
    except CacheSplitError as e:
        print(f"FAIL {e}", file=sys.stderr)
        return 1
    print("All caches share one split.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
