#!/usr/bin/env python
"""Compare classification results between new and old (pre-refactor) result directories.

Walks the *new* results folder, finds matching JSON files in the *old* folder,
and recursively compares all shared keys with a configurable numeric tolerance.

Usage:
    python scripts/compare_classification_results.py
    python scripts/compare_classification_results.py --tolerance 0 --new-dir results/classification --old-dir results/classification_old
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


# ── Recursive comparison ─────────────────────────────────────────────────────


def compare_values(
    old,
    new,
    path: str,
    tolerance: float,
    mismatches: list[dict],
    only_in_old: list[str],
    only_in_new: list[str],
) -> None:
    """Recursively compare *old* and *new* JSON values.

    Parameters
    ----------
    old, new : Any
        The two values to compare.
    path : str
        Dotted JSON path for reporting (e.g. ``config.embed_dim``).
    tolerance : float
        Maximum allowed absolute difference for numeric values.
    mismatches : list[dict]
        Accumulator – appended with details of each mismatch.
    only_in_old, only_in_new : list[str]
        Accumulators for keys present in only one side.
    """
    if isinstance(old, dict) and isinstance(new, dict):
        all_keys = set(old) | set(new)
        for key in sorted(all_keys):
            child_path = f"{path}.{key}" if path else key
            if key not in new:
                only_in_old.append(child_path)
            elif key not in old:
                only_in_new.append(child_path)
            else:
                compare_values(
                    old[key], new[key], child_path, tolerance, mismatches, only_in_old, only_in_new
                )

    elif isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            mismatches.append(
                {
                    "path": path,
                    "reason": f"list length differs: old={len(old)}, new={len(new)}",
                }
            )
            # Compare up to the shorter length
            n = min(len(old), len(new))
        else:
            n = len(old)
        for i in range(n):
            compare_values(
                old[i], new[i], f"{path}[{i}]", tolerance, mismatches, only_in_old, only_in_new
            )

    elif isinstance(old, (int, float)) and isinstance(new, (int, float)):
        if math.isnan(old) and math.isnan(new):
            return
        delta = abs(old - new)
        if delta > tolerance:
            mismatches.append(
                {
                    "path": path,
                    "old": old,
                    "new": new,
                    "delta": delta,
                }
            )

    elif type(old) != type(new):
        # Type mismatch (e.g. string vs number)
        mismatches.append(
            {
                "path": path,
                "reason": f"type differs: old={type(old).__name__}({old!r}), new={type(new).__name__}({new!r})",
            }
        )

    else:
        # Strings, booleans, None, etc. – exact equality
        if old != new:
            mismatches.append(
                {
                    "path": path,
                    "old": old,
                    "new": new,
                    "reason": "values differ",
                }
            )


# ── File-level comparison ────────────────────────────────────────────────────


def compare_files(old_path: Path, new_path: Path, tolerance: float) -> dict:
    """Load two JSON files and return a comparison report."""
    with open(old_path) as f:
        old_data = json.load(f)
    with open(new_path) as f:
        new_data = json.load(f)

    mismatches: list[dict] = []
    only_in_old: list[str] = []
    only_in_new: list[str] = []

    compare_values(old_data, new_data, "", tolerance, mismatches, only_in_old, only_in_new)

    return {
        "mismatches": mismatches,
        "only_in_old": only_in_old,
        "only_in_new": only_in_new,
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare new vs old classification results.")
    parser.add_argument(
        "--new-dir",
        type=Path,
        default=Path("results/classification"),
        help="Directory with new results (default: results/classification)",
    )
    parser.add_argument(
        "--old-dir",
        type=Path,
        default=Path("results/classification_old"),
        help="Directory with old results (default: results/classification_old)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Maximum allowed absolute difference for numeric values (default: 0.01)",
    )
    args = parser.parse_args()

    new_dir: Path = args.new_dir
    old_dir: Path = args.old_dir
    tolerance: float = args.tolerance

    if not new_dir.is_dir():
        print(f"ERROR: new-dir does not exist: {new_dir}")
        raise SystemExit(1)
    if not old_dir.is_dir():
        print(f"ERROR: old-dir does not exist: {old_dir}")
        raise SystemExit(1)

    new_jsons = sorted(new_dir.rglob("*.json"))
    if not new_jsons:
        print(f"No JSON files found in {new_dir}")
        raise SystemExit(0)

    passed = 0
    failed = 0
    skipped = 0

    print(f"Comparing {len(new_jsons)} new result file(s) against {old_dir}")
    print(f"Tolerance: {tolerance}\n")
    print("=" * 72)

    for new_path in new_jsons:
        rel = new_path.relative_to(new_dir)
        old_path = old_dir / rel

        if not old_path.exists():
            print(f"SKIP  {rel}  (no matching old file)")
            skipped += 1
            continue

        report = compare_files(old_path, new_path, tolerance)
        mismatches = report["mismatches"]
        only_old = report["only_in_old"]
        only_new = report["only_in_new"]

        if mismatches:
            failed += 1
            print(f"FAIL  {rel}")
            for m in mismatches:
                if "delta" in m:
                    print(f"  {m['path']}: old={m['old']}, new={m['new']}, delta={m['delta']:.6g}")
                else:
                    print(f"  {m['path']}: {m.get('reason', '')}")
        else:
            passed += 1
            print(f"PASS  {rel}")

        if only_old:
            print(f"  keys only in old: {', '.join(only_old)}")
        if only_new:
            print(f"  keys only in new: {', '.join(only_new)}")

    print("=" * 72)
    total = passed + failed + skipped
    print(f"\nSummary: {total} file(s) checked — {passed} passed, {failed} failed, {skipped} skipped")

    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
