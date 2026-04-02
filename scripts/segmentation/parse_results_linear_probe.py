"""Parse linear probe .out files and extract model evaluation results into a CSV."""

import re
import sys
from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).parent


def parse_block(text: str, arch: str, model: str) -> dict | None:
    """Extract results from a single model's output block."""
    checkpoint = re.search(
        r"Restored best checkpoint from epoch (\d+) \(val Dice=([\d.]+)\)", text
    )
    if not checkpoint:
        return None

    best_epoch = int(checkpoint.group(1))
    val_dice = float(checkpoint.group(2))

    scores = {}
    for cls in ("BG", "RV", "MYO", "LV"):
        m = re.search(rf"{cls}:\s*([\d.]+)", text)
        if m:
            scores[cls] = float(m.group(1))

    macro = re.search(r"Macro Dice \(excl\. BG\):\s*([\d.]+)", text)
    macro_dice = float(macro.group(1)) if macro else None

    overall = re.search(r"Overall mean \+/- std:\s*([\d.]+)\s*\+/-\s*([\d.]+)", text)
    mean_macro = float(overall.group(1)) if overall else None
    std_macro = float(overall.group(2)) if overall else None

    if not scores or macro_dice is None:
        return None

    return {
        "arch": arch,
        "model": model,
        "best_epoch": best_epoch,
        "val_dice": val_dice,
        "bg_dice": scores.get("BG"),
        "rv_dice": scores.get("RV"),
        "myo_dice": scores.get("MYO"),
        "lv_dice": scores.get("LV"),
        "macro_dice": macro_dice,
        "mean_macro_dice": mean_macro,
        "std_macro_dice": std_macro,
    }


def parse_file(path: Path) -> list[dict]:
    text = path.read_text()
    results = []

    if ">>>" in text:
        blocks = re.split(r"(?=^>>> )", text, flags=re.MULTILINE)
        for block in blocks:
            m = re.match(r">>> (.+?):\s*(.+)", block)
            if not m:
                continue
            arch = m.group(1).strip()
            model = m.group(2).strip()
            result = parse_block(block, arch, model)
            if result:
                results.append(result)
    else:
        header = re.search(r"Running (.+)", text)
        arch = header.group(1).strip() if header else path.stem
        result = parse_block(text, arch, arch)
        if result:
            results.append(result)

    return results


def main():
    out_files = sorted(OUT_DIR.glob("*linear_probe*.out"))
    if not out_files:
        print("No linear probe .out files found.", file=sys.stderr)
        sys.exit(1)

    all_results = []
    for path in out_files:
        all_results.extend(parse_file(path))

    if not all_results:
        print("No results parsed.", file=sys.stderr)
        sys.exit(1)

    output_path = OUT_DIR / "results_linear_probe.csv"
    df = pd.DataFrame(
        all_results,
        columns=[
            "arch",
            "model",
            "best_epoch",
            "val_dice",
            "bg_dice",
            "rv_dice",
            "myo_dice",
            "lv_dice",
            "macro_dice",
            "mean_macro_dice",
            "std_macro_dice",
        ],
    )
    df.to_csv(output_path, index=False)

    print(f"Wrote {len(all_results)} rows to {output_path}")
    for r in all_results:
        mean = (
            f"  mean={r['mean_macro_dice']:.4f}"
            if r["mean_macro_dice"] is not None
            else ""
        )
        print(f"  {r['arch']:10s}  {r['model']:40s}  macro={r['macro_dice']:.4f}{mean}")


if __name__ == "__main__":
    main()
