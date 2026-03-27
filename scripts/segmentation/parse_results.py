"""Parse .out files and extract model evaluation results into a CSV."""

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
    }


def parse_file(path: Path) -> list[dict]:
    text = path.read_text()
    results = []

    # Files with multiple models use ">>> ARCH: model" headers
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
        # Single-model files: derive arch from the "Running X" header
        header = re.search(r"Running (.+)", text)
        arch = header.group(1).strip() if header else path.stem
        # Try to get a more specific model name from source lines
        source_m = re.search(r"(?:SAM \d source|CineMA source)[^>]*?>\s*(\S+)", text)
        model = source_m.group(1) if source_m else arch
        result = parse_block(text, arch, model)
        if result:
            results.append(result)

    return results


def main():
    out_files = sorted(p for p in OUT_DIR.glob("*.out") if "unetr" not in p.name)
    if not out_files:
        print("No .out files found.", file=sys.stderr)
        sys.exit(1)

    all_results = []
    for path in out_files:
        all_results.extend(parse_file(path))

    if not all_results:
        print("No results parsed.", file=sys.stderr)
        sys.exit(1)

    output_path = OUT_DIR / "results.csv"
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
        ],
    )
    df.to_csv(output_path, index=False)

    print(f"Wrote {len(all_results)} rows to {output_path}")
    for r in all_results:
        print(f"  {r['arch']:10s}  {r['model']:40s}  macro={r['macro_dice']:.4f}")


if __name__ == "__main__":
    main()
