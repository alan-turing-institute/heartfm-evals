"""Parse *unetr*.out files and extract UNetR evaluation results into a CSV."""

import re
import sys
from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).parent


def _parse_int(s: str) -> int:
    return int(s.replace(",", ""))


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

    # UNetR-specific metadata (best-effort; None if absent)
    embed_m = re.search(r"embed_dim[=:](\d+)", text)
    embed_dim = int(embed_m.group(1)) if embed_m else None

    layers_m = re.search(r"Selected layers:\s*\(([^)]+)\)", text)
    selected_layers = layers_m.group(1).strip() if layers_m else None

    # Frozen / backbone params — several log formats:
    #   "Frozen parameters: 93,735,728"
    #   "Loaded dinov3_vits16 with 21,601,152 parameters (frozen)"
    #   "Loaded CineMA pretrained backbone (125,670,400 params, frozen)"
    frozen_m = re.search(r"Frozen parameters:\s*([\d,]+)", text)
    if not frozen_m:
        frozen_m = re.search(r"with ([\d,]+) parameters \(frozen\)", text)
    if not frozen_m:
        frozen_m = re.search(r"\(([\d,]+) params, frozen\)", text)
    frozen_params = _parse_int(frozen_m.group(1)) if frozen_m else None

    trainable_m = re.search(r"Trainable params:\s*([\d,]+)", text)
    trainable_params = _parse_int(trainable_m.group(1)) if trainable_m else None

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
        "embed_dim": embed_dim,
        "selected_layers": selected_layers,
        "frozen_params": frozen_params,
        "trainable_params": trainable_params,
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
    out_files = sorted(OUT_DIR.glob("*unetr*.out"))
    if not out_files:
        print("No *unetr*.out files found.", file=sys.stderr)
        sys.exit(1)

    all_results = []
    for path in out_files:
        all_results.extend(parse_file(path))

    if not all_results:
        print("No results parsed.", file=sys.stderr)
        sys.exit(1)

    output_path = OUT_DIR / "unetr_results.csv"
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
            "embed_dim",
            "selected_layers",
            "frozen_params",
            "trainable_params",
        ],
    )
    df.to_csv(output_path, index=False)

    print(f"Wrote {len(all_results)} rows to {output_path}")
    for r in all_results:
        print(f"  {r['arch']:15s}  {r['model']:40s}  macro={r['macro_dice']:.4f}")


if __name__ == "__main__":
    main()
