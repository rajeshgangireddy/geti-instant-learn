"""Plot YOLOE benchmark results as grouped bar charts.

Produces two figures:
  1. Text Prompt  – PyTorch GPU vs OpenVINO CPU (FP32/FP16/INT8/INT4)
  2. Visual Prompt – PyTorch GPU vs OpenVINO CPU (FP32/FP16/INT8/INT4)

Usage:
    python examples/plot_yoloe_benchmark.py                           # defaults
    python examples/plot_yoloe_benchmark.py --csv path/to/results.csv
    python examples/plot_yoloe_benchmark.py --out-dir plots/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Aesthetic constants ──────────────────────────────────────────────
COLORS = {
    "pytorch fp32": "#3875B5",
    "openvino fp32": "#E8843A",
    "openvino fp16": "#4CAF50",
    "openvino int8": "#D64541",
    "openvino int4": "#9B59B6",
}

SHORT_LABELS = {
    "pytorch fp32": "PyTorch (GPU)",
    "openvino fp32": "OV FP32 (CPU)",
    "openvino fp16": "OV FP16 (CPU)",
    "openvino int8": "OV INT8 (CPU)",
    "openvino int4": "OV INT4 (CPU)",
}

MODEL_ORDER = [
    "yoloe-26n-seg",
    "yoloe-26s-seg",
    "yoloe-26m-seg",
    "yoloe-26l-seg",
    "yoloe-26x-seg",
]

MODEL_SHORT = {m: m.replace("yoloe-26", "").replace("-seg", "").upper() for m in MODEL_ORDER}
# e.g.  "yoloe-26n-seg" -> "N"

FORMAT_ORDER = ["pytorch fp32", "openvino fp32", "openvino fp16", "openvino int8", "openvino int4"]

BAR_WIDTH = 0.14


# ── Helpers ──────────────────────────────────────────────────────────
def load_results(csv_path: Path) -> pd.DataFrame:
    """Load CSV and create a compound key for backend+format."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df["key"] = df["backend"].str.strip() + " " + df["format"].str.strip()
    df["mode"] = df["mode"].str.strip()
    df["model"] = df["model"].str.strip()
    return df


def plot_mode(df: pd.DataFrame, mode: str, out_path: Path) -> None:
    """Create a grouped bar chart for a single prompt mode."""
    subset = df[df["mode"] == mode].copy()
    if subset.empty:
        print(f"  No data for mode={mode}, skipping.")
        return

    models = [m for m in MODEL_ORDER if m in subset["model"].values]
    keys = [k for k in FORMAT_ORDER if k in subset["key"].values]

    n_models = len(models)
    n_keys = len(keys)
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, key in enumerate(keys):
        vals = []
        for m in models:
            row = subset[(subset["model"] == m) & (subset["key"] == key)]
            vals.append(row["avg_ms"].values[0] if len(row) else 0)
        offset = (i - (n_keys - 1) / 2) * BAR_WIDTH
        bars = ax.bar(
            x + offset,
            vals,
            BAR_WIDTH,
            label=SHORT_LABELS.get(key, key),
            color=COLORS.get(key, "#888888"),
            edgecolor="white",
            linewidth=0.5,
        )
        # Value labels on top of bars
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 2,
                    f"{v:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                )

    title_mode = "Text Prompt" if mode == "text_prompt" else "Visual Prompt"
    ax.set_title(f"YOLOE Inference Latency — {title_mode}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_SHORT[m] for m in models], fontsize=11)
    ax.set_xlabel("Model variant", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Plot YOLOE benchmark results.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("exports/yoloe_benchmark/results.csv"),
        help="Path to benchmark CSV (default: exports/yoloe_benchmark/results.csv)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: same dir as CSV)",
    )
    args = parser.parse_args()

    csv_path: Path = args.csv
    out_dir: Path = args.out_dir or csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(csv_path)

    print("Generating plots …")
    plot_mode(df, "text_prompt", out_dir / "benchmark_text_prompt.png")
    plot_mode(df, "visual_prompt", out_dir / "benchmark_visual_prompt.png")
    print("Done.")


if __name__ == "__main__":
    main()
