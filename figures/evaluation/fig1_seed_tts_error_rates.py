"""Figure 1: Seed-TTS WER and CER comparison."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import (
    ACCENT,
    GRID,
    INK,
    MODEL_COLORS,
    MUTED,
    PAPER,
    add_model_strip,
    add_paper_texture,
    add_poster_header,
    format_value,
    setup_style,
    style_axis,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "figures" / "data" / "seed_tts_error_rates.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
LOGOS_DIR = OUTPUT_DIR / "logos"

MODELS = [
    "Audio8 TTS Preview",
    "Fish S2 Pro",
    "Higgs Audio v2",
    "CosyVoice3-1.5B",
    "MOSS-TTS",
    "VoxCPM2",
]

METRICS = [
    ("en_wer", "ENGLISH", "WER"),
    ("zh_cer", "CHINESE", "CER"),
]


def main() -> None:
    setup_style()
    data = pd.read_csv(DATA_FILE).set_index("model").loc[MODELS]
    parameters = data["parameters_b"].to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.6), facecolor=PAPER)
    add_paper_texture(fig)
    add_poster_header(
        fig,
        title="SEED-TTS",
        descriptor="TWO CORE ERROR-RATE TRACKS  /  LOWER IS BETTER",
        stat="0.6B",
        stat_label="SMALLEST MODEL",
    )
    add_model_strip(fig, MODELS, parameters, LOGOS_DIR)

    x = np.arange(len(MODELS))
    colors = [MODEL_COLORS[model] for model in MODELS]

    for panel_index, (ax, (column, language, metric)) in enumerate(zip(axes, METRICS)):
        style_axis(ax)
        values = data[column].to_numpy(dtype=float)
        display_values = np.nan_to_num(values, nan=0.0)
        bars = ax.bar(
            x,
            display_values,
            width=0.62,
            color=colors,
            edgecolor=INK,
            linewidth=0.55,
            zorder=3,
        )
        winner = int(np.nanargmin(values))
        maximum = float(np.nanmax(values))
        ax.scatter(
            winner,
            values[winner] + maximum * 0.095,
            marker="D",
            s=18,
            color=INK,
            zorder=6,
        )

        ax.set_ylim(0, maximum * 1.21)
        ax.set_xlim(-0.58, len(MODELS) - 0.42)
        ax.set_xticks([])
        ax.set_title(
            f"{language}  /  {metric}",
            color=INK,
            pad=10,
            loc="left",
            fontsize=10,
            fontweight=900,
        )
        if panel_index == 0:
            ax.set_ylabel("ERROR RATE (%)", labelpad=7, fontweight=700)

        for index, (bar, value) in enumerate(zip(bars, values)):
            if np.isnan(value):
                bar.set_facecolor("none")
                bar.set_edgecolor(GRID)
                bar.set_hatch("//")
                ax.text(
                    index,
                    maximum * 0.035,
                    "N/A",
                    ha="center",
                    color=MUTED,
                    fontsize=7.5,
                    fontweight=700,
                )
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + maximum * 0.026,
                format_value(value),
                ha="center",
                va="bottom",
                color=ACCENT if index == 0 else INK,
                fontsize=7.8,
                fontweight=900 if index == winner else 600,
            )

    fig.subplots_adjust(left=0.065, right=0.955, bottom=0.11, top=0.59, wspace=0.20)
    fig.text(0.058, 0.035, "SOURCE  /  AUDIO8 TTS README EVALUATION TABLE", color=MUTED, fontsize=7)
    fig.text(0.955, 0.035, "DIAMOND = LOWEST ERROR", color=INK, fontsize=7, fontweight=700, ha="right")

    output = OUTPUT_DIR / Path(__file__).stem
    fig.savefig(output.with_suffix(".png"), dpi=450, facecolor=PAPER)
    fig.savefig(output.with_suffix(".svg"), facecolor=PAPER)
    plt.close(fig)


if __name__ == "__main__":
    main()
