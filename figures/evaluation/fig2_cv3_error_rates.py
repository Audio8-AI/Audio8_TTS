"""Figure 2: CV3 multilingual WER and CER comparison."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import (
    ACCENT,
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
DATA_FILE = ROOT / "figures" / "data" / "cv3_error_rates.csv"
OUTPUT_DIR = Path(__file__).resolve().parent
LOGOS_DIR = OUTPUT_DIR / "logos"

MODELS = [
    "Audio8 TTS Preview",
    "Fish S2 Pro",
    "Higgs Audio v2",
    "CosyVoice3-1.5B",
    "VoxCPM2",
]

METRICS = [
    ("zh_cer", "ZH", "CER"),
    ("en_wer", "EN", "WER"),
    ("hard_zh_cer", "HARD ZH", "CER"),
    ("hard_en_wer", "HARD EN", "WER"),
    ("ja_cer", "JA", "CER"),
    ("ko_cer", "KO", "CER"),
    ("de_wer", "DE", "WER"),
    ("es_wer", "ES", "WER"),
    ("fr_wer", "FR", "WER"),
    ("it_wer", "IT", "WER"),
    ("ru_wer", "RU", "WER"),
]


def main() -> None:
    setup_style()
    data = pd.read_csv(DATA_FILE).set_index("model").loc[MODELS]
    parameters = data["parameters_b"].to_dict()

    fig, ax = plt.subplots(figsize=(15.0, 7.2), facecolor=PAPER)
    add_paper_texture(fig, seed=11)
    add_poster_header(
        fig,
        title="CV3",
        descriptor="MULTILINGUAL ERROR-RATE MATRIX  /  LOWER IS BETTER",
        stat="11",
        stat_label="EVALUATION SETS",
    )
    add_model_strip(fig, MODELS, parameters, LOGOS_DIR)
    style_axis(ax)

    group_x = np.arange(len(METRICS))
    width = 0.145
    offsets = (np.arange(len(MODELS)) - (len(MODELS) - 1) / 2) * width
    all_values = data[[metric[0] for metric in METRICS]].to_numpy(dtype=float)

    bar_sets = []
    for model_index, model in enumerate(MODELS):
        values = all_values[model_index]
        bars = ax.bar(
            group_x + offsets[model_index],
            np.nan_to_num(values, nan=0.0),
            width=width * 0.91,
            color=MODEL_COLORS[model],
            edgecolor=INK,
            linewidth=0.42,
            zorder=3,
        )
        bar_sets.append(bars)

    for metric_index in range(len(METRICS)):
        column_values = all_values[:, metric_index]
        winner = int(np.nanargmin(column_values))
        winner_value = column_values[winner]
        winner_x = group_x[metric_index] + offsets[winner]
        ax.scatter(
            winner_x,
            winner_value + 0.24,
            marker="D",
            s=10,
            color=INK,
            zorder=6,
        )

        audio8_value = column_values[0]
        audio8_x = group_x[metric_index] + offsets[0]
        if np.isnan(audio8_value):
            ax.text(
                audio8_x,
                0.32,
                "N/A",
                color=MUTED,
                fontsize=6.5,
                fontweight=700,
                ha="center",
            )
        else:
            ax.text(
                audio8_x,
                audio8_value + 0.30,
                format_value(audio8_value),
                color=ACCENT,
                fontsize=6.5,
                fontweight=900,
                ha="center",
                va="bottom",
            )

    for separator in np.arange(len(METRICS) - 1) + 0.5:
        ax.axvline(separator, color=INK, linewidth=0.35, alpha=0.20, zorder=0)

    ax.set_xlim(-0.62, len(METRICS) - 0.38)
    ax.set_ylim(0, 13.4)
    ax.set_ylabel("ERROR RATE (%)", labelpad=8, fontweight=700)
    ax.set_xticks(group_x)
    ax.set_xticklabels(
        [f"{label}\n{metric}" for _, label, metric in METRICS],
        color=INK,
        fontsize=7.6,
        fontweight=800,
        linespacing=1.35,
    )
    ax.tick_params(axis="x", pad=10)

    fig.subplots_adjust(left=0.058, right=0.965, bottom=0.16, top=0.59)
    fig.text(0.058, 0.048, "SOURCE  /  AUDIO8 TTS README EVALUATION TABLE", color=MUTED, fontsize=7)
    fig.text(0.965, 0.048, "AUDIO8 VALUES SHOWN  /  DIAMOND = LOWEST ERROR", color=INK, fontsize=7, fontweight=700, ha="right")

    output = OUTPUT_DIR / Path(__file__).stem
    fig.savefig(output.with_suffix(".png"), dpi=450, facecolor=PAPER)
    fig.savefig(output.with_suffix(".svg"), facecolor=PAPER)
    plt.close(fig)


if __name__ == "__main__":
    main()
