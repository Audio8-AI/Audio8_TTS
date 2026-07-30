"""Poster-inspired visual system for Audio8 evaluation figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Ellipse
from PIL import Image, ImageDraw


PAPER = "#F2F1ED"
INK = "#111111"
MUTED = "#666862"
GRID = "#C9C9C2"
ACCENT = "#F2553D"

MODEL_COLORS = {
    "Audio8 TTS Preview": ACCENT,
    "Fish S2 Pro": "#526A86",
    "Higgs Audio v2": "#4B8978",
    "CosyVoice3-1.5B": "#776DA3",
    "MOSS-TTS": "#C4923E",
    "VoxCPM2": "#527F91",
}

MODEL_SHORT_NAMES = {
    "Audio8 TTS Preview": "Audio8 TTS",
    "Fish S2 Pro": "Fish S2 Pro",
    "Higgs Audio v2": "Higgs v2",
    "CosyVoice3-1.5B": "CosyVoice3",
    "MOSS-TTS": "MOSS-TTS",
    "VoxCPM2": "VoxCPM2",
}

LOGO_FILES = {
    "Audio8 TTS Preview": "audio8.jpg",
    "Fish S2 Pro": "fish_audio.webp",
    "Higgs Audio v2": "higgs_audio.png",
    "CosyVoice3-1.5B": "cosyvoice.webp",
    "MOSS-TTS": "moss_tts.webp",
    "VoxCPM2": "voxcpm.webp",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Helvetica Neue",
                "Arial",
                "Avenir Next",
                "DejaVu Sans",
            ],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": 800,
            "axes.labelsize": 8.5,
            "axes.labelcolor": MUTED,
            "axes.edgecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 7.5,
            "axes.unicode_minus": False,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
            "svg.fonttype": "none",
        }
    )


def format_value(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def add_paper_texture(fig, seed: int = 8) -> None:
    """Keep a warm paper ground without an artificial repeated pattern."""
    del fig
    del seed


def _logo_tile(path: Path, size: int = 112) -> np.ndarray:
    """Normalize logos and remove flat white backgrounds where possible."""
    image = Image.open(path).convert("RGBA")
    pixels = np.asarray(image).copy()
    near_white = np.all(pixels[:, :, :3] > 246, axis=2)
    pixels[near_white, 3] = 0
    image = Image.fromarray(pixels, mode="RGBA")
    image.thumbnail((size - 22, size - 22), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (size, size), (250, 250, 247, 255))
    position = ((size - image.width) // 2, (size - image.height) // 2)
    canvas.alpha_composite(image, position)

    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((0, 0, size - 1, size - 1), radius=20, fill=255)
    canvas.putalpha(mask)
    return np.asarray(canvas)


def add_poster_header(
    fig,
    title: str,
    descriptor: str,
    stat: str,
    stat_label: str,
) -> None:
    """Build the oversized typographic header used by the reference poster."""
    fig.text(
        0.055,
        0.965,
        "AUDIO8 TTS  /  EVALUATION",
        color=INK,
        fontsize=9.5,
        fontweight=700,
        va="top",
    )
    fig.text(
        0.055,
        0.925,
        title,
        color=INK,
        fontsize=34,
        fontweight=900,
        va="top",
    )
    fig.text(0.057, 0.845, descriptor, color=MUTED, fontsize=9, va="center")

    center = (0.845, 0.905)
    for width, height, color, linewidth in [
        (0.245, 0.135, GRID, 0.65),
        (0.205, 0.105, "#92938D", 0.55),
        (0.165, 0.078, ACCENT, 0.85),
    ]:
        fig.add_artist(
            Ellipse(
                center,
                width,
                height,
                transform=fig.transFigure,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
                zorder=1,
            )
        )
    fig.text(
        center[0],
        0.925,
        stat,
        color=INK,
        fontsize=35,
        fontweight=900,
        ha="center",
        va="center",
        zorder=2,
    )
    fig.text(
        center[0],
        0.866,
        stat_label,
        color=INK,
        fontsize=7.5,
        fontweight=800,
        ha="center",
        va="center",
        zorder=2,
    )

    fig.lines.extend(
        [
            plt.Line2D(
                [0.055, 0.31],
                [0.805, 0.805],
                transform=fig.transFigure,
                color=ACCENT,
                linewidth=1.6,
            ),
            plt.Line2D(
                [0.31, 0.945],
                [0.805, 0.805],
                transform=fig.transFigure,
                color=INK,
                linewidth=0.55,
            ),
        ]
    )


def add_model_strip(
    fig,
    models: list[str],
    parameters: dict[str, float],
    logos_dir: Path,
    y_icon: float = 0.742,
    y_name: float = 0.703,
    y_params: float = 0.679,
) -> None:
    """Render a compact logo strip that doubles as the bar-order legend."""
    positions = np.linspace(0.095, 0.905, len(models))
    for x, model in zip(positions, models):
        image = OffsetImage(
            _logo_tile(logos_dir / LOGO_FILES[model]),
            zoom=0.205,
            interpolation="lanczos",
        )
        fig.add_artist(
            AnnotationBbox(
                image,
                (x, y_icon),
                xycoords=fig.transFigure,
                frameon=True,
                pad=0.12,
                bboxprops={
                    "boxstyle": "round,pad=0.12,rounding_size=0.45",
                    "facecolor": "#FAFAF7",
                    "edgecolor": ACCENT if model == "Audio8 TTS Preview" else INK,
                    "linewidth": 1.05 if model == "Audio8 TTS Preview" else 0.65,
                },
            )
        )
        fig.text(
            x,
            y_name,
            MODEL_SHORT_NAMES[model],
            ha="center",
            va="center",
            color=ACCENT if model == "Audio8 TTS Preview" else INK,
            fontsize=7.8,
            fontweight=800 if model == "Audio8 TTS Preview" else 600,
        )
        fig.text(
            x,
            y_params,
            f"{parameters[model]:g}B",
            ha="center",
            va="center",
            color=ACCENT if model == "Audio8 TTS Preview" else MUTED,
            fontsize=7,
            fontweight=800 if model == "Audio8 TTS Preview" else 500,
        )


def style_axis(ax) -> None:
    ax.set_facecolor((1, 1, 1, 0.42))
    ax.grid(axis="y", color=GRID, linewidth=0.65, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(INK)
    ax.spines["bottom"].set_linewidth(0.75)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)
