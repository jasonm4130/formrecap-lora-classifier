"""Generate SVG charts for the LoRA classifier blog post."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path.home() / "Work/Git/jasonmatthew.dev/apps/web/public/images/blog"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BG = "#111827"
TEXT = "#e5e7eb"
SPINE = "#374151"
INDIGO = "#6366f1"
AMBER = "#f59e0b"
GREY = "#6b7280"
GREEN = "#22c55e"
RED = "#ef4444"


def apply_dark_theme(fig, ax):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=10)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE)
    ax.title.set_color(TEXT)


# ─── Chart 1: Baselines ──────────────────────────────────────────────────────

def chart_baselines():
    labels = [
        "Majority class",
        "Claude Haiku 4.5",
        "5-shot Llama 3.2 3B",
        "Zero-shot Gemma 2B",
        "Zero-shot Mistral 7B",
        "Zero-shot Llama 3.2 3B",
        "Gemma 2B LoRA (ours)",
    ]
    values = [0.063, 0.063, 0.065, 0.063, 0.095, 0.108, 0.916]
    colours = [GREY] * 6 + [INDIGO]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    apply_dark_theme(fig, ax)

    bars = ax.barh(labels, values, color=colours, height=0.6)

    ax.set_xlabel("Macro-F1", fontsize=11, color=TEXT)
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.grid(axis="x", color=SPINE, linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(
            val + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left",
            color=TEXT,
            fontsize=9,
        )

    ax.tick_params(axis="y", labelsize=10)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "lora-baselines.svg", format="svg", bbox_inches="tight",
                facecolor=BG)
    plt.close(fig)
    print("Saved lora-baselines.svg")


# ─── Chart 2: F1 vs Params ───────────────────────────────────────────────────

def chart_f1_vs_params():
    full = {
        "Llama 1B":    (1.24e9, 0.196),
        "Gemma 2B":    (2.51e9, 0.916),
        "Llama 3B":    (3.21e9, 0.856),
        "Mistral 7B":  (7.24e9, 0.961),
    }
    cf = {
        "Gemma 2B CF":   (2.51e9, 0.249),
        "Mistral 7B CF": (7.24e9, 0.760),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    apply_dark_theme(fig, ax)

    # Full adapter series
    full_x = [v[0] for v in full.values()]
    full_y = [v[1] for v in full.values()]
    ax.plot(full_x, full_y, color=INDIGO, linewidth=1.2, zorder=2)
    ax.scatter(full_x, full_y, color=INDIGO, marker="o", s=70, zorder=3,
               label="Full adapter (r=16, all modules)")

    for name, (x, y) in full.items():
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=9, color=TEXT)

    # CF-constrained series
    cf_x = [v[0] for v in cf.values()]
    cf_y = [v[1] for v in cf.values()]
    ax.plot(cf_x, cf_y, color=AMBER, linewidth=1.2, zorder=2)
    ax.scatter(cf_x, cf_y, color=AMBER, marker="s", s=70, zorder=3,
               label="CF-constrained (r=8, q/v only)")

    for name, (x, y) in cf.items():
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=(6, -12), fontsize=9, color=TEXT)

    ax.set_xscale("log")
    ax.set_xlabel("Parameter count (log scale)", fontsize=11, color=TEXT)
    ax.set_ylabel("Macro-F1", fontsize=11, color=TEXT)
    ax.set_ylim(0, 1.05)
    ax.grid(color=SPINE, linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    legend = ax.legend(fontsize=9, facecolor="#1f2937", edgecolor=SPINE,
                       labelcolor=TEXT)

    # Format x-axis ticks as billions
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e9:.1f}B")
    )
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "lora-f1-vs-params.svg", format="svg",
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("Saved lora-f1-vs-params.svg")


# ─── Chart 3: Calibration ────────────────────────────────────────────────────

def chart_calibration():
    models = ["Llama 3B", "Gemma 2B", "Mistral 7B CF"]
    verbalized = [0.164, 0.145, 0.245]
    logprob    = [0.098, 0.103, 0.071]
    calibrated = [0.094, 0.056, 0.075]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4.5))
    apply_dark_theme(fig, ax)

    b1 = ax.bar(x - width, verbalized, width, color=RED,   label="Verbalized", zorder=2)
    b2 = ax.bar(x,          logprob,   width, color=AMBER, label="Logprob",    zorder=2)
    b3 = ax.bar(x + width,  calibrated, width, color=GREEN, label="Calibrated", zorder=2)

    ax.set_ylabel("ECE (lower is better)", fontsize=11, color=TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, color=TEXT)
    ax.set_ylim(0, 0.30)
    ax.grid(axis="y", color=SPINE, linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    legend = ax.legend(fontsize=9, facecolor="#1f2937", edgecolor=SPINE,
                       labelcolor=TEXT)

    # Value labels on top of bars
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.004,
                    f"{h:.3f}", ha="center", va="bottom", color=TEXT, fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "lora-calibration.svg", format="svg",
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("Saved lora-calibration.svg")


# ─── Chart 4: Per-class F1 ───────────────────────────────────────────────────

def chart_per_class_f1():
    classes = [
        "validation_error",
        "distraction",
        "comparison_shopping",
        "accidental_exit",
        "bot",
        "committed_leave",
    ]
    f1_scores = [0.957, 1.000, 0.900, 1.000, 0.889, 0.750]

    def bar_colour(f1):
        if f1 > 0.9:
            return GREEN
        elif f1 >= 0.7:
            return AMBER
        return RED

    colours = [bar_colour(f) for f in f1_scores]

    fig, ax = plt.subplots(figsize=(8, 4))
    apply_dark_theme(fig, ax)

    bars = ax.barh(classes, f1_scores, color=colours, height=0.6)

    ax.set_xlabel("F1 score", fontsize=11, color=TEXT)
    ax.set_xlim(0, 1.08)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
    ax.grid(axis="x", color=SPINE, linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Monospace y-tick labels
    for label in ax.get_yticklabels():
        label.set_fontfamily("monospace")
        label.set_fontsize(10)

    # Value labels after bars
    for bar, val in zip(bars, f1_scores):
        ax.text(
            val + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            ha="left",
            color=TEXT,
            fontsize=9,
        )

    # Legend patches
    patches = [
        mpatches.Patch(color=GREEN, label="F1 > 0.9"),
        mpatches.Patch(color=AMBER, label="F1 0.7–0.9"),
        mpatches.Patch(color=RED,   label="F1 < 0.7"),
    ]
    ax.legend(handles=patches, fontsize=9, facecolor="#1f2937",
              edgecolor=SPINE, labelcolor=TEXT, loc="lower right")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "lora-per-class-f1.svg", format="svg",
                bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print("Saved lora-per-class-f1.svg")


if __name__ == "__main__":
    plt.style.use("dark_background")
    chart_baselines()
    chart_f1_vs_params()
    chart_calibration()
    chart_per_class_f1()
    print("All charts written to:", OUTPUT_DIR)
