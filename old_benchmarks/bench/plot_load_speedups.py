#!/usr/bin/env python3
"""Make ServerlessLLM compression/load-speed plots."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sllm_matplotlib")

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_THRESHOLD_CSV = "/home/ben046/sllm/results/plot_threshold_summary.csv"
DEFAULT_PAIR_CSV = "/home/ben046/sllm/results/plot_summary_by_pair.csv"
DEFAULT_TRIALS_CSV = "/home/ben046/sllm/results/plot_trials_coldish.csv"
DEFAULT_OUT_DIR = "/home/ben046/sllm/results/plots"
REFERENCE_THROUGHPUTS = [5, 10, 25, 50, 100, 250, 500]


def model_label(model_name: str) -> str:
    return model_name.replace("facebook/", "")


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=220, bbox_inches="tight")


def set_cr_ticks(ax: plt.Axes, values: pd.Series) -> None:
    ticks = sorted(values.unique())
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{tick:g}x" for tick in ticks])


def plot_break_even(threshold: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    y_values = threshold["min_decomp_gib_s_to_beat_raw_median"]
    y_min = max(0.0, y_values.min() * 0.88)
    y_max = y_values.max() * 1.12
    if y_max <= y_min:
        y_max = y_min + 1.0

    for model_name, group in threshold.groupby("model_name", sort=False):
        group = group.sort_values("compression_ratio")
        ax.plot(
            group["compression_ratio"],
            group["min_decomp_gib_s_to_beat_raw_median"],
            marker="o",
            linewidth=2,
            label=model_label(model_name),
        )

    for y in REFERENCE_THROUGHPUTS:
        if not (y_min <= y <= y_max):
            continue
        ax.axhline(y, color="0.86", linewidth=0.8, zorder=0)
        ax.text(
            1.01,
            y,
            f"{y:g}",
            transform=ax.get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=8,
            color="0.45",
        )

    ax.set_ylim(y_min, y_max)
    set_cr_ticks(ax, threshold["compression_ratio"])
    ax.set_xlabel("Compression ratio")
    ax.set_ylabel("Break-even decode throughput (GiB/s)")
    ax.set_title("GPU Decode Throughput Needed To Beat Raw SLLM Loading")
    ax.grid(True, alpha=0.25)
    ax.legend(title="Model", fontsize=8)
    save_figure(fig, out_dir, "break_even_decode_throughput")
    plt.close(fig)


def subplot_grid(n_items: int) -> tuple[int, int]:
    cols = 2 if n_items > 1 else 1
    rows = math.ceil(n_items / cols)
    return rows, cols


def plot_speedup_by_model(pair: pd.DataFrame, out_dir: Path) -> None:
    models = list(pair["model_name"].drop_duplicates())
    rows, cols = subplot_grid(len(models))
    fig = plt.figure(figsize=(7.2 * cols, 4.2 * rows))
    grid_cols = 4 if cols == 2 else 1
    grid = fig.add_gridspec(rows, grid_cols)
    axes_list: list[plt.Axes] = []
    shared_ax: plt.Axes | None = None

    for model_idx, model_name in enumerate(models):
        row = model_idx // cols
        col = model_idx % cols
        is_centered_last = (
            cols == 2 and len(models) % 2 == 1 and model_idx == len(models) - 1
        )
        if cols == 1:
            subplot_spec = grid[row, 0]
        elif is_centered_last:
            subplot_spec = grid[row, 1:3]
        elif col == 0:
            subplot_spec = grid[row, 0:2]
        else:
            subplot_spec = grid[row, 2:4]
        ax = fig.add_subplot(
            subplot_spec,
            sharex=shared_ax,
            sharey=shared_ax,
        )
        if shared_ax is None:
            shared_ax = ax
        axes_list.append(ax)

        model_data = pair[pair["model_name"] == model_name]
        for throughput, group in model_data.groupby(
            "decomp_throughput_gib_s", sort=True
        ):
            group = group.sort_values("compression_ratio")
            ax.plot(
                group["compression_ratio"],
                group["speedup_median_x"],
                marker="o",
                linewidth=1.8,
                label=f"{throughput:g} GiB/s",
            )

        ax.axhline(1.0, color="black", linestyle=":", linewidth=1.4)
        set_cr_ticks(ax, pair["compression_ratio"])
        ax.set_title(model_label(model_name))
        ax.set_ylabel("Speedup vs raw load")
        ax.grid(True, alpha=0.25)

        if row == rows - 1 or is_centered_last:
            ax.set_xlabel("Compression ratio")

    handles, labels = axes_list[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Decode throughput",
        loc="upper center",
        ncol=min(len(labels), 4),
        bbox_to_anchor=(0.5, 1.02),
        fontsize=8,
    )
    fig.suptitle("Estimated Compression Speedup By Model", y=1.06)
    fig.tight_layout()
    save_figure(fig, out_dir, "speedup_by_model")
    plt.close(fig)


def plot_checkpoint_load_latency(trials: pd.DataFrame, out_dir: Path) -> None:
    summary = (
        trials[["model_name", "checkpoint_gib"]]
        .drop_duplicates()
        .sort_values("checkpoint_gib")
        .reset_index(drop=True)
    )
    models = list(summary["model_name"])
    labels = [model_label(model_name) for model_name in models]
    values = [
        trials.loc[trials["model_name"] == model_name, "load_time_s"].tolist()
        for model_name in models
    ]

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.boxplot(
        values,
        tick_labels=labels,
        showmeans=True,
        meanline=True,
        patch_artist=True,
        boxprops={"facecolor": "#d8ecff", "edgecolor": "#35506b"},
        medianprops={"color": "#12263a", "linewidth": 1.8},
        meanprops={"color": "#d1495b", "linewidth": 1.4, "linestyle": "--"},
        whiskerprops={"color": "#35506b"},
        capprops={"color": "#35506b"},
        flierprops={
            "marker": "o",
            "markerfacecolor": "#ffffff",
            "markeredgecolor": "#35506b",
            "markersize": 4,
        },
    )

    for position, model_values in enumerate(values, start=1):
        for idx, value in enumerate(model_values):
            offset = ((idx % 5) - 2) * 0.025
            ax.plot(
                position + offset,
                value,
                marker="o",
                color="#345995",
                markersize=3,
                alpha=0.65,
                linestyle="none",
            )

    ax.set_xlabel("Model")
    ax.set_ylabel("Load latency (seconds)")
    ax.set_title("Cold-ish ServerlessLLM Checkpoint Load Latency")
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate(rotation=18, ha="right")
    save_figure(fig, out_dir, "checkpoint_load_latency_boxplot")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot SLLM load/compression summary CSVs."
    )
    parser.add_argument("--threshold-csv", default=DEFAULT_THRESHOLD_CSV)
    parser.add_argument("--pair-csv", default=DEFAULT_PAIR_CSV)
    parser.add_argument("--trials-csv", default=DEFAULT_TRIALS_CSV)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    threshold_csv = Path(args.threshold_csv).expanduser()
    pair_csv = Path(args.pair_csv).expanduser()
    trials_csv = Path(args.trials_csv).expanduser()
    out_dir = Path(args.out_dir).expanduser()

    threshold = pd.read_csv(threshold_csv)
    pair = pd.read_csv(pair_csv)
    trials = pd.read_csv(trials_csv)

    plot_break_even(threshold, out_dir)
    plot_speedup_by_model(pair, out_dir)
    plot_checkpoint_load_latency(trials, out_dir)

    print(f"Wrote plots to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
