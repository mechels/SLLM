#!/usr/bin/env python3
"""Create an ACM-style comparison plot for benchmark loading results."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import os
import re
import sys
from pathlib import Path
from statistics import fmean, pstdev

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "Missing plotting dependencies. Run with the project environment, e.g.\n"
        "  ../../.venv/bin/python plot_acm_method_comparison.py\n"
        "or install matplotlib and numpy."
    ) from exc


FORMATS = ("safetensors", "sllm", "sllm-condense")
FORMAT_LABELS = {
    "safetensors": "SafeTensors",
    "sllm": "SLLM",
    "sllm-condense": "SLLM-Condense",
}
FORMAT_COLORS = {
    "safetensors": "#4D4D4D",
    "sllm": "#4477AA",
    "sllm-condense": "#CC6677",
}
FORMAT_HATCHES = {
    "safetensors": "///",
    "sllm": "///",
    "sllm-condense": "///",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot an ACM-style comparison of SafeTensors, SLLM, and "
            "SLLM-Condense loading latency across benchmark result folders."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
        help="Root directory containing per-model result folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/acm_method_comparison"),
        help=(
            "Output path without extension, or with .pdf/.png extension. "
            "Both PDF and PNG are written."
        ),
    )
    parser.add_argument(
        "--benchmark-type",
        default="random",
        help="Benchmark type suffix to include, default: random.",
    )
    parser.add_argument(
        "--num-replicas",
        type=int,
        default=4,
        help="Replica count suffix to include, default: 4.",
    )
    parser.add_argument(
        "--log-y",
        action="store_true",
        help="Use a logarithmic y-axis for loading time.",
    )
    parser.add_argument(
        "--layout",
        choices=["grouped", "small-multiple"],
        default="grouped",
        help=(
            "Plot layout. grouped uses one shared y-axis; small-multiple uses "
            "one linear panel per model."
        ),
    )
    parser.add_argument(
        "--title",
        default="",
        help="Plot title. Pass an empty string to omit.",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=["*_backup"],
        help=(
            "Directory-name glob to skip. Can be passed multiple times. "
            "Default: *_backup."
        ),
    )
    return parser.parse_args()


def model_sort_key(model_name: str) -> tuple[float, str]:
    match = re.search(r"(\d+(?:\.\d+)?)\s*[Bb]\b", model_name)
    if match:
        return (float(match.group(1)), model_name)
    return (float("inf"), model_name)


def clean_model_label(model_name: str) -> str:
    label = model_name.split("/")[-1]
    label = label.replace("Meta-", "")
    label = label.replace("-hf", "")
    return label


def normalize_output_base(output: Path) -> Path:
    if output.suffix.lower() in {".pdf", ".png"}:
        return output.with_suffix("")
    return output


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": fmean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "count": len(values),
    }


def load_result_file(path: Path) -> list[float]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return [float(row["loading_time"]) for row in rows]


def discover_results(
    results_root: Path,
    num_replicas: int,
    benchmark_type: str,
    exclude_dir_patterns: list[str],
) -> dict[str, dict[str, dict[str, float]]]:
    model_results: dict[str, dict[str, dict[str, float]]] = {}

    for result_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        if any(
            fnmatch.fnmatch(result_dir.name, pattern)
            for pattern in exclude_dir_patterns
        ):
            print(f"Skipping excluded directory: {result_dir}", file=sys.stderr)
            continue
        summary_path = result_dir / "summary.json"
        if not summary_path.exists():
            continue
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        model_name = summary.get("model_name")
        if not model_name:
            continue

        per_format = {}
        for format_name in FORMATS:
            filename = (
                f"{model_name}_{format_name}_{num_replicas}_{benchmark_type}.json"
            ).replace("/", "_")
            path = result_dir / filename
            if not path.exists():
                print(f"Skipping missing result: {path}", file=sys.stderr)
                continue
            per_format[format_name] = stats(load_result_file(path))

        if all(format_name in per_format for format_name in FORMATS):
            model_results[model_name] = per_format
        elif per_format:
            print(
                f"Skipping incomplete model result set: {result_dir}",
                file=sys.stderr,
            )

    if not model_results:
        raise FileNotFoundError(
            f"No complete result sets found under {results_root}"
        )
    return dict(sorted(model_results.items(), key=lambda item: model_sort_key(item[0])))


def write_csv(
    output_base: Path, model_results: dict[str, dict[str, dict[str, float]]]
) -> Path:
    csv_path = output_base.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "format",
                "mean_loading_time_s",
                "std_loading_time_s",
                "min_loading_time_s",
                "max_loading_time_s",
                "count",
                "speedup_vs_safetensors",
            ],
        )
        writer.writeheader()
        for model_name, per_format in model_results.items():
            safetensors_mean = per_format["safetensors"]["mean"]
            for format_name in FORMATS:
                row = per_format[format_name]
                writer.writerow(
                    {
                        "model": model_name,
                        "format": format_name,
                        "mean_loading_time_s": f"{row['mean']:.6f}",
                        "std_loading_time_s": f"{row['std']:.6f}",
                        "min_loading_time_s": f"{row['min']:.6f}",
                        "max_loading_time_s": f"{row['max']:.6f}",
                        "count": int(row["count"]),
                        "speedup_vs_safetensors": (
                            f"{safetensors_mean / row['mean']:.6f}"
                        ),
                    }
                )
    return csv_path


def configure_acm_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
            "hatch.linewidth": 0.6,
        }
    )


def plot_results(
    output_base: Path,
    model_results: dict[str, dict[str, dict[str, float]]],
    title: str,
    log_y: bool,
    layout: str,
) -> tuple[Path, Path]:
    configure_acm_style()

    if layout == "small-multiple":
        return plot_small_multiples(output_base, model_results, title)

    model_names = list(model_results.keys())
    labels = [clean_model_label(name) for name in model_names]
    x = np.arange(len(model_names), dtype=float)
    width = 0.24
    offsets = {
        "safetensors": -width,
        "sllm": 0.0,
        "sllm-condense": width,
    }

    fig, ax = plt.subplots(figsize=(7.1, 3.6), constrained_layout=True)

    for format_name in FORMATS:
        means = [
            model_results[model_name][format_name]["mean"]
            for model_name in model_names
        ]
        stds = [
            model_results[model_name][format_name]["std"]
            for model_name in model_names
        ]
        ax.bar(
            x + offsets[format_name],
            means,
            width,
            yerr=stds,
            label=FORMAT_LABELS[format_name],
            color=FORMAT_COLORS[format_name],
            edgecolor="black",
            linewidth=0.55,
            hatch=FORMAT_HATCHES[format_name],
            error_kw={
                "elinewidth": 0.75,
                "ecolor": "black",
                "capsize": 2.5,
                "capthick": 0.75,
            },
        )

    ax.set_ylabel("Mean loading time (s)")
    if title:
        ax.set_title(title, pad=7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.6, color="#B8B8B8")
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if log_y:
        ax.set_yscale("log")
        ax.set_ylabel("Mean loading time (s, log scale)")
    else:
        ax.set_ylim(bottom=0)
        ax.margins(y=0.08)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02 if title else 1.13),
        ncol=3,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.4,
    )

    output_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_base.with_suffix(".pdf")
    png_path = output_base.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def plot_small_multiples(
    output_base: Path,
    model_results: dict[str, dict[str, dict[str, float]]],
    title: str,
) -> tuple[Path, Path]:
    model_names = list(model_results.keys())
    fig, axes = plt.subplots(
        1,
        len(model_names),
        figsize=(7.1, 3.35),
        sharey=False,
        constrained_layout=True,
    )
    if len(model_names) == 1:
        axes = [axes]

    x = np.arange(len(FORMATS), dtype=float)
    labels = [FORMAT_LABELS[format_name] for format_name in FORMATS]

    for ax, model_name in zip(axes, model_names):
        means = [model_results[model_name][format_name]["mean"] for format_name in FORMATS]
        stds = [model_results[model_name][format_name]["std"] for format_name in FORMATS]
        colors = [FORMAT_COLORS[format_name] for format_name in FORMATS]
        hatches = [FORMAT_HATCHES[format_name] for format_name in FORMATS]

        bars = ax.bar(
            x,
            means,
            yerr=stds,
            color=colors,
            edgecolor="black",
            linewidth=0.55,
            error_kw={
                "elinewidth": 0.75,
                "ecolor": "black",
                "capsize": 2.5,
                "capthick": 0.75,
            },
        )
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)

        ax.set_title(clean_model_label(model_name), pad=5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=32, ha="right")
        ax.set_ylim(0, max(mean + std for mean, std in zip(means, stds)) * 1.18)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle=":", linewidth=0.6, color="#B8B8B8")
        ax.xaxis.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Mean loading time (s)")
    if title:
        fig.suptitle(title, y=1.06, fontsize=10)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_base.with_suffix(".pdf")
    png_path = output_base.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pdf_path, png_path


def print_summary(model_results: dict[str, dict[str, dict[str, float]]]) -> None:
    print("Loaded complete result sets:")
    for model_name, per_format in model_results.items():
        parts = []
        safe_mean = per_format["safetensors"]["mean"]
        for format_name in FORMATS:
            mean = per_format[format_name]["mean"]
            parts.append(
                f"{FORMAT_LABELS[format_name]}={mean:.2f}s"
                f" ({safe_mean / mean:.2f}x)"
            )
        print(f"  {model_name}: " + ", ".join(parts))


def main() -> int:
    args = parse_args()
    results_root = args.results_root.resolve()
    output_base = normalize_output_base(args.output).resolve()

    model_results = discover_results(
        results_root,
        args.num_replicas,
        args.benchmark_type,
        args.exclude_dir,
    )
    print_summary(model_results)
    csv_path = write_csv(output_base, model_results)
    pdf_path, png_path = plot_results(
        output_base, model_results, args.title, args.log_y, args.layout
    )
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote PDF: {pdf_path}")
    print(f"Wrote PNG: {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
