import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

pattern = "{model_name}_{model_format}_{num_repeats}_{test_name}.json"

FORMAT_LABELS = {
    "safetensors": "SafeTensors",
    "sllm": "SLLM",
    "sllm-condense": "SLLM Condense",
}


def get_args():
    parser = argparse.ArgumentParser(description="Plot benchmark results.")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="Model names to show in the plot.",
    )
    parser.add_argument(
        "--test-name",
        type=str,
        required=True,
        choices=["random", "cached"],
        help="Name of the test.",
    )
    parser.add_argument(
        "--num-repeats",
        type=int,
        required=True,
        help="Number of repeats for the benchmark.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Directory to load results from.",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="loading_latency.png",
        help="Output filename for the plot.",
    )
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        choices=list(FORMAT_LABELS.keys()),
        default=list(FORMAT_LABELS.keys()),
        help="Model formats to include in the plot.",
    )
    return parser.parse_args()


def load_results(models, model_format, num_repeats, test_name, results_dir):
    """Load results from files and check for the expected number of repeats."""
    all_results = {}
    for model in models:
        model = model.replace("/", "_")
        filename = pattern.format(
            model_name=model,
            model_format=model_format,
            num_repeats=num_repeats,
            test_name=test_name,
        )
        filename = os.path.join(results_dir, filename)
        with open(filename) as f:
            results = json.load(f)
            if len(results) != num_repeats:
                print(
                    f"Error: Expected {num_repeats} repeats, but found {len(results)} in {filename}."
                )
                exit(1)
        all_results[model] = results
    return all_results


def print_statistics(results_by_format):
    """Compute and print statistics for each model."""
    print("\n" + "=" * 80)
    print("ServerlessLLM Benchmark Statistics")
    print("=" * 80)

    first_format = next(iter(results_by_format))
    for model_name in results_by_format[first_format].keys():
        display_name = model_name.replace("_", "/")
        print(f"\nModel: {display_name}")
        print("-" * 80)
        print(
            f"{'Format':<15} {'Avg (s)':<12} {'Min (s)':<12} {'Max (s)':<12} {'Std Dev':<12}"
        )
        print("-" * 80)
        format_means = {}
        for format_type, format_results in results_by_format.items():
            times = [
                result["loading_time"]
                for result in format_results[model_name]
            ]
            format_means[format_type] = np.mean(times)
            print(
                f"{FORMAT_LABELS[format_type]:<15} "
                f"{np.mean(times):<12.3f} "
                f"{np.min(times):<12.3f} "
                f"{np.max(times):<12.3f} "
                f"{np.std(times):<12.3f}"
            )
        print("-" * 80)
        if "safetensors" in format_means:
            for format_type, mean_time in format_means.items():
                if format_type == "safetensors":
                    continue
                speedup = format_means["safetensors"] / mean_time
                print(
                    f"{FORMAT_LABELS[format_type]} Speedup: "
                    f"{speedup:.2f}x faster than SafeTensors"
                )

    print("=" * 80 + "\n")


def create_dataframe(results_by_format):
    """Convert results list to pandas DataFrame."""
    rows = []
    for format_type, format_results in results_by_format.items():
        for model_name, results in format_results.items():
            model_label = (
                model_name.split("_", 1)[1]
                if "_" in model_name
                else model_name
            )
            for result in results:
                rows.append(
                    {
                        "Model": model_label,
                        "System": FORMAT_LABELS[format_type],
                        "Loading Time": result["loading_time"],
                    }
                )
    return pd.DataFrame(rows)


def plot_results(df, output_filename):
    """Plot loading times as a grouped horizontal bar chart."""
    plt.style.use("default")

    models = list(df["Model"].unique())
    n_models = len(models)
    fig_height = max(4, 2 + n_models * 1.2)
    fig, ax = plt.subplots(figsize=(10, fig_height), facecolor="white")

    summary_df = (
        df.groupby(["Model", "System"], as_index=False)["Loading Time"]
        .mean()
        .sort_values(["Model", "System"])
    )
    sns.barplot(
        data=summary_df,
        x="Loading Time",
        y="Model",
        hue="System",
        ax=ax,
        palette="Paired",
        orient="h",
    )

    ax.set_xlabel(
        "Average Loading Time (s)",
        fontsize=12,
        weight="500",
        fontfamily="sans-serif",
    )
    ax.set_title(
        "Model Loading Performance",
        fontsize=16,
        weight="600",
        pad=20,
        fontfamily="sans-serif",
        loc="center",
    )

    ax.legend(loc="upper right", frameon=True, fontsize=11)

    ax.grid(axis="x", alpha=0.15, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["bottom"].set_color("#D1D5DB")

    ax.autoscale(axis="x", tight=False)
    ax.margins(x=0.05)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    args = get_args()

    models = args.models
    test_name = args.test_name
    num_repeats = args.num_repeats
    results_dir = args.results_dir
    output_filename = args.output_filename
    formats = args.formats

    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Directory {results_dir} does not exist.")

    output_dir = os.path.dirname(output_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results_by_format = {
        format_type: load_results(
            models, format_type, num_repeats, test_name, results_dir
        )
        for format_type in formats
    }

    # Print statistics
    print_statistics(results_by_format)

    df = create_dataframe(results_by_format)
    plot_results(df, output_filename)


if __name__ == "__main__":
    main()
