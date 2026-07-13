#!/usr/bin/env python3
"""Collect cold-ish ServerlessLLM load data for compression plots."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import socket
import statistics
import time
from pathlib import Path


DEFAULT_MODELS = (
    "facebook/opt-350m,"
    "facebook/opt-1.3b,"
    "facebook/opt-2.7b,"
    "facebook/opt-6.7b,"
    "facebook/opt-13b"
)
DEFAULT_STORAGE = "/home/ben046/sllm/models/sllm"
DEFAULT_CRS = "2,3,4,8"
DEFAULT_DECOMP_GIBS = "5,10,25,50,100,250,500"
DEFAULT_TRIALS_CSV = "/home/ben046/sllm/results/plot_trials_coldish.csv"
DEFAULT_PAIR_CSV = "/home/ben046/sllm/results/plot_summary_by_pair.csv"
DEFAULT_THRESHOLD_CSV = "/home/ben046/sllm/results/plot_threshold_summary.csv"


def parse_models(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_float_list(value: str, label: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError(f"At least one {label} is required")
    if any(item <= 0.0 for item in parsed):
        raise ValueError(f"All {label} values must be positive")
    return parsed


def parse_crs(value: str) -> list[float]:
    crs = parse_float_list(value, "compression ratio")
    if any(item <= 1.0 for item in crs):
        raise ValueError("Compression ratios must be greater than 1")
    return crs


def path_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def wait_for_store(host: str, port: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.25)
    raise RuntimeError(
        f"Timed out waiting for sllm-store at {host}:{port}: {last_error}"
    )


def clear_store_memory(host: str, port: int, timeout_s: float) -> float:
    import grpc
    from sllm_store.proto import storage_pb2, storage_pb2_grpc

    start = time.perf_counter()
    channel = grpc.insecure_channel(f"{host}:{port}")
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout_s)
        stub = storage_pb2_grpc.StorageStub(channel)
        stub.ClearMem(storage_pb2.ClearMemRequest(), timeout=timeout_s)
    finally:
        channel.close()
    return time.perf_counter() - start


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def median(values: list[float]) -> float:
    return statistics.median(values)


def summarize_model(
    model_name: str,
    checkpoint_gib: float,
    trial_rows: list[dict],
    crs: list[float],
    decomp_gibs: list[float],
) -> tuple[list[dict], list[dict]]:
    load_times = [row["load_time_s"] for row in trial_rows]
    effective_rates = [row["raw_effective_load_gib_s"] for row in trial_rows]
    mean_load = mean(load_times)
    median_load = median(load_times)
    mean_rate = mean(effective_rates)
    median_rate = median(effective_rates)

    threshold_rows = []
    pair_rows = []
    for cr in crs:
        mean_budget = mean_load * (1.0 - 1.0 / cr)
        median_budget = median_load * (1.0 - 1.0 / cr)
        threshold_mean = checkpoint_gib / mean_budget
        threshold_median = checkpoint_gib / median_budget
        threshold_base = {
            "model_name": model_name,
            "checkpoint_gib": checkpoint_gib,
            "trials": len(trial_rows),
            "compression_ratio": cr,
            "load_time_mean_s": mean_load,
            "load_time_median_s": median_load,
            "raw_effective_load_gib_s_mean": mean_rate,
            "raw_effective_load_gib_s_median": median_rate,
            "min_decomp_gib_s_to_beat_raw_mean": threshold_mean,
            "min_decomp_gib_s_to_beat_raw_median": threshold_median,
        }
        threshold_rows.append(threshold_base)

        for decomp_gib_s in decomp_gibs:
            mean_compressed_s = (mean_load / cr) + (
                checkpoint_gib / decomp_gib_s
            )
            median_compressed_s = (median_load / cr) + (
                checkpoint_gib / decomp_gib_s
            )
            pair_rows.append(
                {
                    **threshold_base,
                    "decomp_throughput_gib_s": decomp_gib_s,
                    "compressed_runtime_mean_s": mean_compressed_s,
                    "compressed_runtime_median_s": median_compressed_s,
                    "speedup_mean_x": mean_load / mean_compressed_s,
                    "speedup_median_x": median_load / median_compressed_s,
                }
            )
    return threshold_rows, pair_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect plot-ready SLLM cold-ish load data."
    )
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--storage-path", default=DEFAULT_STORAGE)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--compression-ratios", default=DEFAULT_CRS)
    parser.add_argument("--decomp-throughputs-gib-s", default=DEFAULT_DECOMP_GIBS)
    parser.add_argument("--trials-csv", default=DEFAULT_TRIALS_CSV)
    parser.add_argument("--pair-summary-csv", default=DEFAULT_PAIR_CSV)
    parser.add_argument("--threshold-summary-csv", default=DEFAULT_THRESHOLD_CSV)
    parser.add_argument("--store-host", default="127.0.0.1")
    parser.add_argument("--store-port", type=int, default=8073)
    parser.add_argument("--server-timeout-s", type=float, default=30.0)
    parser.add_argument("--cuda-visible-devices", default="0")
    args = parser.parse_args()

    if args.cuda_visible_devices != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    import torch
    from sllm_store.transformers import load_model

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not visible; run in the GPU environment.")

    wait_for_store(args.store_host, args.store_port, args.server_timeout_s)

    storage_path = Path(args.storage_path).expanduser().resolve()
    models = parse_models(args.models)
    crs = parse_crs(args.compression_ratios)
    decomp_gibs = parse_float_list(
        args.decomp_throughputs_gib_s,
        "decompressor throughput",
    )

    all_threshold_rows: list[dict] = []
    all_pair_rows: list[dict] = []
    for model_name in models:
        model_dir = storage_path / model_name
        marker = model_dir / "tensor_index.json"
        if not marker.exists():
            raise FileNotFoundError(
                f"Missing converted checkpoint for {model_name}: {marker}"
            )
        checkpoint_bytes = path_size_bytes(model_dir)
        checkpoint_gib = checkpoint_bytes / (1024**3)

        model_trial_rows: list[dict] = []
        for trial in range(args.trials):
            clear_s = clear_store_memory(
                args.store_host,
                args.store_port,
                args.server_timeout_s,
            )
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.ones(1, device="cuda:0")
            torch.cuda.synchronize()

            start = time.perf_counter()
            model = load_model(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                storage_path=str(storage_path),
                fully_parallel=True,
            )
            torch.cuda.synchronize()
            load_s = time.perf_counter() - start
            row = {
                "model_name": model_name,
                "checkpoint_bytes": checkpoint_bytes,
                "checkpoint_gib": checkpoint_gib,
                "trial": trial,
                "load_time_s": load_s,
                "raw_effective_load_gib_s": checkpoint_gib / load_s,
                "store_clear_time_s": clear_s,
                "gpu_name": torch.cuda.get_device_name(0),
                "cuda_visible_devices": os.environ.get(
                    "CUDA_VISIBLE_DEVICES", ""
                ),
            }
            append_csv(Path(args.trials_csv), row)
            model_trial_rows.append(row)
            print(json.dumps(row, indent=2, sort_keys=True))

            del model
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        threshold_rows, pair_rows = summarize_model(
            model_name=model_name,
            checkpoint_gib=checkpoint_gib,
            trial_rows=model_trial_rows,
            crs=crs,
            decomp_gibs=decomp_gibs,
        )
        all_threshold_rows.extend(threshold_rows)
        all_pair_rows.extend(pair_rows)
        write_csv(Path(args.threshold_summary_csv), all_threshold_rows)
        write_csv(Path(args.pair_summary_csv), all_pair_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
