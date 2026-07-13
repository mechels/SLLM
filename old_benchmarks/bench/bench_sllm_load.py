#!/usr/bin/env python3
"""Benchmark ServerlessLLM Store model loading latency."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import socket
import time
from pathlib import Path


DEFAULT_MODEL = "facebook/opt-1.3b"
DEFAULT_STORAGE = "/home/ben046/sllm/models/sllm"
DEFAULT_JSONL = "/home/ben046/sllm/results/baseline_load_coldish.jsonl"
DEFAULT_CSV = "/home/ben046/sllm/results/baseline_load_coldish.csv"
DEFAULT_COMPRESSION_RATIOS = "2,4,8,16"
DEFAULT_DECOMP_THROUGHPUTS_GIB_S = "5,10,50,100,500"
DEFAULT_CUDA_VISIBLE_DEVICES = "0"


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

    address = f"{host}:{port}"
    start = time.perf_counter()
    channel = grpc.insecure_channel(address)
    try:
        grpc.channel_ready_future(channel).result(timeout=timeout_s)
        stub = storage_pb2_grpc.StorageStub(channel)
        stub.ClearMem(storage_pb2.ClearMemRequest(), timeout=timeout_s)
    finally:
        channel.close()
    return time.perf_counter() - start


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_positive_float_list(value: str, label: str) -> list[float]:
    values: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed = float(item)
        if parsed <= 0.0:
            raise ValueError(f"{label} values must be > 0, got {parsed}")
        values.append(parsed)
    if not values:
        raise ValueError(f"At least one {label} value is required")
    return values


def parse_compression_ratios(value: str) -> list[float]:
    ratios = parse_positive_float_list(value, "compression ratio")
    for ratio in ratios:
        if ratio <= 1.0:
            raise ValueError(f"Compression ratio must be > 1, got {ratio}")
    return ratios


def float_suffix(value: float) -> str:
    text = f"{value:g}".replace(".", "p")
    return text


def cr_suffix(value: float) -> str:
    return f"cr_{float_suffix(value)}x"


def decomp_suffix(value: float) -> str:
    return f"decomp_{float_suffix(value)}gibs"


def add_compression_metrics(
    row: dict,
    checkpoint_bytes: int,
    load_time_s: float,
    compression_ratios: list[float],
    decomp_throughputs_gib_s: list[float],
) -> None:
    raw_gib = checkpoint_bytes / (1024**3)
    for ratio in compression_ratios:
        cr_name = cr_suffix(ratio)
        compressed_transfer_s = load_time_s / ratio
        decomp_budget_s = load_time_s - compressed_transfer_s
        row[f"{cr_name}_min_decomp_gib_s_to_beat_raw"] = (
            raw_gib / decomp_budget_s
        )

        for throughput_gib_s in decomp_throughputs_gib_s:
            throughput_name = decomp_suffix(throughput_gib_s)
            decomp_time_s = raw_gib / throughput_gib_s
            compressed_total_s = compressed_transfer_s + decomp_time_s
            row[f"{cr_name}_{throughput_name}_speedup_x"] = (
                load_time_s / compressed_total_s
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Measure ServerlessLLM Store load time from converted checkpoint "
            "to GPU-ready model."
        )
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--storage-path", default=DEFAULT_STORAGE)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--jsonl", default=DEFAULT_JSONL)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--store-host", default="127.0.0.1")
    parser.add_argument("--store-port", type=int, default=8073)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--no-fully-parallel", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--server-timeout-s", type=float, default=15.0)
    parser.add_argument(
        "--clear-store-between-trials",
        action="store_true",
        help=(
            "Call the Store ClearMem RPC before each measured load. This frees "
            "ServerlessLLM Store's pinned host-memory cache so repeated trials "
            "re-read the checkpoint from disk into host memory."
        ),
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=DEFAULT_CUDA_VISIBLE_DEVICES,
        help=(
            "CUDA_VISIBLE_DEVICES mask for this benchmark process. Use an "
            "empty string to leave the caller's environment unchanged."
        ),
    )
    parser.add_argument(
        "--compression-ratios",
        default=DEFAULT_COMPRESSION_RATIOS,
        help=(
            "Comma-separated CR values used to compute break-even throughput "
            "and speedup estimates."
        ),
    )
    parser.add_argument(
        "--decomp-throughputs-gib-s",
        default=DEFAULT_DECOMP_THROUGHPUTS_GIB_S,
        help=(
            "Comma-separated decompressor throughputs in raw-output GiB/s. "
            "Each value is paired with each CR to estimate speedup."
        ),
    )
    args = parser.parse_args()

    if args.cuda_visible_devices != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    import torch
    from sllm_store.transformers import load_model

    storage_path = Path(args.storage_path).expanduser().resolve()
    model_path = storage_path / args.model_name
    marker = model_path / "tensor_index.json"
    if not marker.exists():
        raise FileNotFoundError(
            f"Missing converted checkpoint marker: {marker}. "
            "Run prepare_opt13b.py first."
        )

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "CUDA is not visible. Re-run where the GPU is available, or pass "
            "--allow-cpu only for a non-baseline smoke test."
        )

    wait_for_store(args.store_host, args.store_port, args.server_timeout_s)

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    compression_ratios = parse_compression_ratios(args.compression_ratios)
    decomp_throughputs_gib_s = parse_positive_float_list(
        args.decomp_throughputs_gib_s,
        "decompressor throughput",
    )

    checkpoint_bytes = path_size_bytes(model_path)
    for trial in range(args.trials):
        store_clear_time_s = 0.0
        if args.clear_store_between_trials:
            store_clear_time_s = clear_store_memory(
                args.store_host,
                args.store_port,
                args.server_timeout_s,
            )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            for device_idx in range(torch.cuda.device_count()):
                torch.ones(1, device=f"cuda:{device_idx}")
                torch.cuda.synchronize(device_idx)

        start = time.perf_counter()
        model = load_model(
            args.model_name,
            device_map=args.device_map,
            torch_dtype=dtype,
            storage_path=str(storage_path),
            fully_parallel=not args.no_fully_parallel,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        load_time_s = time.perf_counter() - start
        raw_effective_bytes_s = checkpoint_bytes / load_time_s
        raw_effective_gib_s = raw_effective_bytes_s / (1024**3)
        raw_effective_gbps = raw_effective_bytes_s * 8 / 1_000_000_000

        row = {
            "model_name": args.model_name,
            "storage_path": str(storage_path),
            "checkpoint_bytes": checkpoint_bytes,
            "checkpoint_gib": checkpoint_bytes / (1024**3),
            "trial": trial,
            "load_time_s": load_time_s,
            "raw_effective_load_bytes_s": raw_effective_bytes_s,
            "raw_effective_load_gib_s": raw_effective_gib_s,
            "raw_effective_load_gbps": raw_effective_gbps,
            "observed_gib_s": raw_effective_gib_s,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "gpu_name": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "",
            "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
            if torch.cuda.is_available()
            else 0,
            "max_cuda_reserved_bytes": torch.cuda.max_memory_reserved()
            if torch.cuda.is_available()
            else 0,
            "fully_parallel": not args.no_fully_parallel,
            "device_map": args.device_map,
            "dtype": args.dtype,
            "clear_store_between_trials": args.clear_store_between_trials,
            "store_clear_time_s": store_clear_time_s,
        }
        add_compression_metrics(
            row,
            checkpoint_bytes=checkpoint_bytes,
            load_time_s=load_time_s,
            compression_ratios=compression_ratios,
            decomp_throughputs_gib_s=decomp_throughputs_gib_s,
        )
        append_jsonl(Path(args.jsonl), row)
        append_csv(Path(args.csv), row)
        print(json.dumps(row, indent=2, sort_keys=True))

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
