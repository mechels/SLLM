#!/usr/bin/env python3
"""Download and convert facebook/opt-1.3b to ServerlessLLM Store format."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path


DEFAULT_MODEL = "facebook/opt-1.3b"
DEFAULT_STORAGE = "/home/ben046/sllm/models/sllm"
DEFAULT_RAW_CACHE = "/home/ben046/sllm/models/raw/hf_cache"


def path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def set_hf_cache(raw_cache: Path) -> None:
    raw_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(raw_cache))
    os.environ.setdefault("HF_HUB_CACHE", str(raw_cache / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(raw_cache / "hub"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download OPT-1.3B and save it in ServerlessLLM Store format."
        )
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--storage-path", default=DEFAULT_STORAGE)
    parser.add_argument("--raw-cache", default=DEFAULT_RAW_CACHE)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove an existing converted checkpoint before converting.",
    )
    args = parser.parse_args()

    storage_path = Path(args.storage_path).expanduser().resolve()
    raw_cache = Path(args.raw_cache).expanduser().resolve()
    model_name = args.model_name
    output_path = storage_path / model_name
    marker = output_path / "tensor_index.json"

    if marker.exists() and not args.force:
        print(f"Converted checkpoint already exists: {output_path}")
        print(f"Converted bytes: {path_size_bytes(output_path)}")
        return 0

    if output_path.exists():
        if not args.force:
            print(
                f"Refusing to overwrite existing path without --force: {output_path}",
                file=sys.stderr,
            )
            return 2
        shutil.rmtree(output_path)

    set_hf_cache(raw_cache)

    import torch
    from sllm_store.transformers import save_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Model: {model_name}")
    print(f"Raw Hugging Face cache: {raw_cache}")
    print(f"ServerlessLLM output: {output_path}")
    print("Downloading/loading model in float16 on CPU...")
    start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=str(raw_cache),
        low_cpu_mem_usage=True,
    )
    load_s = time.perf_counter() - start
    print(f"HF load/download time: {load_s:.3f}s")

    print("Saving ServerlessLLM Store checkpoint...")
    start = time.perf_counter()
    save_model(model, str(output_path))
    save_s = time.perf_counter() - start

    print("Saving tokenizer/config sidecar files...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(raw_cache))
    tokenizer.save_pretrained(str(output_path))

    meta = {
        "model_name": model_name,
        "storage_path": str(storage_path),
        "output_path": str(output_path),
        "raw_cache": str(raw_cache),
        "hf_load_download_s": load_s,
        "sllm_save_s": save_s,
        "converted_bytes": path_size_bytes(output_path),
        "raw_cache_bytes": path_size_bytes(raw_cache),
    }
    (output_path / "sllm_baseline_prepare_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )

    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
