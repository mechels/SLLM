#!/usr/bin/env python3
"""Prepare recommended OPT checkpoints in ServerlessLLM Store format."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
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
DEFAULT_HF_CACHE = "/mnt/data/vllm_models/hub"


def parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def model_cache_dir(cache_root: Path, model_name: str) -> Path:
    return cache_root / f"models--{model_name.replace('/', '--')}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert recommended local Hugging Face models to SLLM Store."
    )
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument("--storage-path", default=DEFAULT_STORAGE)
    parser.add_argument("--hf-cache", default=DEFAULT_HF_CACHE)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face download if a model is not already cached.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Remove and reconvert existing SLLM checkpoints.",
    )
    args = parser.parse_args()

    storage_path = Path(args.storage_path).expanduser().resolve()
    hf_cache = Path(args.hf_cache).expanduser().resolve()
    storage_path.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(hf_cache.parent))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache))

    import torch
    from sllm_store.transformers import save_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = []
    for model_name in parse_list(args.models):
        out_dir = storage_path / model_name
        marker = out_dir / "tensor_index.json"
        if marker.exists() and not args.force:
            row = {
                "model_name": model_name,
                "status": "exists",
                "output_path": str(out_dir),
                "converted_bytes": path_size_bytes(out_dir),
            }
            rows.append(row)
            print(json.dumps(row, indent=2))
            continue

        local_cache_dir = model_cache_dir(hf_cache, model_name)
        if not local_cache_dir.exists() and not args.allow_download:
            row = {
                "model_name": model_name,
                "status": "missing_cache",
                "expected_cache_dir": str(local_cache_dir),
            }
            rows.append(row)
            print(json.dumps(row, indent=2))
            continue

        if out_dir.exists():
            shutil.rmtree(out_dir)

        start = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            cache_dir=str(hf_cache),
            local_files_only=not args.allow_download,
            low_cpu_mem_usage=True,
        )
        hf_load_s = time.perf_counter() - start

        start = time.perf_counter()
        save_model(model, str(out_dir))
        sllm_save_s = time.perf_counter() - start

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=str(hf_cache),
            local_files_only=not args.allow_download,
        )
        tokenizer.save_pretrained(str(out_dir))

        row = {
            "model_name": model_name,
            "status": "converted",
            "output_path": str(out_dir),
            "hf_cache": str(hf_cache),
            "hf_load_s": hf_load_s,
            "sllm_save_s": sllm_save_s,
            "converted_bytes": path_size_bytes(out_dir),
        }
        (out_dir / "sllm_prepare_meta.json").write_text(
            json.dumps(row, indent=2) + "\n",
            encoding="utf-8",
        )
        rows.append(row)
        print(json.dumps(row, indent=2))

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = storage_path / "plot_model_prepare_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
