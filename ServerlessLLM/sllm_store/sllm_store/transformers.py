# ---------------------------------------------------------------------------- #
#  ServerlessLLM                                                               #
#  Copyright (c) ServerlessLLM Team 2024                                       #
#                                                                              #
#  Licensed under the Apache License, Version 2.0 (the "License");             #
#  you may not use this file except in compliance with the License.            #
#                                                                              #
#  You may obtain a copy of the License at                                     #
#                                                                              #
#                  http://www.apache.org/licenses/LICENSE-2.0                  #
#                                                                              #
#  Unless required by applicable law or agreed to in writing, software         #
#  distributed under the License is distributed on an "AS IS" BASIS,           #
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    #
#  See the License for the specific language governing permissions and         #
#  limitations under the License.                                              #
# ---------------------------------------------------------------------------- #
import concurrent.futures
import json
import os
import shutil
import sys
import time
import uuid
from typing import Optional, Union, Dict, Any

import torch
from accelerate import dispatch_model, init_empty_weights

# from accelerate.hooks import add_hook_to_module
from accelerate.utils import set_module_tensor_to_device
from sllm_store._C import (
    allocate_cuda_memory,
    free_cuda_memory,
    get_cuda_memory_addresses,
    get_cuda_memory_handles,
    get_device_uuid_map,
    restore_tensors,
)
from sllm_store.client import SllmStoreClient
from sllm_store.device_map_utils import (
    DeviceMapType,
    _compute_device_placement_from_map,
    _compute_device_placement_from_map_fast,
    _expand_tensor_name,
    _transform_device_map_to_dict,
)
from sllm_store.logger import init_logger
from sllm_store.torch import load_dict_non_blocking, save_dict
from sllm_store.utils import (
    calculate_device_memory,
    calculate_tensor_device_offsets,
    get_no_split_modules,
    get_tied_no_split_modules,
    send_module_buffers_to_device,
    quantize,
)
from torch import nn
from transformers import AutoConfig
import importlib
from peft import (
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    LoraConfig,
)
from peft.utils import set_peft_model_state_dict
from transformers.utils.quantization_config import (
    QuantizationConfigMixin,
)

logger = init_logger(__name__)

# ############# SLLM-CONDENSE #########
CONDENSE_INDEX_FILENAME = "tensor_index.condense.json"
CONDENSE_META_FILENAME = "condense_meta.json"
DEFAULT_FPTC_DIR = "/home/ben046/projects/TensorProcessing/generate_compressed/comp"
DEFAULT_FPTC_BIND_PATH = "/home/ben046/projects/TensorCodec/FPTC-2/bind"
TENSOR_DATA_PREFIX = "tensor.data_"
TENSOR_DATA_PARTITION_SIZE = 10 * 1024**3


def _get_uuid():
    return str(uuid.uuid4())


def save_model(model: nn.Module, model_path: str):
    """
    Args:
        model: nn.Module
            a model to be saved
        storage_path: str
            a local path to save the converted model
    """
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    model = model.cpu()
    model_state_dict = model.state_dict()

    save_dict(model_state_dict, model_path)

    # This section of code was adopted from the Hugging Face Transformers project under Apache-2.0 License. # noqa: E501
    # Source: https://github.com/huggingface/transformers/blob/241c04d36867259cdf11dbb4e9d9a60f9cb65ebc/src/transformers/modeling_utils.py#L2812-L2856
    # Modifications made: Removed the support for '_hf_peft_config_loaded'
    #
    # Save the config
    model.config.save_pretrained(model_path)
    if model.can_generate():
        model.generation_config.save_pretrained(model_path)

    # save module index
    no_split_modules = get_no_split_modules(model, model._no_split_modules)
    with open(os.path.join(model_path, "no_split_modules.json"), "w") as f:
        json.dump(no_split_modules, f)

    # save tied parameters
    tied_no_split_modules = get_tied_no_split_modules(model, no_split_modules)
    with open(os.path.join(model_path, "tied_no_split_modules.json"), "w") as f:
        json.dump(tied_no_split_modules, f)










################################ SLLM-CONDENSE ################################
def save_model_condense(
    model: nn.Module,
    model_path: str,
    fptc_dir: Optional[str] = None,
):
    save_model(model, model_path)
    rewrite_model_with_fptc_packages(model_path, fptc_dir)


def rewrite_model_with_fptc_packages(
    model_path: str,
    fptc_dir: Optional[str] = None,
):
    if fptc_dir is None:
        fptc_dir = os.getenv("SLLM_CONDENSE_FPTC_DIR", DEFAULT_FPTC_DIR)

    tensor_index_path = os.path.join(model_path, "tensor_index.json")
    with open(tensor_index_path, "r") as f:
        tensor_index = json.load(f)

    temp_prefix = "tensor.data.condense_tmp_"
    compressed_index = {}
    partition_id = 0
    partition_size = 0
    logical_offset = 0
    current_file = None

    def close_current_file():
        nonlocal current_file
        if current_file is not None:
            current_file.close()
            current_file = None

    def open_next_partition():
        nonlocal current_file, partition_id, partition_size
        close_current_file()
        current_path = os.path.join(model_path, f"{temp_prefix}{partition_id}")
        current_file = open(current_path, "wb")
        partition_id += 1
        partition_size = 0

    try:
        for name, (raw_offset, raw_size, shape, stride, dtype) in tensor_index.items():
            fptc_path = _fptc_path_for_tensor(fptc_dir, name)
            compressed_size = os.path.getsize(fptc_path)
            if compressed_size == 0:
                raise ValueError(f"FPTC package is empty: {fptc_path}")

            if current_file is None or (
                partition_size > 0
                and partition_size + compressed_size > TENSOR_DATA_PARTITION_SIZE
            ):
                open_next_partition()

            compressed_offset = logical_offset
            with open(fptc_path, "rb") as src:
                shutil.copyfileobj(src, current_file, length=16 * 1024 * 1024)

            padding = (8 - (compressed_size % 8)) % 8
            if padding:
                current_file.write(b"\0" * padding)

            written = compressed_size + padding
            partition_size += written
            logical_offset += written
            compressed_index[name] = {
                "compressed_offset": compressed_offset,
                "compressed_size": compressed_size,
                "uncompressed_offset": raw_offset,
                "uncompressed_size": raw_size,
                "shape": shape,
                "stride": stride,
                "dtype": "torch.bfloat16",
                "source_dtype": dtype,
                "fptc_path": os.path.relpath(fptc_path, fptc_dir),
            }
    finally:
        close_current_file()

    _replace_tensor_data_files(model_path, temp_prefix)

    with open(os.path.join(model_path, CONDENSE_INDEX_FILENAME), "w") as f:
        json.dump(compressed_index, f)
    with open(os.path.join(model_path, CONDENSE_META_FILENAME), "w") as f:
        json.dump(
            {
                "format": "sllm-condense",
                "codec": "fptc-2",
                "schema_version": 1,
                "fptc_dir": fptc_dir,
            },
            f,
            indent=2,
        )


def _replace_tensor_data_files(model_path: str, temp_prefix: str):
    for filename in os.listdir(model_path):
        if filename.startswith(TENSOR_DATA_PREFIX):
            os.remove(os.path.join(model_path, filename))

    for filename in os.listdir(model_path):
        if not filename.startswith(temp_prefix):
            continue
        partition_id = filename.removeprefix(temp_prefix)
        os.replace(
            os.path.join(model_path, filename),
            os.path.join(model_path, f"{TENSOR_DATA_PREFIX}{partition_id}"),
        )


def _fptc_path_for_tensor(fptc_dir: str, tensor_name: str) -> str:
    parts = tensor_name.split(".")
    if len(parts) >= 4 and parts[0] == "model" and parts[1] == "layers":
        layer = parts[2]
        leaf_name = ".".join(parts[3:])
        path = os.path.join(fptc_dir, f"layer{layer}", f"{leaf_name}.fptc")
    elif tensor_name == "model.embed_tokens.weight":
        path = os.path.join(fptc_dir, "layer_misc", "embed_tokens.weight.fptc")
    elif tensor_name == "model.norm.weight":
        path = os.path.join(fptc_dir, "layer_misc", "norm.weight.fptc")
    elif tensor_name == "lm_head.weight":
        path = os.path.join(fptc_dir, "layer_misc", "lm_head.weight.fptc")
    else:
        safe_name = tensor_name.removeprefix("model.")
        path = os.path.join(fptc_dir, "layer_misc", f"{safe_name}.fptc")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing FPTC package for tensor {tensor_name}: {path}"
        )
    return path


def _load_fptc2_binding():
    bind_path = os.getenv("FPTC2_BIND_PATH", DEFAULT_FPTC_BIND_PATH)
    if bind_path and bind_path not in sys.path:
        sys.path.insert(0, bind_path)
    module_name = os.getenv("FPTC2_BIND_MODULE", "fptc2_bindings")
    return importlib.import_module(module_name)


def _shape_to_fptc_rows_cols(shape):
    if len(shape) == 1:
        return 1, int(shape[0])
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    raise ValueError(f"FPTC-2 binding only supports 1D/2D tensors, got {shape}")

################################ SLLM-CONDENSE ################################











def save_lora(model: PeftModel, lora_path: str):
    if not os.path.exists(lora_path):
        os.makedirs(lora_path, exist_ok=True)

    model = model.to("cpu")

    lora_state_dict = get_peft_model_state_dict(model)

    save_dict(lora_state_dict, lora_path)

    # save the config
    if hasattr(model, "peft_config") and model.peft_config:
        adapter_name = getattr(model, "active_adapter", None)
        if adapter_name is None:
            logger.warning("No active_adapter found")
            return

        config = model.peft_config.get(adapter_name, None)
        if config is None:
            logger.warning(f"No config found for adapter: {adapter_name}")
            return

        config.save_pretrained(lora_path)
        logger.info(
            f"Saved LoRA config for adapter: {adapter_name} to {lora_path}"
        )


def load_model(
    model_path: Optional[Union[str, os.PathLike]],
    device_map: DeviceMapType = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    quantization_config: Optional[
        Union[QuantizationConfigMixin, Dict[str, Any]]
    ] = None,
    storage_path: Optional[str] = None,
    fully_parallel: bool = False,
    hf_model_class: str = "AutoModelForCausalLM",
):
    if fully_parallel:
        return fully_parallel_load(
            model_path=model_path,
            hf_model_class=hf_model_class,
            device_map=device_map,
            torch_dtype=torch_dtype,
            quantization_config=quantization_config,
            storage_path=storage_path,
        )

    # if fully_parallel is disabled, we still try to parallelize the model
    # initialization and data loading in the best effort
    return best_effort_load(
        model_path=model_path,
        hf_model_class=hf_model_class,
        device_map=device_map,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        storage_path=storage_path,
    )


def load_model_condense(
    model_path: Optional[Union[str, os.PathLike]],
    device_map: DeviceMapType = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    quantization_config: Optional[
        Union[QuantizationConfigMixin, Dict[str, Any]]
    ] = None,
    storage_path: Optional[str] = None,
    hf_model_class: str = "AutoModelForCausalLM",
):
    # ############# SLLM-CONDENSE #########
    return best_effort_load_condense(
        model_path=model_path,
        hf_model_class=hf_model_class,
        device_map=device_map,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        storage_path=storage_path,
    )


def fully_parallel_load(
    model_path: Optional[Union[str, os.PathLike]],
    hf_model_class: str,
    device_map: DeviceMapType = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    quantization_config: Optional[
        Union[QuantizationConfigMixin, Dict[str, Any]]
    ] = None,
    storage_path: Optional[str] = None,
):
    if not storage_path:
        storage_path = os.getenv("STORAGE_PATH", os.path.expanduser("~/models"))

    start = time.time()
    device_map = _transform_device_map_to_dict(device_map)
    with open(
        os.path.join(storage_path, model_path, "tied_no_split_modules.json"),
        "r",
    ) as f:
        tied_no_split_modules = json.load(f)

    if isinstance(device_map, str):
        with open(
            os.path.join(storage_path, model_path, "no_split_modules.json"),
            "r",
        ) as f:
            no_split_modules = json.load(f)
        device_map = _compute_device_placement_from_map_fast(
            no_split_modules, tied_no_split_modules, device_map
        )

    # TODO: offload `load_dict_non_blocking` to c++ for real parallelism
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(
            load_dict_non_blocking, model_path, device_map, storage_path
        )
        logger.debug(
            f"load_dict_non_blocking takes {time.time() - start} seconds"
        )

        start = time.time()
        config = AutoConfig.from_pretrained(
            f"{os.path.join(storage_path, model_path)}", trust_remote_code=True
        )
        if torch_dtype is not None:
            config.torch_dtype = torch_dtype
        logger.debug(f"load config takes {time.time() - start} seconds")

        start = time.time()
        with init_empty_weights():
            module = importlib.import_module("transformers")
            _class = getattr(module, hf_model_class)
            model = _class.from_config(
                config,
                trust_remote_code=True,
            ).to(config.torch_dtype)
        model.tie_weights()
        logger.debug(f"load model takes {time.time() - start} seconds")

        replica_uuid, state_dict = future.result()

    with torch.no_grad():
        if quantization_config and torch.cuda.is_available():
            model = quantize(
                model,
                state_dict,
                quantization_config,
                torch_dtype,
                device_map,
                model_path,
                replica_uuid,
                logger,
            )
        else:
            if quantization_config is not None:
                logger.debug(
                    "Quantization on current device is not supported yet"
                )

            for name, param in state_dict.items():
                set_module_tensor_to_device(model, name, param.device, param)
        send_module_buffers_to_device(model, device_map)

    dispatch_model(
        model, device_map, skip_keys=model._skip_keys_device_placement
    )
    client = SllmStoreClient("127.0.0.1:8073")
    client.confirm_model_loaded(model_path, replica_uuid)
    model.hf_device_map = device_map
    model.eval()
    return model


def best_effort_load(
    model_path: Optional[Union[str, os.PathLike]],
    hf_model_class: str,
    device_map: DeviceMapType = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    quantization_config: Optional[
        Union[QuantizationConfigMixin, Dict[str, Any]]
    ] = None,
    storage_path: Optional[str] = None,
):
    client = SllmStoreClient("127.0.0.1:8073")
    ret = client.load_into_cpu(model_path)
    if not ret:
        raise ValueError(f"Failed to load model {model_path} into CPU")

    replica_uuid = _get_uuid()
    device_map = _transform_device_map_to_dict(device_map)

    if isinstance(device_map, dict) and (
        torch.device("cpu") in device_map.values()
        or "cpu" in device_map.values()
    ):
        raise ValueError("CPU is not supported in device_map.")

    if not storage_path:
        storage_path = os.getenv("STORAGE_PATH", os.path.expanduser("~/models"))
    start = time.time()
    config = AutoConfig.from_pretrained(
        f"{os.path.join(storage_path, model_path)}", trust_remote_code=True
    )
    if torch_dtype is not None:
        config.torch_dtype = torch_dtype

    logger.debug(f"load config takes {time.time() - start} seconds")
    start = time.time()
    with init_empty_weights():
        module = importlib.import_module("transformers")
        _class = getattr(module, hf_model_class)
        model = _class.from_config(config, trust_remote_code=True).to(
            config.torch_dtype
        )

    model.tie_weights()
    logger.debug(f"load model takes {time.time() - start} seconds")

    start = time.time()
    if isinstance(device_map, str):
        device_map = _compute_device_placement_from_map(
            model, device_map, config.torch_dtype
        )
        logger.debug(f"device_map: {device_map}")
    # check if 'cpu' is in device_map values and raise an exception
    if "cpu" in device_map.values():
        raise ValueError(
            "The GPUs are either unavailable or do not have enough memory. Please ensure they are available and ready for use."  # noqa: E501
        )

    logger.debug(
        f"compute_device_placement takes {time.time() - start} seconds"
    )

    with open(
        os.path.join(storage_path, model_path, "tensor_index.json"), "r"
    ) as f:
        tensor_index = json.load(f)

    tensor_meta_index = {}
    tensor_data_index = {}
    for name, (offset, size, shape, stride, dtype) in tensor_index.items():
        tensor_meta_index[name] = (shape, stride, dtype)
        tensor_data_index[name] = (offset, size)

    start = time.time()
    expanded_device_map = _expand_tensor_name(
        device_map, list(tensor_index.keys())
    )
    device_memory = calculate_device_memory(
        expanded_device_map, tensor_data_index
    )
    # logger.debug(f"calculate_device_memory {device_memory}")
    cuda_memory_ptrs = allocate_cuda_memory(device_memory)
    # cuda_memory_ptrs = { k: [v] for k,v in cuda_memory_ptrs.items()}
    cuda_memory_handles = get_cuda_memory_handles(cuda_memory_ptrs)
    device_uuid_map = get_device_uuid_map()
    # logger.debug(f"determine device_uuid_map {device_uuid_map}")
    tensor_device_offsets, tensor_copy_chunks = calculate_tensor_device_offsets(
        expanded_device_map, tensor_data_index
    )
    logger.debug(f"allocate_cuda_memory takes {time.time() - start} seconds")

    ret = client.load_into_gpu(
        model_path,
        replica_uuid,
        {
            device_uuid_map[device_id]: v
            for device_id, v in tensor_copy_chunks.items()
        },
        {
            device_uuid_map[device_id]: [v]
            for device_id, v in cuda_memory_handles.items()
        },
    )
    if not ret:
        raise ValueError(f"Failed to load model {model_path} into GPU")

    # load model state_dict
    start = time.time()
    state_dict = restore_tensors(
        tensor_meta_index, cuda_memory_ptrs, tensor_device_offsets
    )
    logger.info(f"restore state_dict takes {time.time() - start} seconds")

    with torch.no_grad():
        if quantization_config and torch.cuda.is_available():
            model = quantize(
                model,
                state_dict,
                quantization_config,
                torch_dtype,
                device_map,
                model_path,
                replica_uuid,
                logger,
            )
        else:
            if quantization_config is not None:
                logger.debug(
                    "Quantization on current device is not supported yet"
                )

            for name, param in state_dict.items():
                set_module_tensor_to_device(model, name, param.device, param)
        send_module_buffers_to_device(model, device_map)

    dispatch_model(
        model, device_map, skip_keys=model._skip_keys_device_placement
    )

    client.confirm_model_loaded(model_path, replica_uuid)
    model.eval()
    model.hf_device_map = device_map

    return model














################################ SLLM-CONDENSE ################################

def best_effort_load_condense(
    model_path: Optional[Union[str, os.PathLike]],
    hf_model_class: str,
    device_map: DeviceMapType = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    quantization_config: Optional[
        Union[QuantizationConfigMixin, Dict[str, Any]]
    ] = None,
    storage_path: Optional[str] = None,
):
    timing_sections = []

    def record_timing(name, start, adjusted_time=None):
        wall_time = time.time() - start
        if adjusted_time is None:
            adjusted_time = wall_time
        timing_sections.append((name, wall_time, adjusted_time))
        return wall_time

    start = time.time()
    client = SllmStoreClient("127.0.0.1:8073")
    ret = client.load_into_cpu(model_path)
    if not ret:
        raise ValueError(f"Failed to load model {model_path} into CPU")
    record_timing("load_into_cpu", start)

    start = time.time()
    replica_uuid = _get_uuid()
    device_map = _transform_device_map_to_dict(device_map)

    if isinstance(device_map, dict) and (
        torch.device("cpu") in device_map.values()
        or "cpu" in device_map.values()
    ):
        raise ValueError("CPU is not supported in device_map.")

    if not storage_path:
        storage_path = os.getenv("STORAGE_PATH", os.path.expanduser("~/models"))
    record_timing("setup", start)

    start = time.time()
    config = AutoConfig.from_pretrained(
        f"{os.path.join(storage_path, model_path)}", trust_remote_code=True
    )
    if torch_dtype is not None:
        config.torch_dtype = torch_dtype

    record_timing("load_config", start)

    start = time.time()
    with init_empty_weights():
        module = importlib.import_module("transformers")
        _class = getattr(module, hf_model_class)
        model = _class.from_config(config, trust_remote_code=True).to(
            config.torch_dtype
        )

    model.tie_weights()
    record_timing("init_empty_model", start)

    start = time.time()
    if isinstance(device_map, str):
        device_map = _compute_device_placement_from_map(
            model, device_map, config.torch_dtype
        )
        logger.debug(f"device_map: {device_map}")
    if "cpu" in device_map.values():
        raise ValueError(
            "The GPUs are either unavailable or do not have enough memory. Please ensure they are available and ready for use."  # noqa: E501
        )

    record_timing("compute_device_placement", start)

    start = time.time()
    with open(
        os.path.join(storage_path, model_path, CONDENSE_INDEX_FILENAME), "r"
    ) as f:
        tensor_index = json.load(f)

    tensor_meta_index = {}
    compressed_data_index = {}
    final_data_index = {}
    for name, meta in tensor_index.items():
        shape = meta["shape"]
        stride = meta["stride"]
        dtype = meta["dtype"]
        tensor_meta_index[name] = (shape, stride, dtype)
        compressed_data_index[name] = (
            meta["compressed_offset"],
            meta["compressed_size"],
        )
        final_data_index[name] = (
            meta["uncompressed_offset"],
            meta["uncompressed_size"],
        )
    record_timing("load_condense_index", start)

    start = time.time()
    expanded_device_map = _expand_tensor_name(
        device_map, list(tensor_index.keys())
    )
    compressed_device_memory = calculate_device_memory(
        expanded_device_map, compressed_data_index
    )
    final_device_memory = calculate_device_memory(
        expanded_device_map, final_data_index
    )
    if set(final_device_memory.keys()) != {0}:
        raise ValueError(
            "sllm-condense staging prototype only supports one visible GPU "
            f"(cuda:0), got device memory map {final_device_memory}"
        )
    record_timing("plan_device_memory", start)

    final_cuda_memory_ptrs = None
    staging_cuda_memory_ptrs = None
    try:
        start = time.time()
        final_cuda_memory_ptrs = allocate_cuda_memory(final_device_memory)
        staging_cuda_memory_ptrs = allocate_cuda_memory(compressed_device_memory)
        staging_cuda_memory_handles = get_cuda_memory_handles(
            staging_cuda_memory_ptrs
        )
        device_uuid_map = get_device_uuid_map()
        tensor_device_offsets, _ = calculate_tensor_device_offsets(
            expanded_device_map, final_data_index
        )
        compressed_device_offsets, tensor_copy_chunks = (
            calculate_tensor_device_offsets(
                expanded_device_map, compressed_data_index
            )
        )
        copy_group_by_record = {}
        next_group_id = 0
        for name in expanded_device_map.keys():
            record = compressed_data_index[name]
            if record not in copy_group_by_record:
                copy_group_by_record[record] = next_group_id
                next_group_id += 1
        grouped_tensor_copy_chunks = {}
        for device_id, chunks in tensor_copy_chunks.items():
            grouped_tensor_copy_chunks[device_id] = [
                (
                    src_offset,
                    size,
                    dst_offset,
                    handle_idx,
                    copy_group_by_record[(src_offset, size)],
                )
                for src_offset, size, dst_offset, handle_idx in chunks
            ]
        group_id_by_name = {
            name: copy_group_by_record[compressed_data_index[name]]
            for name in expanded_device_map.keys()
        }
        record_timing("allocate_cuda_memory_and_offsets", start)

        start = time.time()
        ret = client.load_into_gpu(
            model_path,
            replica_uuid,
            {
                device_uuid_map[device_id]: v
                for device_id, v in grouped_tensor_copy_chunks.items()
            },
            {
                device_uuid_map[device_id]: [v]
                for device_id, v in staging_cuda_memory_handles.items()
            },
        )
        if not ret:
            raise ValueError(f"Failed to load model {model_path} into GPU")
        record_timing("load_into_gpu_async", start)

        start = time.time()
        fptc_decompress_time = 0.0
        fptc_call_wall_time = 0.0
        fptc_call_timings = []
        group_wait_time = 0.0
        first_group_wait_time = None
        fptc2 = _load_fptc2_binding()
        staging_addresses = get_cuda_memory_addresses(staging_cuda_memory_ptrs)
        final_addresses = get_cuda_memory_addresses(final_cuda_memory_ptrs)
        device = 0
        for name in expanded_device_map.keys():
            wait_start = time.time()
            group_id = group_id_by_name[name]
            if not client.confirm_gpu_group(
                model_path, replica_uuid, group_id
            ):
                raise ValueError(
                    f"Failed to confirm GPU copy group {group_id} "
                    f"for model {model_path}"
                )
            wait_s = time.time() - wait_start
            group_wait_time += wait_s
            if first_group_wait_time is None:
                first_group_wait_time = wait_s

            meta = tensor_index[name]
            rows, cols = _shape_to_fptc_rows_cols(meta["shape"])
            fptc_call_start = time.time()
            fptc_reported_time = float(
                fptc2.decompress_gpu_package_to_address(
                    staging_addresses[device]
                    + compressed_device_offsets[device][name],
                    meta["compressed_size"],
                    final_addresses[device]
                    + tensor_device_offsets[device][name],
                    meta["uncompressed_size"],
                    rows,
                    cols,
                    meta["dtype"],
                    device,
                )
            )
            fptc_call_wall_s = time.time() - fptc_call_start
            fptc_decompress_time += fptc_reported_time
            fptc_call_wall_time += fptc_call_wall_s
            fptc_call_timings.append(
                (
                    name,
                    group_id,
                    fptc_call_wall_s,
                    fptc_reported_time,
                )
            )
        torch.cuda.synchronize(device)
        decompress_wall_time = time.time() - start
        condense_timing_adjustment = (
            decompress_wall_time - fptc_decompress_time
        )
        timing_sections.append(
            (
                "pipelined_decompress_staging_to_final",
                decompress_wall_time,
                fptc_decompress_time,
            )
        )
    except Exception:
        if staging_cuda_memory_ptrs:
            free_cuda_memory(staging_cuda_memory_ptrs)
        if final_cuda_memory_ptrs:
            free_cuda_memory(final_cuda_memory_ptrs)
        raise
    else:
        start = time.time()
        free_cuda_memory(staging_cuda_memory_ptrs)
        record_timing("free_staging_cuda_memory", start)

    start = time.time()
    state_dict = restore_tensors(
        tensor_meta_index, final_cuda_memory_ptrs, tensor_device_offsets
    )
    record_timing("restore_state_dict", start)

    start = time.time()
    with torch.no_grad():
        if quantization_config and torch.cuda.is_available():
            model = quantize(
                model,
                state_dict,
                quantization_config,
                torch_dtype,
                device_map,
                model_path,
                replica_uuid,
                logger,
            )
        else:
            if quantization_config is not None:
                logger.debug(
                    "Quantization on current device is not supported yet"
                )

            for name, param in state_dict.items():
                set_module_tensor_to_device(model, name, param.device, param)
        send_module_buffers_to_device(model, device_map)
    record_timing("apply_state_dict", start)

    start = time.time()
    dispatch_model(
        model, device_map, skip_keys=model._skip_keys_device_placement
    )
    record_timing("dispatch_model", start)

    start = time.time()
    client.confirm_model_loaded(model_path, replica_uuid)
    model.eval()
    model.hf_device_map = device_map
    record_timing("final_confirm_and_finalize", start)

    section_wall_sum = sum(wall for _, wall, _ in timing_sections)
    section_adjusted_sum = sum(adjusted for _, _, adjusted in timing_sections)
    timing_lines = [
        "====TIMING INFO====",
        f"model_path={model_path}",
    ]
    for name, wall_time, adjusted_time in timing_sections:
        if adjusted_time == wall_time:
            timing_lines.append(f"{name}: wall={wall_time:.6f}s")
        else:
            timing_lines.append(
                f"{name}: wall={wall_time:.6f}s "
                f"adjusted={adjusted_time:.6f}s "
                f"difference={wall_time - adjusted_time:.6f}s"
            )
    timing_lines.append(
        "pipelined_wait_for_groups="
        f"{group_wait_time:.6f}s first_group_wait="
        f"{(first_group_wait_time or 0.0):.6f}s"
    )
    timing_lines.append(
        "fptc_call_wall_sum="
        f"{fptc_call_wall_time:.6f}s fptc_reported_sum="
        f"{fptc_decompress_time:.6f}s"
    )
    for name, group_id, wall_time, reported_time in fptc_call_timings:
        timing_lines.append(
            f"fptc_call name={name} group_id={group_id} "
            f"wall={wall_time:.6f}s reported={reported_time:.6f}s"
        )
    timing_lines.extend(
        [
            f"section_wall_sum={section_wall_sum:.6f}s",
            f"section_adjusted_sum={section_adjusted_sum:.6f}s",
            f"timing_adjustment={condense_timing_adjustment:.6f}s",
        ]
    )
    logger.info("\n".join(timing_lines))

    return model, condense_timing_adjustment

################################ SLLM-CONDENSE ################################









def load_lora(
    model: nn.Module,
    adapter_name: str,
    adapter_path: Optional[Union[str, os.PathLike]],
    device_map: DeviceMapType = "auto",
    storage_path: Optional[str] = None,
    is_trainable: bool = False,
    torch_dtype: Optional[torch.dtype] = None,
):
    if not storage_path:
        storage_path = os.getenv("STORAGE_PATH", os.path.expanduser("~/models"))

    config_path = os.path.join(
        storage_path, adapter_path, "adapter_config.json"
    )
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    lora_config = LoraConfig(**config_dict)

    if lora_config.is_prompt_learning and is_trainable:
        raise ValueError(
            "Cannot set a prompt learning adapter to trainable\
            when loading pretrained adapter."
        )

    lora_config.inference_mode = not is_trainable

    client = SllmStoreClient("127.0.0.1:8073")
    client.register_model(adapter_path)

    model.add_adapter(lora_config, adapter_name=adapter_name)

    replica_uuid, state_dict = load_dict_non_blocking(
        adapter_path, {"": 0}, storage_path
    )

    # https://github.com/huggingface/transformers/blob/de182ba2690fe6c3466f6463c7f4b3a61694b885/src/transformers/integrations/peft.py#L228-L265
    processed_adapter_state_dict = {}
    prefix = "base_model.model."
    for key, value in state_dict.items():
        new_key = key[len(prefix) :] if key.startswith(prefix) else key
        processed_adapter_state_dict[new_key] = value

    incompatible_keys = set_peft_model_state_dict(
        model, processed_adapter_state_dict, adapter_name
    )
    if incompatible_keys is not None:
        err_msg = ""
        origin_name = adapter_path if adapter_path is not None else "state_dict"
        # Check for unexpected keys.
        if (
            hasattr(incompatible_keys, "unexpected_keys")
            and len(incompatible_keys.unexpected_keys) > 0
        ):
            err_msg = (
                f"Loading adapter weights from {origin_name} led to \
                    unexpected keys not found in the model: "
                f"{', '.join(incompatible_keys.unexpected_keys)}. "
            )

        # Check for missing keys.
        missing_keys = getattr(incompatible_keys, "missing_keys", None)
        if missing_keys:
            # Filter missing keys specific to the current adapter, \
            # as missing base model keys are expected.
            lora_missing_keys = [
                k for k in missing_keys if "lora_" in k and adapter_name in k
            ]
            if lora_missing_keys:
                err_msg += (
                    f"Loading adapter weights from {origin_name} led to \
                        missing keys in the model: "
                    f"{', '.join(lora_missing_keys)}"
                )

        if err_msg:
            logger.warning(err_msg)

    # convert base model to PeftModel
    peft_model = get_peft_model(model, lora_config)

    # synchronize
    client.confirm_model_loaded(adapter_path, replica_uuid)

    if lora_config.inference_mode:
        peft_model.eval()

    logger.info(f"Available adapters: {peft_model.peft_config.keys()}")

    return peft_model
