# SLLM-CONDENSE Integration Notes

This document describes the SLLM-CONDENSE changes in this checkout. It is
written for readers who already know the baseline ServerlessLLM flow: models are
converted into SLLM tensor data files, `sllm-store` asynchronously loads model
bytes through host memory, and the Python loader restores tensors from GPU
memory into an initialized Hugging Face model.

SLLM-CONDENSE keeps that overall shape, but changes the payload being stored and
the point at which GPU work can begin. Instead of storing every tensor as raw
uncompressed tensor bytes, it stores per-tensor FPTC-2 compressed packages in
the existing `tensor.data_*` files. At load time those compressed packages are
copied into a temporary GPU staging allocation and decompressed on the GPU into
the final tensor allocation.

The main implementation areas are marked in source with:

```text
############# SLLM-CONDENSE #########
```

or a wider banner with the same text.

## High-Level Idea

Baseline SLLM loads raw tensor bytes:

```text
disk tensor.data_* -> pinned CPU memory -> final GPU allocation -> tensors
```

SLLM-CONDENSE loads compressed packages:

```text
disk tensor.data_* -> pinned CPU memory -> GPU staging allocation
                                       -> FPTC-2 GPU decompression
                                       -> final GPU allocation -> tensors
```

The purpose is to reduce the amount of data moved from storage and host memory
to GPU memory. The cost is that the loader must allocate staging memory, call
the FPTC-2 decompressor, and track when each compressed package has arrived in
GPU memory.

The recent pipelining change makes that last part finer grained. The loader no
longer waits for the entire compressed model to be copied into staging before
starting decompression. Instead, each compressed tensor package is assigned a
copy group. Once `sllm-store` reports that a group is ready in GPU staging
memory, Python immediately calls:

```python
fptc2.decompress_gpu_package_to_address(...)
```

for that tensor while later groups may still be transferring.

This overlaps FPTC-2 decompression for earlier tensors with CPU-to-GPU copies
for later tensors. It does not currently launch multiple FPTC-2 calls
concurrently.

## Save-Time Format

The SLLM-CONDENSE save entry point is in
`ServerlessLLM/sllm_store/sllm_store/transformers.py`:

```python
save_model_condense(model, model_path, fptc_dir=None)
```

It first calls the normal SLLM `save_model(...)`, then rewrites the saved model
directory with FPTC-2 packages:

1. Read the baseline `tensor_index.json`.
2. Locate one `.fptc` file for each tensor.
3. Pack those compressed package bytes into temporary
   `tensor.data.condense_tmp_*` files.
4. Pad each package to an 8-byte boundary.
5. Replace the baseline raw `tensor.data_*` files with compressed
   `tensor.data_*` files.
6. Write `tensor_index.condense.json`.
7. Write `condense_meta.json`.

The compressed index stores both compressed and original tensor metadata:

```json
{
  "compressed_offset": 0,
  "compressed_size": 12345,
  "uncompressed_offset": 0,
  "uncompressed_size": 67890,
  "shape": [4096, 4096],
  "stride": [4096, 1],
  "dtype": "torch.bfloat16",
  "source_dtype": "torch.bfloat16",
  "fptc_path": "layer0/self_attn.q_proj.weight.fptc"
}
```

The external FPTC directory is selected from, in order:

1. `fptc_dir` passed to `save_model_condense(...)`.
2. `SLLM_CONDENSE_FPTC_DIR`.
3. The current hard-coded development default in `transformers.py`.

Tensor package lookup currently assumes the generated FPTC layout used in this
project:

```text
model.layers.N.*      -> layerN/*.fptc
model.embed_tokens.*  -> layer_misc/embed_tokens.weight.fptc
model.norm.weight     -> layer_misc/norm.weight.fptc
lm_head.weight        -> layer_misc/lm_head.weight.fptc
other tensors         -> layer_misc/<safe tensor name>.fptc
```

## Load-Time Path

The load entry point is also in `transformers.py`:

```python
load_model_condense(...)
best_effort_load_condense(...)
```

The loader performs these steps:

1. Ask `sllm-store` to load the model into host memory with `load_into_cpu`.
   This call starts or reuses the store-side load path. The measured wall time
   is not the full disk-to-CPU time when asynchronous store work is still
   outstanding.
2. Initialize an empty Hugging Face model and compute the device map.
3. Read `tensor_index.condense.json`.
4. Compute two memory plans:
   - compressed package bytes for the GPU staging allocation
   - uncompressed tensor bytes for the final GPU allocation
5. Allocate both CUDA memory regions.
6. Export CUDA IPC handles for the staging allocation.
7. Assign copy groups to compressed packages.
8. Ask `sllm-store` to copy compressed bytes into staging with
   `load_into_gpu`.
9. For each tensor in model order:
   - wait for that tensor's copy group with `confirm_gpu_group`
   - call FPTC-2 to decompress from staging address to final address
10. Synchronize CUDA after the decompression loop.
11. Free the staging allocation.
12. Restore tensors from the final allocation and apply them to the model.
13. Run the final whole-model `confirm_model_loaded` as a backstop.

The current staging prototype only supports a single visible GPU. The code
expects the final device memory map to be `{0}`.

## Copy Groups and Readiness

The core pipelining mechanism is a small extension to the SLLM store protocol.

`MemCopyChunk` now carries:

```proto
uint64 group_id = 5;
```

Baseline SLLM callers do not need to provide this field. The Python client
defaults missing values to `0`, so the regular SLLM path continues to work.

SLLM-CONDENSE assigns one group id per distinct compressed package record:

```text
(compressed_offset, compressed_size) -> group_id
```

Those group ids travel through:

```text
Python load_into_gpu request
-> storage.proto MemCopyChunk
-> Python gRPC server
-> C++ MemCopyChunk
-> C++ dispatch queue GpuBatch
```

On the C++ side, each `GpuReplica` tracks:

```text
group_total_bytes_
group_loaded_bytes_
ready_groups_
```

Before dispatch, `Model::ToGpu` counts the total bytes expected for every group.
After each blocking `cudaMemcpyHostToDevice` completes, the store increments the
loaded byte count for that group. When loaded bytes reach total bytes, the group
is marked ready and waiters are notified.

Python waits through a new RPC:

```proto
rpc ConfirmGpuGroup(ConfirmGpuGroupRequest)
    returns (ConfirmGpuGroupResponse);
```

The RPC is implemented in:

```text
ServerlessLLM/sllm_store/sllm_store/server.py
ServerlessLLM/sllm_store/sllm_store/client.py
ServerlessLLM/sllm_store/csrc/sllm_store/checkpoint_store.*
ServerlessLLM/sllm_store/csrc/sllm_store/model.*
```

The older whole-model `ConfirmModel` RPC remains in place. In SLLM-CONDENSE it
is no longer used before decompression, because that would force decompression
to wait until the entire compressed model was staged. It is still called after
the model has been restored and finalized.

## FPTC-2 Address Handoff

FPTC-2 needs raw CUDA addresses for both the compressed package and final output
tensor. SLLM-CONDENSE exposes CUDA allocation addresses from the existing native
checkpoint extension via:

```python
get_cuda_memory_addresses(...)
```

The decompressor receives:

```text
staging_base_address + compressed_device_offset
compressed_size
final_base_address + tensor_device_offset
uncompressed_size
rows
cols
dtype
device
```

The shape adapter currently supports only 1D and 2D tensors:

```text
1D -> rows = 1, cols = shape[0]
2D -> rows = shape[0], cols = shape[1]
```

## Benchmark Integration

The benchmark scripts know about a third format:

```text
sllm-condense
```

Relevant files:

```text
ServerlessLLM/benchmarks/download_models.py
ServerlessLLM/benchmarks/benchmark_utils.py
ServerlessLLM/benchmarks/run-benchmark.sh
ServerlessLLM/benchmarks/test_loading.py
ServerlessLLM/benchmarks/generate_report.py
ServerlessLLM/benchmarks/plot.py
```

`download_models.py` accepts:

```text
--save-format sllm-condense
--fptc-dir PATH
```

`run-benchmark.sh` accepts:

```text
--formats "safetensors sllm sllm-condense"
--fptc-dir PATH
```

It starts `sllm-store` for both `sllm` and `sllm-condense`, and skips the store
for `safetensors`.

The benchmark result `loading_time` now records the full user-visible latency:

```python
loading_time = end_time - start_time
```

It does not subtract the SLLM-CONDENSE timing adjustment. The internal timing
logs still show adjusted values so that experiments can separate full latency
from FPTC-reported kernel time.

## Timing Log Meaning

SLLM-CONDENSE prints a timing block from `best_effort_load_condense`:

```text
====TIMING INFO====
model_path=...
load_into_cpu: wall=...
...
pipelined_decompress_staging_to_final: wall=... adjusted=... difference=...
pipelined_wait_for_groups=... first_group_wait=...
fptc_call_wall_sum=... fptc_reported_sum=...
fptc_call name=... group_id=... wall=... reported=...
section_wall_sum=...
section_adjusted_sum=...
timing_adjustment=...
```

Important interpretation details:

- `load_into_cpu` is the wall time for asking the store to start or reuse the
  CPU load. It may not include all disk-to-CPU work, because the store can still
  be doing asynchronous loading after this call returns.
- `load_into_gpu_async` is the wall time for submitting the GPU copy request.
  It is not the full CPU-to-GPU transfer time.
- `pipelined_decompress_staging_to_final wall` includes time spent waiting for
  per-group readiness plus Python-side FPTC call wall time plus the final CUDA
  synchronize for the loop.
- `pipelined_decompress_staging_to_final adjusted` is the sum of the times
  returned by FPTC-2 itself.
- `difference` is the loader's timing adjustment for that section:

```text
wall - FPTC-reported time
```

- `pipelined_wait_for_groups` is the sum of all Python waits on
  `confirm_gpu_group`. Because those waits are sequential measurements, this
  number is useful for debugging but should not be interpreted as pure transfer
  time.
- `first_group_wait` is often large when no group is ready yet and the loader is
  waiting for the first package it needs.
- `fptc_call_wall_sum` is the sum of Python wall-clock time around each
  `fptc2.decompress_gpu_package_to_address(...)` call.
- `fptc_reported_sum` is the sum of the values returned by FPTC-2. In recent
  logs this has been much smaller than Python wall time, which means the binding
  call includes overhead or synchronization that FPTC-2 is not reporting as
  kernel time.
- `section_wall_sum` is the sum of all logged section wall times. It includes
  wait time and other loader work, not just FPTC call wall time.
- `section_adjusted_sum` is the sum after replacing the pipelined decompress
  wall time with FPTC-reported time.
- `timing_adjustment` is the amount by which adjusted timing differs from full
  wall timing for the pipelined decompression section. The benchmark no longer
  subtracts this from final reported latency.

For estimating how much FPTC overhead remains after all transfer is complete,
the current logs are approximate. A good rough estimate is usually the wall time
of the final FPTC calls that happen after the final group wait returns. An exact
metric would require an additional timestamp for "all GPU copy groups complete"
and another timestamp when the last FPTC call returns.

## What Changed for Baseline SLLM

The regular `sllm` path should keep the same behavior.

The shared store protocol and C++ copy structs now have an optional group id,
but normal SLLM loads do not wait on individual groups. They still use the
whole-model GPU confirmation path.

The main visible benchmark change that applies broadly is that benchmark
`loading_time` reports full measured latency instead of removing the
SLLM-CONDENSE adjusted time.

## Rebuild and Regeneration Notes

After changing `storage.proto`, regenerate the Python gRPC stubs:

```bash
cd ServerlessLLM/sllm_store
python3 -m grpc_tools.protoc \
  -I. \
  --python_out=. \
  --grpc_python_out=. \
  sllm_store/proto/storage.proto
```

After changing the C++ store or checkpoint extension, rebuild the native
extension:

```bash
cd ServerlessLLM/sllm_store
python3 setup.py build_ext --inplace
```

If the package is installed into an environment instead of used in place, rebuild
or reinstall it there as well.

The benchmark runner starts and stops `sllm-store` automatically for SLLM-backed
formats, but it does not rebuild code. To test C++ or proto changes, rebuild
first, then run the benchmark.

## Current Limitations

- Single-GPU staging only in `best_effort_load_condense`.
- FPTC-2 shape adapter supports only 1D and 2D tensors.
- FPTC package layout is project-specific and inferred from tensor names.
- Decompression is pipelined with transfer readiness, but FPTC calls are still
  issued sequentially from Python.
- The current timing logs do not exactly measure how much FPTC wall time
  overlapped with transfers. They expose enough detail to estimate it from the
  last group wait and late FPTC calls.
- Generated proto files include SLLM-CONDENSE comments only because they were
  regenerated from `storage.proto`; manual edits should stay in the source proto.

## Future Work

The current implementation is a first pipelined version. Natural next steps are:

1. Add exact copy-complete and last-FPTC-return timestamps to report
   non-overlapped FPTC overhead directly.
2. Move from sequential FPTC calls to a scheduler that can issue decompression
   as soon as each package is ready, even when another package is already
   decompressing.
3. Use CUDA streams or events to express copy/decompress dependencies without
   forcing unnecessary host-side synchronization.
4. Generalize staging and final allocations across multiple GPUs.
5. Make FPTC package discovery and dtype handling configurable rather than tied
   to the current development directory layout.
