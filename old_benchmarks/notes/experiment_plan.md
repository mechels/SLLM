# ServerlessLLM OPT-1.3B Loading Baseline

This folder sets up the first ServerlessLLM Store baseline for:

```text
facebook/opt-1.3b checkpoint on local storage -> GPU-ready model
```

## Paths

```text
~/sllm/serverlessllm      cloned ServerlessLLM repository
~/sllm/env                Python virtual environment
~/sllm/models/raw         Hugging Face cache
~/sllm/models/sllm        ServerlessLLM Store converted checkpoints
~/sllm/bench              helper scripts
~/sllm/results            benchmark output
```

## Model

```text
facebook/opt-1.3b
backend: transformers
dtype: float16
```

## Prepare Checkpoint

```bash
/home/ben046/sllm/env/bin/python /home/ben046/sllm/bench/prepare_opt13b.py
```

Expected converted checkpoint:

```text
/home/ben046/sllm/models/sllm/facebook/opt-1.3b/tensor_index.json
```

## Prepare Plot Model Set

Recommended initial plot set:

```text
facebook/opt-350m
facebook/opt-1.3b
facebook/opt-2.7b
facebook/opt-6.7b
facebook/opt-13b
```

Prepare missing ServerlessLLM Store checkpoints from the local Hugging Face
cache:

```bash
/home/ben046/sllm/env/bin/python /home/ben046/sllm/bench/prepare_plot_models.py
```

The default Hugging Face cache is:

```text
/mnt/data/vllm_models/hub
```

If a model is missing from that cache and downloads are allowed in your shell,
add:

```bash
--allow-download
```

## Start Store Server

Start this in a separate terminal before benchmarking:

```bash
/home/ben046/sllm/bench/start_sllm_store_one_gpu.sh
```

The wrapper starts the Store with:

```text
CUDA_VISIBLE_DEVICES=0
SLLM_MEM_POOL_SIZE=32GB
```

This is important because the Store server probes visible CUDA devices and can
create CUDA state on every visible GPU. Stop any old Store server that was
started without the mask before running this one-GPU wrapper.

## Benchmark Load

```bash
/home/ben046/sllm/env/bin/python /home/ben046/sllm/bench/bench_sllm_load.py \
  --trials 5 \
  --clear-store-between-trials
```

`--clear-store-between-trials` calls the Store server's `ClearMem` RPC before
each measured load. This frees ServerlessLLM Store's pinned host-memory cache,
so repeated trials re-read the checkpoint from disk into the Store's host memory
instead of immediately reusing the previously loaded host copy.

The benchmark process also defaults to:

```text
CUDA_VISIBLE_DEVICES=0
```

To select a different physical GPU, start both the Store and benchmark with the
same mask, for example:

```bash
CUDA_VISIBLE_DEVICES=3 /home/ben046/sllm/bench/start_sllm_store_one_gpu.sh
/home/ben046/sllm/env/bin/python /home/ben046/sllm/bench/bench_sllm_load.py \
  --trials 5 \
  --cuda-visible-devices 3 \
  --clear-store-between-trials
```

The benchmark writes:

```text
/home/ben046/sllm/results/baseline_load_coldish.jsonl
/home/ben046/sllm/results/baseline_load_coldish.csv
```

## Plot Data Collection

To collect plot-ready aggregate data for:

```text
CRs: 2,3,4,8
decompressor throughputs: 5,10,25,50,100,250,500 GiB/s
trials per model: 5 cold-ish trials
```

run:

```bash
/home/ben046/sllm/env/bin/python /home/ben046/sllm/bench/collect_plot_data.py
```

Outputs:

```text
/home/ben046/sllm/results/plot_trials_coldish.csv
/home/ben046/sllm/results/plot_threshold_summary.csv
/home/ben046/sllm/results/plot_summary_by_pair.csv
```

`plot_threshold_summary.csv` has one row per model and CR.  
`plot_summary_by_pair.csv` has one row per model, CR, and decompressor
throughput, with mean/median estimated speedup.

## Plot Results

After collecting plot data, generate PNG and PDF plots with:

```bash
/home/ben046/sllm/env/bin/python /home/ben046/sllm/bench/plot_load_speedups.py
```

Outputs are written under:

```text
/home/ben046/sllm/results/plots
```

Generated PNG plots:

```text
break_even_decode_throughput
speedup_by_model
checkpoint_load_latency_boxplot
```

The measured interval is:

```text
start: immediately before sllm_store.transformers.load_model()
end: after load_model() returns and torch.cuda.synchronize() completes
```

## Transfer And Compression Metrics

Each row includes an effective raw load bandwidth:

```text
raw_effective_load_gib_s = checkpoint_gib / load_time_s
raw_effective_load_gbps = checkpoint_bytes * 8 / load_time_s / 1e9
```

This is an end-to-end ServerlessLLM load rate, not a pure PCIe or NVMe metric.
That makes it useful for a first compression break-even estimate because it
includes loader overhead.

For each candidate compression ratio, the benchmark writes one break-even
decompression throughput:

```text
cr_<N>x_min_decomp_gib_s_to_beat_raw
```

The default ratios are:

```text
2,4,8,16
```

Example:

```text
cr_4x_min_decomp_gib_s_to_beat_raw
```

is the minimum decompressor throughput, measured in raw output GiB/s, needed to
beat the raw ServerlessLLM load time at CR=4.

For each compression ratio and decompressor throughput pair, the benchmark also
writes a predicted speedup ratio:

```text
cr_<N>x_decomp_<M>gibs_speedup_x
```

The default decompressor throughputs are:

```text
5,10,50,100,500 GiB/s
```

Example:

```text
cr_8x_decomp_100gibs_speedup_x
```

is:

```text
raw_load_time_s / ((raw_load_time_s / 8) + (checkpoint_gib / 100))
```

So `3.1` means the compressed path is estimated to be 3.1x faster than raw
ServerlessLLM loading under the benchmark's simple bandwidth assumptions.
