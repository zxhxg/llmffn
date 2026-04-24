# CUTracer FFN Trace Workflow

This directory contains a minimal two-stage workflow for tracing the real FFN
memory-access sequence of one target layer with CUTracer.

The workflow fixes the token semantics to:

- `first generated token = prefill last prompt token`

That means the capture step runs one prompt prefill forward and grabs the FFN
input vector seen by the target layer on the final prompt token. The replay step
then executes only that layer's `mlp` once, which keeps the CUTracer trace much
cleaner than tracing a full `generate()` call.

## Files

- `run_full_cutracer_ffn_trace.py`
  One-click entrypoint that runs capture, CUTracer replay, and postprocess in sequence.
- `capture_first_generated_ffn_input.py`
  Captures the target layer FFN input for the first generated token semantics.
- `replay_single_ffn_mlp.py`
  Replays one target layer `mlp(...)` call from a saved capture. Wrap this
  script with CUTracer.
- `postprocess_cutracer_ffn_trace.py`
  Filters raw CUTracer traces down to the replay-triggered FFN memory sequence.
- `common.py`
  Shared helpers and output-path defaults.

## One-click flow

If you want a single command that runs the whole workflow end-to-end, use:

```bash
python3 scripts/cutracer_ffn_trace/run_full_cutracer_ffn_trace.py \
  --layer 0 \
  --device-map auto \
  --cutracer-so /path/to/cutracer.so
```

This creates a per-run directory under:

```text
scripts/cutracer_ffn_trace/output/runs/
```

Each run directory includes:

- `capture.pt`
- `raw_trace/`
- `processed_ffn_mem_sequence.jsonl`
- `processed_preview.jsonl`
- `run_summary.json`

`processed_preview.jsonl` contains the first 10,000 lines from
`processed_ffn_mem_sequence.jsonl` by default, so you can inspect a manageable
slice of a very large processed trace. You can change that limit with
`--processed-preview-lines`.

## Step-by-step flow

### 1. Capture the FFN input

```bash
python3 scripts/cutracer_ffn_trace/capture_first_generated_ffn_input.py \
  --layer 0 \
  --prompt "Explain briefly what the FFN layer does in a transformer."
```

If your GPU cannot hold the whole model in FP16, retry with:

```bash
python3 scripts/cutracer_ffn_trace/capture_first_generated_ffn_input.py \
  --layer 0 \
  --device-map auto \
  --prompt "Explain briefly what the FFN layer does in a transformer."
```

This writes a `.pt` capture under:

```text
scripts/cutracer_ffn_trace/output/captures/
```

### 2. Replay that layer under CUTracer

```bash
cutracer trace \
  -i mem_addr_trace \
  --trace-format ndjson \
  --kernel-events full \
  --cpu-callstack auto \
  --output-dir scripts/cutracer_ffn_trace/output/raw_trace/layer0_run \
  -- python3 scripts/cutracer_ffn_trace/replay_single_ffn_mlp.py \
    --capture scripts/cutracer_ffn_trace/output/captures/layer_0_first_generated_token_capture.pt \
    --device-map auto
```

### 3. Postprocess the raw trace

```bash
python3 scripts/cutracer_ffn_trace/postprocess_cutracer_ffn_trace.py \
  scripts/cutracer_ffn_trace/output/raw_trace/layer0_run
```

This writes a filtered JSONL sequence under:

```text
scripts/cutracer_ffn_trace/output/processed/
```

## Output semantics

The processed JSONL keeps one record per CUTracer memory-access event and
preserves runtime order using `(timestamp, trace_index, kernel_launch_id)`.

Each output record includes:

- `kernel_launch_id`
- `trace_index`
- `timestamp`
- `mangled_name`
- `unmangled_name`
- `cta`
- `warp`
- `pc`
- `sass`
- `addrs`

## Notes

- This workflow assumes the target layer is on CUDA. If the layer ends up on CPU
  or another device, the scripts will fail early and suggest `--device-map cuda`.
- The replay trace may still contain multiple kernels if the backend lowers the
  MLP into several GPU kernels. The postprocessor keeps all replay-triggered
  kernels whose CPU callstack points back to `replay_single_ffn_mlp.py`.
- The `cutracer` CLI itself must be healthy in your environment. If it fails at
  import time, fix that environment issue first before debugging the scripts in
  this directory.
- In some environments `cutracer trace` may exit with a timeout-based code after
  raw trace files have already been produced. The one-click script treats that as
  acceptable if the expected raw trace artifacts exist and postprocessing succeeds.
