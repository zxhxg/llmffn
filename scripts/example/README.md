# CUTracer 128x128 Matmul Example

This directory contains a small CUTracer example for comparing address patterns
across GPUs.

## Run the matmul directly

```bash
python3 scripts/example/matmul_128.py --size 128 --dtype float16
```

## Run with CUTracer and summarize addresses

```bash
python3 scripts/example/run_cutracer_matmul_128.py \
  --size 128 \
  --dtype float16 \
  --cutracer-so "$CUTRACER_SO" \
  --no-dump-cubin
```

Outputs are written under:

```text
scripts/example/output/runs/matmul_128_float16_<timestamp>/
```

Important files:

- `raw_trace/`: CUTracer raw `.ndjson` files.
- `addr_stats.json`: statistics for memory-address records.
- `addr_preview.jsonl`: the first matching memory-address events.
- `run_summary.json`: command, return code, and a compact stats digest.

The default statistics focus on kernels whose CPU callstack contains
`target_matmul_once`, so the warmup matmul is not included when Python
callstacks are available. If the CUTracer environment does not emit a matching
Python callstack, the script falls back to all kernel metadata and records that
in `addr_stats.json` as:

```json
"selection_mode": "fallback_all_kernel_metadata_no_marker_match"
```
