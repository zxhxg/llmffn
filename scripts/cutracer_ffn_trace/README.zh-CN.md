# CUTracer FFN Trace 工作流

这个目录提供了一套最小可用的两阶段工作流，用来借助 CUTracer 跟踪某一层 FFN 的真实访存序列。

如果你是第一次在新机器上配置这个仓库，建议先看仓库根目录下的 [ENV_SETUP.zh-CN.md](/home/wlh/llmffn/ENV_SETUP.zh-CN.md:1)。

这套流程固定采用下面的 token 语义：

- `首个生成 token = prefill 阶段最后一个 prompt token`

也就是说，第一步不是去跑完整的 `generate()` 并在过程中定位第一个生成 token，而是直接执行一次 prompt prefill，然后抓取目标 layer 在最后一个 prompt token 上看到的 FFN 输入向量。第二步再单独重放该层的 `mlp`，这样可以显著降低 trace 噪声。

## 文件说明

- `run_full_cutracer_ffn_trace.py`
  一键入口脚本，会依次执行 capture、CUTracer replay 和 postprocess。
- `capture_first_generated_ffn_input.py`
  抓取目标 layer 在“首个生成 token”语义下对应的 FFN 输入向量。
- `replay_single_ffn_mlp.py`
  从保存的 capture 文件中恢复 FFN 输入，并只执行一次目标 layer 的 `mlp(...)`。这个脚本是 CUTracer 的直接包裹入口。
- `postprocess_cutracer_ffn_trace.py`
  将 CUTracer 的原始 trace 过滤并整理成最终的 FFN 访存序列。
- `common.py`
  公共辅助函数和默认输出路径。

## 一键流程

如果你想用一条命令直接从 capture 跑到 processed 输出，可以执行：

```bash
python3 scripts/cutracer_ffn_trace/run_full_cutracer_ffn_trace.py \
  --layer 0 \
  --device-map auto \
  --cutracer-so /path/to/cutracer.so
```

如果你本机是按常见方式把 CUTracer 放在家目录下，`cutracer.so` 往往会在：

```text
~/CUTracer/lib/cutracer.so
```

旧实验环境里也可能是：

```text
/tmp/CUTracer/lib/cutracer.so
```

它会在下面的目录中为每次运行创建一个独立子目录：

```text
scripts/cutracer_ffn_trace/output/runs/
```

每个运行目录中会包含：

- `capture.pt`
- `raw_trace/`
- `processed_ffn_mem_sequence.jsonl`
- `processed_preview.jsonl`
- `run_summary.json`

其中 `processed_preview.jsonl` 是 `processed_ffn_mem_sequence.jsonl` 的前 10000 行预览文件，方便查看超大处理结果的开头部分；如果需要更多或更少，可以通过 `--processed-preview-lines` 调整。

## 分步流程

### 1. 抓取 FFN 输入

```bash
python3 scripts/cutracer_ffn_trace/capture_first_generated_ffn_input.py \
  --layer 0 \
  --prompt "Explain briefly what the FFN layer does in a transformer."
```

如果显卡放不下完整 FP16 模型，可以改成：

```bash
python3 scripts/cutracer_ffn_trace/capture_first_generated_ffn_input.py \
  --layer 0 \
  --device-map auto \
  --prompt "Explain briefly what the FFN layer does in a transformer."
```

默认会把 capture 文件写到：

```text
scripts/cutracer_ffn_trace/output/captures/
```

### 2. 用 CUTracer 重放该层 FFN

```bash
cutracer trace \
  --cutracer-so /path/to/cutracer.so \
  -i mem_addr_trace \
  --trace-format ndjson \
  --kernel-events full \
  --cpu-callstack auto \
  --output-dir scripts/cutracer_ffn_trace/output/raw_trace/layer0_run \
  -- python3 scripts/cutracer_ffn_trace/replay_single_ffn_mlp.py \
    --capture scripts/cutracer_ffn_trace/output/captures/layer_0_first_generated_token_capture.pt \
    --device-map auto
```

### 3. 后处理原始 trace

```bash
python3 scripts/cutracer_ffn_trace/postprocess_cutracer_ffn_trace.py \
  scripts/cutracer_ffn_trace/output/raw_trace/layer0_run
```

默认会把处理后的 JSONL 输出到：

```text
scripts/cutracer_ffn_trace/output/processed/
```

## 输出语义

处理后的 JSONL 文件中，每一行都代表一条 CUTracer 记录下来的访存事件，并且保持运行时顺序。排序键是：

- `timestamp`
- `trace_index`
- `kernel_launch_id`

每条记录包含：

- `sequence_index`
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
- `source_trace`

## 注意事项

- 这套流程要求目标 layer 最终落在 CUDA 上。如果目标 layer 不在 CUDA，上游脚本会直接报错，并提示优先尝试 `--device-map cuda`。
- 如果显存不足，`--device-map cuda` 可能会 OOM，此时应改用 `--device-map auto`。
- replay 阶段触发的真实 kernel 数量取决于后端实现，不一定只有一个 kernel。一个 `mlp(...)` 可能会对应多个 GEMV / elementwise kernel。
- `cutracer` Python CLI 本身和 `cutracer.so` 都必须可用。如果 `cutracer` 命令导入失败，或者找不到 `cutracer.so`，需要先修复环境。
- 大模型和大矩阵下的 `mem_addr_trace` 文件会非常大，后处理也会花一些时间。
- 某些环境里 `cutracer trace` 可能会在 raw trace 已经写完后因为 timeout 退出。只要 raw trace 完整生成且后处理成功，一键脚本会把这种情况视为可接受结果。

## 用 Nsight Compute 采 replay 的 L2 hit rate

如果你想采 `replay_single_ffn_mlp.py` 这次 replay 的真实 L2 cache hit rate，可以使用：

```bash
python3 scripts/cutracer_ffn_trace/profile_replay_l2_hit_rate.py \
  --run-dir scripts/cutracer_ffn_trace/output/runs/layer_24_20260422_111157 \
  --device-map auto
```

这会对 replay 脚本包一层 `ncu`，并收集 NVIDIA 官方 `MemoryWorkloadAnalysis_Tables` 里直接提供的两个指标：

- `lts__t_sector_op_read_hit_rate.pct`
- `lts__t_sector_op_write_hit_rate.pct`

脚本会生成：

- `*.ncu-rep`
- `*_raw.csv`
- `*_summary.json`
- `profile_replay_l2_command.sh`

说明：

- 这里额外使用了 NVTX range，只对 replay 的目标 `mlp(...)` 范围内的 kernel 做 profile。
- 这里额外使用了 NVTX push/pop range，只对 replay 的目标 `mlp(...)` 范围内的 kernel 做 profile；对应的 `ncu` 过滤参数会写成 `--nvtx-include "ffn_replay_layer_<layer>/"`。
- 命令默认带 `--cache-control none`，这样更接近真实运行时 cache 状态，不会在 profile 前强制清空 cache。
- `ncu` 需要当前用户有 GPU performance counters 权限。如果报 `ERR_NVGPUCTRPERM`，需要先在系统层开启非管理员 profiling 权限，或者改用 root 运行对应命令。



python3 scripts/cutracer_ffn_trace/run_full_cutracer_ffn_trace.py \
  --layer 24 \
  --device-map auto \
  --cutracer-so /home/wlh/CUTracer/lib/cutracer.so \
  --processed-preview-lines 50

python3 scripts/cutracer_ffn_trace/extract_addrs_to_jsonl.py   scripts/cutracer_ffn_trace/output/runs/layer_24_20260422_111157/processed_ffn_mem_sequence.jsonl   --output local.jsonl


sudo -E /home/wlh/miniconda3/envs/llmffn/bin/python3   /home/wlh/llmffn/scripts/cutracer_ffn_trace/profile_replay_l2_hit_rate.py   --run-dir /home/wlh/llmffn/scripts/cutracer_ffn_trace/output/runs/layer_24_20260422_111157   --device-map auto
