# CUTracer FFN Trace 工作流（fuwuqi 服务器分支）

这个目录提供一套服务器上使用 CUTracer 采集某一层 FFN 真实 GPU 访存序列的工具链。当前 `fuwuqi` 分支主要面向 Linux/HPC 服务器环境，尤其是无 sudo 权限、需要 `module load CUDA/...`、以及 H100/sm90 会产生大量 cubin/trace 文件的场景。

如果是第一次配置环境，先看仓库根目录的 [ENV_SETUP.zh-CN.md](../../ENV_SETUP.zh-CN.md)。

## 目标语义

本流程固定采用：

```text
首个生成 token = prefill 阶段最后一个 prompt token
```

也就是说，第一步不 trace 整个 `generate()`，而是先单独跑一次 prompt prefill，抓取目标 layer 在最后一个 prompt token 上看到的 FFN 输入向量。第二步再只 replay 这个目标 layer 的 `mlp(...)`，用 CUTracer 采 replay 的真实访存地址。这样能显著减少整段生成带来的噪声。

## 服务器推荐启动模板

在服务器上每次进入作业节点后，建议先确认 CUDA 工具链和 Python 环境：

```bash
cd ~/Performance01/wlh/llmffn

module load CUDA/12.4
conda activate llmffn

which nvcc
which ptxas
which nvdisasm
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
```

如果 `cutracer.so` 不在仓库内的 `third_party/CUTracer/lib/cutracer.so`，用绝对路径导出：

```bash
export CUTRACER_SO="$HOME/Performance01/wlh/third_party/CUTracer/lib/cutracer.so"
```

H100 上推荐的一键命令：

```bash
python3 scripts/cutracer_ffn_trace/run_full_cutracer_ffn_trace.py \
  --layer 24 \
  --device-map auto \
  --cutracer-so "$CUTRACER_SO" \
  --no-data-timeout-s 120 \
  --no-dump-cubin \
  --processed-preview-lines 50 \
  --delete-raw-trace-after-postprocess
```

说明：

- `--no-dump-cubin` 会向 `cutracer trace` 传 `--no-dump-cubin`，避免 H100/sm90 下大量 `kernel_*.cubin` 文件占用数百 GB。
- `--delete-raw-trace-after-postprocess` 只在 postprocess 成功后删除 `raw_trace/`，不会影响已经生成的 `processed_ffn_mem_sequence.jsonl`。
- 这条命令没有限制访存 trace 大小，因此更适合完整采集；运行过程中仍然需要临时磁盘空间。
- 如果只是调试流程、防止爆盘，可以额外加 `--trace-size-limit-mb 8192`，但一旦触发上限，最终访存序列就是不完整的前缀样本。

## 工具脚本总览

### `run_full_cutracer_ffn_trace.py`

一键入口，按顺序执行：

1. `capture_first_generated_ffn_input.py`
2. `cutracer trace -- ... replay_single_ffn_mlp.py`
3. `postprocess_cutracer_ffn_trace.py`
4. 生成 `processed_preview.jsonl` 和 `run_summary.json`

常用命令：

```bash
python3 scripts/cutracer_ffn_trace/run_full_cutracer_ffn_trace.py \
  --layer 24 \
  --device-map auto \
  --cutracer-so "$CUTRACER_SO" \
  --no-data-timeout-s 120 \
  --no-dump-cubin \
  --processed-preview-lines 50 \
  --delete-raw-trace-after-postprocess
```

主要参数：

- `--model-id`：模型路径，默认来自 `scripts/statistic/run_fp16.py` 的 `MODEL_ID`。
- `--layer`：目标 Transformer layer 编号。
- `--prompt`：capture 阶段使用的 prompt。
- `--device-map {auto,cuda}`：模型放置策略。服务器和 8B 模型优先用 `auto`。
- `--cutracer-so`：`cutracer.so` 路径。
- `--no-data-timeout-s`：CUTracer 无数据超时秒数；H100 上建议从 `120` 起。
- `--no-dump-cubin`：关闭 cubin dump，H100 上强烈建议开启。
- `--trace-size-limit-mb`：限制 CUTracer trace 大小；触发后结果不完整，只适合调试。
- `--processed-preview-lines`：预览文件行数，不影响完整 processed 文件。
- `--delete-raw-trace-after-postprocess`：postprocess 成功后删除 raw trace 中间文件。

输出目录：

```text
scripts/cutracer_ffn_trace/output/runs/layer_<N>_<timestamp>/
```

典型输出：

```text
capture.pt
raw_trace/
processed_ffn_mem_sequence.jsonl
processed_preview.jsonl
run_summary.json
```

### `capture_first_generated_ffn_input.py`

只执行 capture 阶段，生成后续 replay 使用的 `.pt` 文件。

```bash
python3 scripts/cutracer_ffn_trace/capture_first_generated_ffn_input.py \
  --layer 24 \
  --device-map auto \
  --prompt "Explain briefly what the FFN layer does in a transformer." \
  --output scripts/cutracer_ffn_trace/output/captures/layer24_capture.pt
```

输出字段包括：

- `ffn_input`：目标 layer FFN 输入向量，1D tensor。
- `layer`：目标层编号。
- `prompt`：本次 prompt。
- `prompt_token_count`：prompt token 数量。
- `model_id`：模型路径。
- `hidden_size`：输入向量长度。
- `token_semantics`：固定为 `first_generated_token_from_prefill_last_prompt_token`。

### `replay_single_ffn_mlp.py`

这个脚本位于 `scripts/statistic/`，但它是 CUTracer trace 阶段真正执行的 replay 入口。它读取 capture 文件，只跑一次目标 layer 的 `mlp(...)`。

单独验证 replay：

```bash
python3 scripts/statistic/replay_single_ffn_mlp.py \
  --capture scripts/cutracer_ffn_trace/output/captures/layer24_capture.pt \
  --device-map auto
```

手动包一层 CUTracer：

```bash
cutracer trace \
  --cutracer-so "$CUTRACER_SO" \
  -i mem_addr_trace \
  --trace-format ndjson \
  --kernel-events full \
  --cpu-callstack auto \
  --no-data-timeout-s 120 \
  --no-dump-cubin \
  --output-dir scripts/cutracer_ffn_trace/output/raw_trace/layer24_manual \
  -- python3 scripts/statistic/replay_single_ffn_mlp.py \
    --capture scripts/cutracer_ffn_trace/output/captures/layer24_capture.pt \
    --device-map auto
```

注意：`run_full` 的输出是缓存打印的，trace 阶段不会实时刷屏。判断是否还在运行，使用：

```bash
ps -fu "$USER" | grep -E 'cutracer|replay_single|run_full|python3' | grep -v grep
nvidia-smi
du -h --max-depth=2 scripts/cutracer_ffn_trace/output/runs/<run_dir>
```

### `postprocess_cutracer_ffn_trace.py`

把 CUTracer 原始 `.ndjson` 过滤并整理为最终访存序列。

```bash
python3 scripts/cutracer_ffn_trace/postprocess_cutracer_ffn_trace.py \
  scripts/cutracer_ffn_trace/output/runs/<run_dir>/raw_trace \
  --output scripts/cutracer_ffn_trace/output/runs/<run_dir>/processed_ffn_mem_sequence.jsonl
```

可选限定 callstack marker：

```bash
python3 scripts/cutracer_ffn_trace/postprocess_cutracer_ffn_trace.py \
  scripts/cutracer_ffn_trace/output/runs/<run_dir>/raw_trace \
  --output scripts/cutracer_ffn_trace/output/runs/<run_dir>/processed_ffn_mem_sequence.jsonl \
  --callstack-marker replay_target_mlp_once
```

当前分支的 postprocess 对坏行更宽容：如果 CUTracer 生成的 `.ndjson` 中存在非 UTF-8 字节或损坏 JSON 行，会跳过坏记录并在 stderr 打印 warning，继续处理有效记录。

### `extract_addrs_to_jsonl.py`

从 `processed_ffn_mem_sequence.jsonl` 中只抽取 `addrs` 字段，便于下游只消费地址序列。大文件会自动切分。

```bash
python3 scripts/cutracer_ffn_trace/extract_addrs_to_jsonl.py \
  scripts/cutracer_ffn_trace/output/runs/<run_dir>/processed_ffn_mem_sequence.jsonl \
  --output scripts/cutracer_ffn_trace/output/runs/<run_dir>/addrs.jsonl \
  --max-bytes 1073741824
```

输出文件形如：

```text
addrs.part00001.jsonl
addrs.part00002.jsonl
...
```

### `profile_replay_l2_hit_rate.py`

用 Nsight Compute 采 replay 的 L2 hit rate。它依赖 `ncu` 和 GPU performance counters 权限。

先 dry-run 看命令：

```bash
python3 scripts/cutracer_ffn_trace/profile_replay_l2_hit_rate.py \
  --run-dir scripts/cutracer_ffn_trace/output/runs/<run_dir> \
  --device-map auto \
  --dry-run
```

正式采集：

```bash
python3 scripts/cutracer_ffn_trace/profile_replay_l2_hit_rate.py \
  --run-dir scripts/cutracer_ffn_trace/output/runs/<run_dir> \
  --device-map auto
```

如果系统只允许管理员访问 GPU performance counters，可能会报 `ERR_NVGPUCTRPERM`。没有 sudo 权限时需要让管理员开启普通用户 profiling，或者使用已有允许 profiling 的节点/队列。

输出：

```text
<run_dir>/ncu_l2/layer_<N>_replay_l2.ncu-rep
<run_dir>/ncu_l2/layer_<N>_replay_l2_raw.csv
<run_dir>/ncu_l2/layer_<N>_replay_l2_summary.json
<run_dir>/ncu_l2/profile_replay_l2_command.sh
```

### `common.py`

公共辅助模块，不直接作为命令行工具运行。它负责：

- 默认路径解析。
- 从 `scripts/statistic/run_fp16.py` 读取默认 `MODEL_ID`。
- 加载模型和 tokenizer。
- 在 `--device-map auto` 下尽量保证目标 MLP 落到 CUDA。
- 校验目标 layer 和 MLP。

## H100 / 服务器注意事项

### 为什么 H100 输出特别大

H100 是 Hopper/sm90，cuBLAS/cuBLASLt 会使用 `sm90_xmma_gemm...`、HGMMA 和 TMA kernel。CUTracer 记录的是底层 GPU 行为，不是 Python 层调用次数，所以 H100 上同样一次 `target_mlp(...)` 可能暴露出大量 kernel 变体和反汇编信息。

4070S 是 Ada/sm89，没有 H100 的 TMA 路径，因此 raw trace 和 log 通常小很多。

### cubin 文件过大

如果看到大量文件：

```text
kernel_*.cubin
```

且每个约 160MB，说明 cubin dump 没关。检查：

```bash
grep "CUTRACER_DUMP_CUBIN" \
  scripts/cutracer_ffn_trace/output/runs/<run_dir>/raw_trace/cutracer_main_*.log
```

期望看到 disabled 或 0。运行时请加：

```bash
--no-dump-cubin
```

已经生成的 cubin 文件不是 postprocess 必需文件。如果确认没有进程仍在写该目录，可以删除：

```bash
find scripts/cutracer_ffn_trace/output/runs/<run_dir>/raw_trace \
  -maxdepth 1 -type f -name "*.cubin" -delete
```

### trace size limit 的含义

`--trace-size-limit-mb` 是防爆盘保险。触发上限后，访存记录不完整，只能当作前缀样本：

```bash
--trace-size-limit-mb 8192
```

如果目标是完整访存序列，不要设置这个参数，或者设置足够大，并确保临时磁盘空间充足。

### postprocess 成功后清理 raw trace

完整采集时建议使用：

```bash
--delete-raw-trace-after-postprocess
```

它只在 postprocess 成功后删除 `raw_trace/`。如果 trace 或 postprocess 失败，raw trace 会保留，方便排查。

### 检查空间占用

```bash
du -h --max-depth=1 scripts/cutracer_ffn_trace/output/runs

RUN=scripts/cutracer_ffn_trace/output/runs/<run_dir>
RAW="$RUN/raw_trace"
du -h --max-depth=1 "$RUN" "$RAW"
find "$RAW" -maxdepth 1 -type f -name "*.cubin" | wc -l
find "$RAW" -maxdepth 1 -type f -name "*.ndjson" | wc -l
du -ch "$RAW"/*.cubin 2>/dev/null | tail -1
du -ch "$RAW"/*.ndjson 2>/dev/null | tail -1
ls -lhS "$RAW" | head -30
```

## 输出字段

`processed_ffn_mem_sequence.jsonl` 每行是一条访存事件，字段包括：

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

排序依据是：

```text
timestamp -> trace_index -> kernel_launch_id
```

## 常见问题

### `nvcc: not found` / `ptxas: not found` / `nvdisasm not found`

加载 CUDA module：

```bash
module avail cuda
module load CUDA/12.4
which nvcc
which ptxas
which nvdisasm
```

注意服务器模块名可能是 `CUDA/12.4`，不是 `cuda/12.4`。

### `libcusparse.so.12: undefined symbol: __nvJitLinkComplete_12_4`

通常是加载了过旧 CUDA module，例如 `CUDA/12.1`，导致 PyTorch 优先拿到旧的 `libnvJitLink.so.12`。优先使用：

```bash
module unload CUDA/12.1
module load CUDA/12.4
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### `No-data hang detected`

H100 上某些 cuBLAS kernel 采集启动慢，默认 15 秒可能太短。建议：

```bash
--no-data-timeout-s 120
```

如果仍然误杀，可临时设为 `0` 关闭无数据超时，但要确保 Slurm 时间和磁盘空间足够。

### postprocess 报 UnicodeDecodeError

当前分支的 postprocess 已经对非 UTF-8 或坏 JSON 行做容错。如果服务器代码不是最新版，先同步 `postprocess_cutracer_ffn_trace.py`，再重跑 postprocess。

### `__vsc_prompt_cmd_original: command not found`

这是 VS Code shell prompt 集成残留，通常不影响 CUTracer、CUDA 或 Python 运行。
