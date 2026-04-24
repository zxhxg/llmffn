# 项目工作流程说明

本文档介绍 `scripts/cutracer_ffn_trace/` 这套工具链的完整工作流程，包括：

- 输入是什么
- 每一步做什么
- 输出有哪些
- 收集到了哪些数据
- 每个数据项分别表示什么

## 1. 目标

这个项目的目标是：

**针对某个模型、某一层 FFN，在“首个生成 token”语义下，收集这层 FFN 的真实动态访存序列。**

这里的“首个生成 token”采用固定定义：

- 它不是 KV cache 之后的第一个单 token decode forward
- 它对应的是 **prefill 阶段最后一个 prompt token**

这样定义的原因是：在 Hugging Face `generate()` 语义下，第一个新 token 的 logits 来自 prompt 最后一个 token 的前向结果。

## 2. 整体流程

整个流程分成三步：

1. `capture`
   从 prompt prefill 中抓取目标 layer 的 FFN 输入向量
2. `replay`
   把这一个 FFN 输入重新喂给目标 layer 的 `mlp(...)`
3. `postprocess`
   从 CUTracer 的原始 trace 中提取并整理出最终访存序列

此外，目录中还提供了一个一键封装入口：

4. `run_full`
   用一条命令串起 capture、replay 和 postprocess

这样做的原因是：

- 如果直接 trace 整个 `generate()`，trace 噪声会非常大
- 如果只 replay 一个目标 layer 的 `mlp`，就能把访存事件收缩到我们真正关心的范围

## 3. 每一步的输入与输出

### 3.1 Capture 阶段

入口脚本：

- [capture_first_generated_ffn_input.py](/home/wlh/llmffn/scripts/cutracer_ffn_trace/capture_first_generated_ffn_input.py)

输入参数：

- `--model-id`
  模型路径或模型名
- `--layer`
  目标 layer 编号
- `--prompt`
  输入 prompt
- `--device-map`
  模型放置策略，支持 `cuda` 或 `auto`
- `--output`
  capture 文件输出路径

输入数据本质上包括两类：

- 模型权重
- 一条 prompt 文本

Capture 阶段做的事：

- 加载模型和 tokenizer
- 在 `model.model.layers[layer].mlp` 上注册 `forward_pre_hook`
- 对 prompt 执行一次 prefill forward
- 取目标 layer 在最后一个 prompt token 上看到的 hidden state
- 把这个 hidden state 当作后续 replay 的 `ffn_input`

输出文件：

- 一个 `.pt` 文件

输出字段：

- `ffn_input`
  目标 layer FFN 的输入向量，1D tensor
- `layer`
  目标层编号
- `prompt`
  本次使用的 prompt 文本
- `prompt_token_count`
  prompt 分词后的 token 数
- `model_id`
  本次使用的模型标识
- `dtype`
  保存时 `ffn_input` 的数据类型
- `device`
  目标 layer 所在设备
- `hidden_size`
  FFN 输入向量长度
- `token_semantics`
  固定字符串，说明这里抓取的是 `first_generated_token_from_prefill_last_prompt_token`

### 3.2 Replay 阶段

入口脚本：

- [replay_single_ffn_mlp.py](/home/wlh/llmffn/scripts/cutracer_ffn_trace/replay_single_ffn_mlp.py)

输入参数：

- `--capture`
  capture 阶段输出的 `.pt` 文件
- `--model-id`
  可选，覆盖 capture 文件里的模型路径
- `--layer`
  可选，覆盖 capture 文件里的 layer 编号
- `--device-map`
  模型放置策略

输入数据本质上包括：

- capture 文件中的 `ffn_input`
- 目标模型权重

Replay 阶段做的事：

- 读取 `ffn_input`
- 加载目标模型
- 定位到 `model.model.layers[layer].mlp`
- 把 `ffn_input` reshape 成 `[1, 1, hidden_size]`
- 只执行一次 `target_mlp(input_tensor)`

输出：

- 脚本本身只打印少量运行结果
- 真正重要的输出由外部 `cutracer trace` 生成

### 3.3 Trace 阶段

入口命令：

- `cutracer trace ... -- python replay_single_ffn_mlp.py ...`

本项目固定推荐的采集方式：

- instrumentation：`mem_addr_trace`
- format：`ndjson`
- `--kernel-events full`
- `--cpu-callstack auto`

这一阶段的输入：

- replay 脚本
- `cutracer.so`
- `CUTracer` Python CLI

这一阶段的输出通常在一个目录里，包括：

- `cutracer_main_*.log`
  主日志
- `cutracer_kernel_events_*.ndjson`
  kernel launch 元数据
- `kernel_*.ndjson`
  各个 kernel 的原始访存 trace
- `kernel_*.cubin`
  kernel cubin 文件

### 3.4 Postprocess 阶段

入口脚本：

- [postprocess_cutracer_ffn_trace.py](/home/wlh/llmffn/scripts/cutracer_ffn_trace/postprocess_cutracer_ffn_trace.py)

输入参数：

- `inputs`
  一个或多个 raw trace 文件或目录
- `--output`
  输出 JSONL 路径
- `--callstack-marker`
  可选，用来限定只保留 callstack 命中的 kernel

做的事：

- 扫描 kernel launch / kernel metadata 记录
- 找出属于 replay 阶段的 kernel launch
- 扫描所有 mem trace 文件
- 只保留属于这些 launch id 的访存记录
- 按时间顺序归并输出

输出：

- 一个最终的处理后 JSONL 文件
- 如果走一键脚本，还会额外生成一个 `processed_preview.jsonl`

`processed_preview.jsonl` 是 `processed_ffn_mem_sequence.jsonl` 的前 10000 行预览文件，用来查看超大结果文件的开头部分。它保持原始 JSONL 行内容不变，只做截取；如果需要调整行数，可以在一键脚本里使用 `--processed-preview-lines`。

## 4. 收集到的结果是什么

最终收集到的结果可以分成三层：

### 4.1 Capture 结果

表示“这个目标 layer 在首个生成 token 语义下的 FFN 输入是什么”。

这是一个 **逻辑输入向量**，不是 trace。

### 4.2 Raw Trace 结果

表示“当这个 FFN 输入被 replay 到目标 layer 时，GPU 实际发生了哪些 kernel launch，以及每个 kernel 的真实访存地址是什么”。

这是最原始、最接近运行时事实的数据。

### 4.3 Processed Trace 结果

表示“从 raw trace 中筛出来的、属于本次 FFN replay 的访存事件序列”。

这是最适合后续分析和消费的结果。

## 5. 处理后 JSONL 的字段含义

处理后每一行都有这些字段：

- `sequence_index`
  后处理阶段重新分配的全局顺序编号，从 1 开始
- `kernel_launch_id`
  本条访存事件所属的 kernel launch 编号
- `trace_index`
  该事件在对应 kernel trace 中的原始顺序编号
- `timestamp`
  CUTracer 记录该事件时的时间戳
- `mangled_name`
  kernel 的 mangled 名称；某些环境下可能为空
- `unmangled_name`
  更可读的 kernel 名称；在当前实现里会优先取 `unmangled_name`，没有时回退到 `kernel_name`
- `cta`
  cooperative thread array 坐标，格式通常是 `[x, y, z]`
- `warp`
  warp 编号
- `pc`
  SASS 指令对应的程序计数器偏移
- `sass`
  对应的 SASS 指令文本
- `addrs`
  该条访存指令访问到的地址列表
- `source_trace`
  这条记录来自哪个原始 ndjson 文件

## 6. 目录中的关键文件及职责

- [common.py](/home/wlh/llmffn/scripts/cutracer_ffn_trace/common.py)
  公共路径、模型加载和 layer/MLP 定位逻辑
- [run_full_cutracer_ffn_trace.py](/home/wlh/llmffn/scripts/cutracer_ffn_trace/run_full_cutracer_ffn_trace.py)
  一键执行完整工作流，并为每次运行生成独立输出目录、`processed_preview.jsonl` 和摘要文件
- [capture_first_generated_ffn_input.py](/home/wlh/llmffn/scripts/cutracer_ffn_trace/capture_first_generated_ffn_input.py)
  负责抓取 FFN 输入
- [replay_single_ffn_mlp.py](/home/wlh/llmffn/scripts/cutracer_ffn_trace/replay_single_ffn_mlp.py)
  负责只重放目标 layer 的 `mlp`
- [postprocess_cutracer_ffn_trace.py](/home/wlh/llmffn/scripts/cutracer_ffn_trace/postprocess_cutracer_ffn_trace.py)
  负责把 raw trace 转成最终序列
- [README.md](/home/wlh/llmffn/scripts/cutracer_ffn_trace/README.md)
  英文版快速使用说明
- [README.zh-CN.md](/home/wlh/llmffn/scripts/cutracer_ffn_trace/README.zh-CN.md)
  中文版快速使用说明

## 6.1 一键脚本的输入与输出

一键入口脚本：

- [run_full_cutracer_ffn_trace.py](/home/wlh/llmffn/scripts/cutracer_ffn_trace/run_full_cutracer_ffn_trace.py)

常用输入参数：

- `--model-id`
- `--layer`
- `--prompt`
- `--device-map`
- `--cutracer-so`
- `--run-name`
- `--output-root`

一键脚本的输出目录结构：

- `capture.pt`
  capture 阶段产物
- `raw_trace/`
  `cutracer trace` 生成的原始 trace 目录
- `processed_ffn_mem_sequence.jsonl`
  后处理后的最终访存序列
- `run_summary.json`
  本次运行的摘要，包括命令行、退出码、产物路径和文件大小

## 7. 目前这套实现的特点

优点：

- 语义明确，专门针对“首个生成 token”
- 避开整段 `generate()` 的大噪声
- 可以拿到真实 GPU 动态访存地址
- 后处理已经改成流式归并，能处理大 trace

当前限制：

- 默认语义固定，不覆盖“首个 KV-cache 单 token decode forward”
- replay 触发的 kernel 可能不止一个
- `mem_addr_trace` 文件会非常大
- `cutracer` 环境本身必须先配置好，包括 Python CLI、`cutracer.so` 和依赖项

## 8. 一句话总结

这套项目做的事情可以概括成：

**先从真实模型运行中抓到目标 layer 的 FFN 输入，再单独重放这层 FFN，并用 CUTracer 记录其真实访存地址，最后整理成可直接分析的访存序列文件。**
