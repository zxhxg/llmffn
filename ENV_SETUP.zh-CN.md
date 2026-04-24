# 环境配置说明

这份文档整理了当前仓库在本机验证通过的一套运行环境，目标是让别人克隆仓库后，能够把 `llmffn`、CUTracer 和 Nsight Compute 配好，并顺利跑通现有脚本。

## 已验证环境

当前这台机器上实际验证通过的组合如下：

- 操作系统：`Ubuntu 24.04.3 LTS`
- Python：`3.10.20`
- Conda 环境名：`llmffn`
- PyTorch：`2.5.1+cu121`
- Transformers：`4.43.1`
- Accelerate：`1.13.0`
- CUDA 可见性：`torch.cuda.is_available() == True`
- GPU 数量：`1`
- Nsight Compute CLI：由 `which ncu` 动态解析
- Nsight Compute 版本：本机验证时为 `2025.2.0.0`
- CUTracer Python CLI：安装后由 `which cutracer` 动态解析
- `cutracer.so`：推荐位于 `third_party/CUTracer/lib/cutracer.so`

说明：

- 当前 PyTorch 使用的是 `cu121` 轮子。
- 当前 `ncu` 来自系统安装的 CUDA 12.9 工具链。
- 这两个版本在这台机器上是可以共存并正常工作的。
- 仓库里附带的安装清单现在默认采用更通用的 `torch==2.5.1` / `torchvision==0.20.1` / `torchaudio==2.5.1` 写法，目的是提高在集群和镜像环境中的安装成功率。

## 当前 llmffn 环境中的关键包

下面是和本仓库直接相关、需要重点关注的包。

### 模型与推理

| 包 | 版本 | 用途 |
|---|---:|---|
| `torch` | `2.5.1+cu121` | GPU 推理与张量计算 |
| `torchvision` | `0.20.1+cu121` | Torch 依赖配套 |
| `torchaudio` | `2.5.1+cu121` | Torch 依赖配套 |
| `transformers` | `4.43.1` | Hugging Face 模型加载与生成 |
| `accelerate` | `1.13.0` | `device_map="auto"`、分层 offload |
| `bitsandbytes` | `0.49.2` | 低比特量化支持 |
| `sentencepiece` | `0.2.1` | 分词器依赖 |
| `safetensors` | `0.7.0` | 模型权重加载 |
| `huggingface_hub` | `0.36.2` | 模型下载与管理 |

### 数据与分析

| 包 | 版本 | 用途 |
|---|---:|---|
| `numpy` | `2.2.6` | 数值处理 |
| `pandas` | `2.3.3` | 表格结果整理 |
| `datasets` | `4.8.4` | 数据集支持 |
| `pyarrow` | `23.0.1` | 数据集后端依赖 |
| `jsonschema` | `4.26.0` | CUTracer trace 验证依赖 |
| `zstandard` | `0.25.0` | 压缩 trace 支持 |
| `tabulate` | `0.10.0` | 命令行表格输出 |
| `rich` | `15.0.0` | 终端输出增强 |

### Triton / Trace / Profiling

| 包 | 版本 | 用途 |
|---|---:|---|
| `importlib_resources` | `7.1.0` | `cutracer` CLI 依赖 |
| `triton` | `3.1.0` | Triton 相关运行时 |
| `tritonparse` | `0.4.3` | Triton 相关解析辅助 |
| `yscope-clp-core` | `0.9.1b1` | CUTracer CLP trace 支持 |
| `click` | `8.3.2` | CLI 基础依赖 |

如果你想查看当前环境完整包列表，可以执行：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llmffn
conda list
python3 -m pip list --format=freeze
```

## 从零配置环境

下面这套步骤适合“新机器克隆仓库后重新配置”。

仓库里现在也附带了两份可直接复现的依赖清单：

- [environment.llmffn.yml](environment.llmffn.yml)
- [requirements.llmffn.txt](requirements.llmffn.txt)

### 1. 克隆仓库

```bash
git clone <your-repo-url> llmffn
cd llmffn
```

### 2. 创建 Conda 环境

推荐直接使用仓库里的 `environment.llmffn.yml`：

```bash
conda env create -f environment.llmffn.yml
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llmffn
```

如果你的机器无法访问 `repo.anaconda.com`，仓库里的 `environment.llmffn.yml` 已经默认切到清华镜像，并显式加了 `nodefaults`，避免 Conda 再回退到官方默认源。

`requirements.llmffn.txt` 里的 `pip` 默认源也已经切到清华 PyPI 镜像，所以直接执行：

```bash
python3 -m pip install -r requirements.llmffn.txt
```

默认就会走：

```text
https://pypi.tuna.tsinghua.edu.cn/simple
```

如果你更想手动创建一个空环境，也可以：

```bash
conda create -n llmffn python=3.10 -y
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llmffn
python3 -m pip install --upgrade pip
```

### 3. 安装 Python 依赖

如果你没有用 `environment.llmffn.yml`，可以直接：

```bash
python3 -m pip install -r requirements.llmffn.txt
```

如果你想把当前账号的 `pip` 也永久切到清华源，可以额外执行：

```bash
mkdir -p ~/.pip
cat > ~/.pip/pip.conf <<'EOF'
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
timeout = 120
EOF
```

或者手动分两步装。先安装 PyTorch：

```bash
python3 -m pip install \
  torch==2.5.1 \
  torchvision==0.20.1 \
  torchaudio==2.5.1 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

再安装本仓库当前验证过的关键 Python 依赖：

```bash
python3 -m pip install \
  transformers==4.43.1 \
  accelerate==1.13.0 \
  bitsandbytes==0.49.2 \
  sentencepiece==0.2.1 \
  safetensors==0.7.0 \
  huggingface_hub==0.36.2 \
  datasets==4.8.4 \
  pandas==2.3.3 \
  pyarrow==23.0.1 \
  numpy==2.2.6 \
  click==8.3.2 \
  rich==15.0.0 \
  tabulate==0.10.0 \
  jsonschema==4.26.0 \
  importlib_resources==7.1.0 \
  zstandard==0.25.0 \
  tritonparse==0.4.3 \
  modelscope==1.35.4
```

说明：

- `environment.llmffn.yml` 和 `requirements.llmffn.txt` 都是基于当前机器上已经验证通过的环境整理出来的。
- 这两份文件只覆盖 Python 侧依赖，不会自动安装系统级工具，例如 `ncu`、NVIDIA 驱动、CUDA toolkit，也不会替你编译 `cutracer.so`。
- 如果你的环境访问不到 `download.pytorch.org`，优先使用当前仓库附带的这两份清单；它们已经避免依赖 PyTorch CUDA 专用 wheel 索引。
- `cutracer` 不再放进通用依赖清单里统一安装，因为它通常应当和你本地编译出的 `cutracer.so` 配套；推荐始终在 `third_party/CUTracer/python` 目录里执行 `pip install .` 或 `pip install -e .`。

## 配置模型

当前脚本默认会从 [scripts/statistic/run_fp16.py](scripts/statistic/run_fp16.py) 读取默认模型路径：

```python
MODEL_ID = str(REPO_ROOT / "models" / "Meta-Llama-3.1-8B")
```

也就是说，最省事的做法是把模型放到：

```text
<repo-root>/models/Meta-Llama-3.1-8B
```

如果你的模型不在这个路径，也可以在脚本运行时显式传：

```bash
--model-id /path/to/your/model
```

## 配置 CUTracer

### 1. 获取并编译 CUTracer

```bash
mkdir -p third_party
cd third_party
git clone https://github.com/facebookresearch/CUTracer.git
cd CUTracer
sudo apt-get install libzstd-dev
./install_third_party.sh
make -j"$(nproc)"
```

构建成功后，关键产物应该在：

```text
third_party/CUTracer/lib/cutracer.so
```

### 2. 安装 CUTracer Python CLI

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llmffn
cd third_party/CUTracer/python
python3 -m pip install .
```

如果你已经先执行了：

```bash
python3 -m pip install -r requirements.llmffn.txt
```

那么 `cutracer` 这一步仍然需要单独执行，不能省略。

安装后可以验证：

```bash
which cutracer
cutracer --help
```

在当前已验证环境中，`cutracer` 的位置是：

```text
<conda-env>/bin/cutracer
```

### 3. 在本仓库中使用 CUTracer

最稳妥的方式有两种：

1. 直接显式传 `--cutracer-so`
2. 预先导出 `CUTRACER_SO`

推荐：

```bash
export CUTRACER_SO=third_party/CUTracer/lib/cutracer.so
```

然后运行：

```bash
python3 scripts/cutracer_ffn_trace/run_full_cutracer_ffn_trace.py \
  --layer 24 \
  --device-map auto \
  --cutracer-so third_party/CUTracer/lib/cutracer.so
```

## 配置 Nsight Compute

### 1. 确认 `ncu` 可用

当前机器上的 `ncu` 可以通过下面的命令确认：

```text
which ncu
```

如果你的 PATH 里没有它，再按本机实际安装位置手动加入：

```bash
export PATH=/path/to/cuda/bin:$PATH
```

验证方式：

```bash
which ncu
ncu --version
```

### 2. 处理 GPU Performance Counter 权限

如果运行 `ncu` 或 [profile_replay_l2_hit_rate.py](scripts/cutracer_ffn_trace/profile_replay_l2_hit_rate.py) 时看到：

```text
ERR_NVGPUCTRPERM
```

说明当前用户没有 GPU performance counters 权限。

当前机器上的状态可以这样查看：

```bash
grep RmProfilingAdminOnly /proc/driver/nvidia/params
```

如果输出是：

```text
RmProfilingAdminOnly: 1
```

说明只允许管理员 profile。

有两种处理方式：

1. 永久开启普通用户 profiling 权限
2. 临时用 `sudo -E` 跑 Nsight Compute 相关脚本

永久方式：

```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-perf.conf
sudo update-initramfs -u -k all
sudo reboot
```

### 3. 关于 `sudo`、锁文件和 TMPDIR

本仓库里的 [profile_replay_l2_hit_rate.py](scripts/cutracer_ffn_trace/profile_replay_l2_hit_rate.py) 已经做了两层兼容：

- 会自动解析 `ncu` 的绝对路径，不依赖 `sudo` 继承 PATH
- 会为 `ncu` 分配私有 `TMPDIR`，避免 `/tmp/nsight-compute-lock` 冲突

所以如果系统权限仍然只允许 root profile，可以直接这样运行：

```bash
sudo -E "$(command -v python3)" \
  scripts/cutracer_ffn_trace/profile_replay_l2_hit_rate.py \
  --run-dir scripts/cutracer_ffn_trace/output/runs/<your_run_dir> \
  --device-map auto
```

## 克隆后建议做的验证

### 基础环境验证

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llmffn

python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
python3 -c "import transformers; print(transformers.__version__)"
which cutracer
which ncu
ncu --version
```

### 仓库脚本验证

先做一次不真正 profile 的 dry-run：

```bash
python3 scripts/cutracer_ffn_trace/profile_replay_l2_hit_rate.py \
  --run-dir scripts/cutracer_ffn_trace/output/runs/layer_24_20260422_111157 \
  --device-map auto \
  --dry-run
```

再做一次 replay 验证：

```bash
python3 scripts/statistic/replay_single_ffn_mlp.py \
  --capture scripts/cutracer_ffn_trace/output/runs/layer_24_20260422_111157/capture.pt \
  --device-map auto
```

如果上面两步都能过，说明：

- Python 环境可用
- 模型可加载
- GPU 可见
- replay 路径没坏

## 常见问题

### `cutracer: command not found`

通常是因为：

- 没有激活 `llmffn`
- 没有在 `third_party/CUTracer/python` 下执行 `pip install .`

### `Could not find cutracer.so`

通常是因为：

- 没有完成 `make`
- `cutracer.so` 不在 `third_party/CUTracer/lib/cutracer.so`
- 没传 `--cutracer-so`

### `ERR_NVGPUCTRPERM`

这是 Nsight Compute 的 GPU 计数器权限问题，不是仓库脚本逻辑错误。

### `--device-map cuda` OOM

对 8B 模型和 12GB 显卡来说很常见。优先尝试：

```bash
--device-map auto
```

### `cutracer trace` 超时退出

本仓库的一键脚本已经把“raw trace 已经完整写出，但最后因为 no-data timeout 退出”的情况当成可接受结果处理。

## 推荐入口

如果你主要是想复现当前仓库里这条链路，最常用的两个入口是：

### CUTracer 访存序列

```bash
python3 scripts/cutracer_ffn_trace/run_full_cutracer_ffn_trace.py \
  --layer 24 \
  --device-map auto \
  --cutracer-so third_party/CUTracer/lib/cutracer.so \
  --processed-preview-lines 50
```

### Nsight Compute L2 hit rate

```bash
sudo -E "$(command -v python3)" \
  scripts/cutracer_ffn_trace/profile_replay_l2_hit_rate.py \
  --run-dir scripts/cutracer_ffn_trace/output/runs/<your_run_dir> \
  --device-map auto
```
