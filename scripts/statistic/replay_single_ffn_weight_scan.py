import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

SCRIPT_DIR = Path(__file__).resolve().parent
CUTRACER_FFN_TRACE_DIR = SCRIPT_DIR.parent / "cutracer_ffn_trace"
if str(CUTRACER_FFN_TRACE_DIR) not in sys.path:
    sys.path.insert(0, str(CUTRACER_FFN_TRACE_DIR))

from common import (  # noqa: E402
    configure_preferred_blas_library,
    ensure_cuda_module,
    get_target_mlp,
    load_model_and_tokenizer,
    resolve_default_model_id,
)


CUDA_SOURCE = r"""
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ __launch_bounds__(256) void scan_fp16_weight_kernel(
    const unsigned short* __restrict__ data,
    long long n,
    unsigned long long* __restrict__ checksum) {
  unsigned long long local = 0;
  const long long stride = (long long)blockDim.x * (long long)gridDim.x;

  for (long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x; index < n; index += stride) {
    unsigned short value;
    const unsigned short* ptr = data + index;
    asm volatile("ld.global.u16 %0, [%1];" : "=h"(value) : "l"(ptr) : "memory");
    local += (unsigned long long)value;
  }

  if (local != 0) {
    atomicAdd(checksum, local);
  }
}

torch::Tensor scan_one_weight(torch::Tensor weight, int blocks) {
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(weight.scalar_type() == at::kHalf, "weight must be float16");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(blocks > 0, "blocks must be positive");

  auto checksum = torch::zeros({1}, torch::TensorOptions().device(weight.device()).dtype(torch::kUInt64));
  const auto n = (long long)weight.numel();
  const auto* data = reinterpret_cast<const unsigned short*>(weight.data_ptr<at::Half>());
  auto* out = reinterpret_cast<unsigned long long*>(checksum.data_ptr<unsigned long long>());

  scan_fp16_weight_kernel<<<blocks, 256, 0, at::cuda::getCurrentCUDAStream()>>>(data, n, out);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return checksum;
}
"""

CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor scan_one_weight(torch::Tensor weight, int blocks);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scan_one_weight", &scan_one_weight, "Scan one contiguous fp16 CUDA weight tensor with ld.global.u16");
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay one FFN layer as explicit global-memory weight scans. "
            "This is intended to be wrapped by CUTracer to record ordinary LDG addresses "
            "for gate/up/down projection weights on H100."
        ),
    )
    parser.add_argument("--capture", type=Path, default=None, help="Path to the saved capture .pt file.")
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument(
        "--device-map",
        choices=["cuda", "auto"],
        default="cuda",
        help="Model placement strategy. With auto, only the target MLP is moved to cuda:0.",
    )
    parser.add_argument(
        "--preferred-blas",
        choices=["default", "cublas", "cublaslt"],
        default="cublas",
        help="Accepted for CLI compatibility; the scan kernel itself does not call BLAS.",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=4096,
        help="CUDA blocks per weight scan kernel. Default is 4096.",
    )
    parser.add_argument(
        "--extension-name",
        type=str,
        default="ffn_weight_scan_ext",
        help="Name for the torch CUDA extension cache entry.",
    )
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Compile/load the CUDA extension and exit without loading the model or launching scan kernels.",
    )
    return parser.parse_args()


def load_scan_extension(name: str):
    extra_cuda_cflags = ["-O2"]
    arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if arch_list:
        print(f"TORCH_CUDA_ARCH_LIST={arch_list}")
    return load_inline(
        name=name,
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        functions=None,
        extra_cuda_cflags=extra_cuda_cflags,
        with_cuda=True,
        verbose=False,
    )


def require_contiguous_weight(name: str, weight: torch.Tensor) -> torch.Tensor:
    if not weight.is_contiguous():
        raise RuntimeError(
            f"{name}.weight is not contiguous; refusing to copy because the goal is tracing "
            "the model weight allocation addresses."
        )
    if weight.dtype != torch.float16:
        raise RuntimeError(f"{name}.weight must be float16 for this scanner, got {weight.dtype}.")
    return weight


def replay_weight_scan_once(extension, target_mlp, blocks: int) -> torch.Tensor:
    weights = [
        ("gate_proj", require_contiguous_weight("gate_proj", target_mlp.gate_proj.weight)),
        ("up_proj", require_contiguous_weight("up_proj", target_mlp.up_proj.weight)),
        ("down_proj", require_contiguous_weight("down_proj", target_mlp.down_proj.weight)),
    ]

    outputs = []
    torch.cuda.nvtx.range_push("ffn_weight_scan_replay")
    try:
        for name, weight in weights:
            torch.cuda.nvtx.range_push(f"scan_{name}")
            try:
                outputs.append(extension.scan_one_weight(weight, blocks))
            finally:
                torch.cuda.nvtx.range_pop()
    finally:
        torch.cuda.nvtx.range_pop()

    return torch.cat(outputs)


def replay_single_ffn_weight_scan(args: argparse.Namespace) -> torch.Tensor:
    configure_preferred_blas_library(args.preferred_blas)
    if args.capture is None:
        raise RuntimeError("--capture is required unless --compile-only is used.")
    payload = torch.load(args.capture, map_location="cpu", weights_only=True)
    model_id = args.model_id or payload.get("model_id") or resolve_default_model_id()
    layer = args.layer if args.layer is not None else int(payload["layer"])

    load_mode = args.device_map
    required_cuda_module = f"model.layers.{layer}.mlp"
    if args.device_map == "auto":
        load_mode = "cpu"
        required_cuda_module = None

    model, _tokenizer = load_model_and_tokenizer(
        model_id,
        load_mode,
        required_cuda_module=required_cuda_module,
    )
    target_mlp = get_target_mlp(model, layer)
    if args.device_map == "auto":
        if not torch.cuda.is_available():
            raise RuntimeError("Replay requires CUDA, but torch.cuda.is_available() is False.")
        target_mlp = target_mlp.to("cuda:0")
    target_device = ensure_cuda_module(target_mlp, f"layer {layer} mlp")

    extension = load_scan_extension(args.extension_name)
    torch.cuda.synchronize(target_device)

    weights = {
        "gate_proj": target_mlp.gate_proj.weight,
        "up_proj": target_mlp.up_proj.weight,
        "down_proj": target_mlp.down_proj.weight,
    }
    for name, weight in weights.items():
        print(
            f"{name}: shape={tuple(weight.shape)} dtype={weight.dtype} "
            f"device={weight.device} data_ptr={weight.data_ptr()} bytes={weight.numel() * weight.element_size()}"
        )

    checksums = replay_weight_scan_once(extension, target_mlp, args.blocks)
    torch.cuda.synchronize(target_device)
    return checksums


def main() -> None:
    args = parse_args()
    if args.compile_only:
        load_scan_extension(args.extension_name)
        print(f"compiled extension: {args.extension_name}")
        return
    checksums = replay_single_ffn_weight_scan(args)
    print(f"capture: {args.capture}")
    print(f"weight_scan_checksums: {[int(value) for value in checksums.cpu()]}")


if __name__ == "__main__":
    main()
