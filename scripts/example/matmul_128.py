from __future__ import annotations

import argparse

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one CUDA matrix multiplication, intended to be wrapped by CUTracer.",
    )
    parser.add_argument("--size", type=int, default=128, help="Matrix side length. Default: 128.")
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Input dtype used for the CUDA matmul.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup matmuls before the traced target call.",
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def build_inputs(size: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    base = torch.arange(size * size, dtype=torch.float32).reshape(size, size)
    a_cpu = (base.remainder(97) / 97.0).contiguous()
    b_cpu = (base.t().remainder(89) / 89.0).contiguous()
    return a_cpu.to(device=device, dtype=dtype), b_cpu.to(device=device, dtype=dtype)


def target_matmul_once(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mm(a, b)


def main() -> None:
    args = parse_args()
    if args.size <= 0:
        raise ValueError("--size must be positive.")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this CUTracer example.")

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    dtype = resolve_dtype(args.dtype)
    a, b = build_inputs(args.size, dtype, device)

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = torch.mm(a, b)
        torch.cuda.synchronize(device)

        nvtx_range = f"matmul_{args.size}x{args.size}_{args.dtype}"
        torch.cuda.nvtx.range_push(nvtx_range)
        try:
            c = target_matmul_once(a, b)
        finally:
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize(device)

    print(f"matrix_size: {args.size}x{args.size}")
    print(f"dtype: {args.dtype}")
    print(f"output_shape: {tuple(c.shape)}")
    print(f"output_dtype: {c.dtype}")
    print(f"output_0_0: {float(c[0, 0].item())}")


if __name__ == "__main__":
    main()
