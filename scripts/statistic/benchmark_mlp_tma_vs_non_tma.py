import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

SCRIPT_DIR = Path(__file__).resolve().parent
CUTRACER_FFN_TRACE_DIR = SCRIPT_DIR.parent / "cutracer_ffn_trace"
if str(CUTRACER_FFN_TRACE_DIR) not in sys.path:
    sys.path.insert(0, str(CUTRACER_FFN_TRACE_DIR))

from common import configure_preferred_blas_library


DEFAULT_CONFIG_PATH = Path(
    "/HOME/pxyai/pxyaih_0028/Performance01/wlh/llmffn/models/"
    "Meta-Llama-3.1-70B/config.json"
)


if triton is not None:

    @triton.jit
    def _linear_weight_t_kernel(
        a_ptr,
        w_ptr,
        out_ptr,
        M: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k = k0 + offs_k
            a = tl.load(
                a_ptr + offs_m[:, None] * K + k[None, :],
                mask=(offs_m[:, None] < M) & (k[None, :] < K),
                other=0.0,
            )
            w = tl.load(
                w_ptr + offs_n[:, None] * K + k[None, :],
                mask=(offs_n[:, None] < N) & (k[None, :] < K),
                other=0.0,
            )
            acc += tl.dot(a, tl.trans(w))

        tl.store(
            out_ptr + offs_m[:, None] * N + offs_n[None, :],
            acc,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark one Llama-style SwiGLU MLP with a cuBLAS TMA-candidate "
            "path and a Triton non-TMA reference path."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to model config.json. Uses hidden_size/intermediate_size/hidden_act.",
    )
    parser.add_argument("--hidden-size", type=int, default=None, help="Override config hidden_size.")
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=None,
        help="Override config intermediate_size.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1)
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="float16",
        help="Benchmark dtype. Hopper TMA GEMM comparisons are usually fp16 or bf16.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--mode",
        choices=["both", "tma", "non-tma"],
        default="both",
        help="Which path(s) to benchmark.",
    )
    parser.add_argument(
        "--preferred-blas",
        choices=["default", "cublas", "cublaslt"],
        default="cublas",
        help="Preferred CUDA BLAS backend for the PyTorch F.linear path.",
    )
    parser.add_argument("--block-m", type=int, default=16, help="Triton non-TMA matmul block M.")
    parser.add_argument("--block-n", type=int, default=64, help="Triton non-TMA matmul block N.")
    parser.add_argument("--block-k", type=int, default=64, help="Triton non-TMA matmul block K.")
    parser.add_argument("--num-warps", type=int, default=4, help="Triton non-TMA kernel warps.")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare final outputs from both paths and print max absolute difference.",
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def load_mlp_shape(args: argparse.Namespace) -> tuple[int, int, float]:
    with args.config.open("r", encoding="utf-8") as file:
        config = json.load(file)

    hidden_act = str(config.get("hidden_act", ""))
    if hidden_act != "silu":
        raise ValueError(f"This benchmark implements silu SwiGLU MLPs, got hidden_act={hidden_act!r}.")

    hidden_size = args.hidden_size if args.hidden_size is not None else int(config["hidden_size"])
    intermediate_size = (
        args.intermediate_size
        if args.intermediate_size is not None
        else int(config["intermediate_size"])
    )
    initializer_range = float(config.get("initializer_range", 0.02))
    if hidden_size <= 0 or intermediate_size <= 0:
        raise ValueError("hidden_size and intermediate_size must be positive.")
    return hidden_size, intermediate_size, initializer_range


def make_inputs_and_weights(
    hidden_size: int,
    intermediate_size: int,
    initializer_range: float,
    batch_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    x = torch.full((batch_tokens, hidden_size), 0.01, device=device, dtype=dtype)
    gate_w = torch.empty((intermediate_size, hidden_size), device=device, dtype=dtype)
    up_w = torch.empty((intermediate_size, hidden_size), device=device, dtype=dtype)
    down_w = torch.empty((hidden_size, intermediate_size), device=device, dtype=dtype)

    gate_w.normal_(mean=0.0, std=initializer_range, generator=generator)
    up_w.normal_(mean=0.0, std=initializer_range, generator=generator)
    down_w.normal_(mean=0.0, std=initializer_range, generator=generator)
    torch.cuda.synchronize(device)
    return x, gate_w, up_w, down_w


def run_mlp_cublas_tma_candidate(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
) -> torch.Tensor:
    gate = F.linear(x, gate_w)
    up = F.linear(x, up_w)
    return F.linear(F.silu(gate) * up, down_w)


def triton_linear_weight_t(
    a: torch.Tensor,
    weight: torch.Tensor,
    block_m: int,
    block_n: int,
    block_k: int,
    num_warps: int,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("The non-TMA path requires Triton, but 'import triton' failed.")
    if not a.is_cuda or not weight.is_cuda:
        raise RuntimeError("Triton non-TMA path requires CUDA tensors.")
    if not a.is_contiguous():
        a = a.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if a.dim() != 2 or weight.dim() != 2:
        raise ValueError("Expected 2D tensors for linear matmul.")
    m, k = a.shape
    n, weight_k = weight.shape
    if k != weight_k:
        raise ValueError(f"Shape mismatch: a.shape={tuple(a.shape)} weight.shape={tuple(weight.shape)}")

    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _linear_weight_t_kernel[grid](
        a,
        weight,
        out,
        m,
        k,
        n,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=3,
    )
    return out


def run_mlp_triton_non_tma(
    x: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    gate = triton_linear_weight_t(
        x,
        gate_w,
        block_m=args.block_m,
        block_n=args.block_n,
        block_k=args.block_k,
        num_warps=args.num_warps,
    )
    up = triton_linear_weight_t(
        x,
        up_w,
        block_m=args.block_m,
        block_n=args.block_n,
        block_k=args.block_k,
        num_warps=args.num_warps,
    )
    hidden = F.silu(gate) * up
    return triton_linear_weight_t(
        hidden,
        down_w,
        block_m=args.block_m,
        block_n=args.block_n,
        block_k=args.block_k,
        num_warps=args.num_warps,
    )


def benchmark(
    name: str,
    fn: Callable[[], torch.Tensor],
    warmup: int,
    iterations: int,
    repeats: int,
    device: torch.device,
) -> tuple[list[float], torch.Tensor]:
    if warmup < 0 or iterations <= 0 or repeats <= 0:
        raise ValueError("--warmup must be >= 0, --iterations and --repeats must be positive.")

    out = None
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize(device)

    samples_ms = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.nvtx.range_push(name)
        try:
            start.record()
            for _ in range(iterations):
                out = fn()
            end.record()
        finally:
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize(device)
        samples_ms.append(start.elapsed_time(end) / iterations)

    assert out is not None
    return samples_ms, out


def summarize(samples_ms: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(samples_ms),
        "median": statistics.median(samples_ms),
        "min": min(samples_ms),
        "max": max(samples_ms),
    }


def print_result(name: str, samples_ms: list[float]) -> None:
    stats = summarize(samples_ms)
    samples = ", ".join(f"{sample:.4f}" for sample in samples_ms)
    print(
        f"{name:24s} mean={stats['mean']:.4f} ms  "
        f"median={stats['median']:.4f} ms  min={stats['min']:.4f} ms  "
        f"max={stats['max']:.4f} ms  samples=[{samples}]"
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    configure_preferred_blas_library(args.preferred_blas)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = dtype_from_name(args.dtype)
    hidden_size, intermediate_size, initializer_range = load_mlp_shape(args)
    batch_tokens = args.batch_size * args.seq_len
    if batch_tokens <= 0:
        raise ValueError("--batch-size and --seq-len must be positive.")

    x, gate_w, up_w, down_w = make_inputs_and_weights(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        initializer_range=initializer_range,
        batch_tokens=batch_tokens,
        dtype=dtype,
        device=device,
        seed=args.seed,
    )

    print(f"device: {device}  capability={torch.cuda.get_device_capability(device)}")
    print(f"config: {args.config}")
    print(
        "shapes: "
        f"x=({batch_tokens}, {hidden_size}) "
        f"gate/up=({intermediate_size}, {hidden_size}) "
        f"down=({hidden_size}, {intermediate_size})"
    )
    print(f"dtype: {args.dtype}  warmup={args.warmup} iterations={args.iterations} repeats={args.repeats}")
    print(
        "note: the TMA path is the PyTorch F.linear/cuBLAS path and should be "
        "verified on H100 with CUTracer/Nsight by checking for UTMALDG/UTMASTG."
    )
    print("note: the non-TMA path is a plain Triton tl.load/tl.dot/tl.store matmul reference.")

    outputs: dict[str, torch.Tensor] = {}
    timings: dict[str, list[float]] = {}

    if args.mode in {"both", "tma"}:
        samples, out = benchmark(
            "benchmark_mlp_tma_candidate",
            lambda: run_mlp_cublas_tma_candidate(x, gate_w, up_w, down_w),
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
            device=device,
        )
        timings["tma_candidate"] = samples
        outputs["tma_candidate"] = out
        print_result("tma_candidate", samples)

    if args.mode in {"both", "non-tma"}:
        samples, out = benchmark(
            "benchmark_mlp_non_tma_triton",
            lambda: run_mlp_triton_non_tma(x, gate_w, up_w, down_w, args),
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
            device=device,
        )
        timings["non_tma_triton"] = samples
        outputs["non_tma_triton"] = out
        print_result("non_tma_triton", samples)

    if "tma_candidate" in timings and "non_tma_triton" in timings:
        tma_mean = summarize(timings["tma_candidate"])["mean"]
        non_tma_mean = summarize(timings["non_tma_triton"])["mean"]
        print(f"speedup: non_tma_triton / tma_candidate = {non_tma_mean / tma_mean:.3f}x")

    if args.verify and {"tma_candidate", "non_tma_triton"} <= outputs.keys():
        diff = (outputs["tma_candidate"].float() - outputs["non_tma_triton"].float()).abs()
        print(f"verify: max_abs_diff={float(diff.max().cpu()):.6e}")


if __name__ == "__main__":
    main()
