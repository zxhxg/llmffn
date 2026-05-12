import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
CUTRACER_FFN_TRACE_DIR = SCRIPT_DIR.parent / "cutracer_ffn_trace"
if str(CUTRACER_FFN_TRACE_DIR) not in sys.path:
    sys.path.insert(0, str(CUTRACER_FFN_TRACE_DIR))

from common import configure_preferred_blas_library


DEFAULT_HIDDEN_SIZE = 4096
DEFAULT_INTERMEDIATE_SIZE = 14336

REFERENCE_TMA_KERNELS = (
    "sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize64x128x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas",
    "sm90_xmma_gemm_f16f16_f16f32_f32_tn_n_tilesize64x64x64_warpgroupsize1x1x1_execute_segment_k_off_kernel__5x_cublas",
)

REFERENCE_TMA_OPCODES = (
    "UTMALDG.4D",
    "UTMALDG.4D.MULTICAST",
    "UTMASTG.4D",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a minimal Llama-style FFN GEMM workload that triggers the same "
            "Hopper cuBLAS TMA instruction families seen in "
            "layer_24_20260429_105401, without loading the full model."
        ),
    )
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--intermediate-size", type=int, default=DEFAULT_INTERMEDIATE_SIZE)
    parser.add_argument("--batch-tokens", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--preferred-blas",
        choices=["default", "cublas", "cublaslt"],
        default="cublas",
        help="Preferred CUDA BLAS backend requested through torch.backends.cuda.preferred_blas_library.",
    )
    parser.add_argument(
        "--skip-activation",
        action="store_true",
        help="Skip SiLU and elementwise multiply. The TMA coverage comes from the GEMMs either way.",
    )
    parser.add_argument(
        "--print-reference",
        action="store_true",
        help="Print the reference kernel and TMA opcode families from layer_24_20260429_105401.",
    )
    return parser.parse_args()


def check_args(args: argparse.Namespace) -> None:
    if args.hidden_size <= 0 or args.intermediate_size <= 0 or args.batch_tokens <= 0:
        raise ValueError("hidden-size, intermediate-size, and batch-tokens must all be positive.")
    if args.iterations <= 0:
        raise ValueError("iterations must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("This validation workload requires CUDA.")


def make_tensor(shape: tuple[int, ...], device: torch.device, fill_value: float) -> torch.Tensor:
    # Initialize on CPU so CUTracer sees no non-TMA CUDA kernel before the first GEMM.
    cpu_tensor = torch.full(shape, fill_value, device="cpu", dtype=torch.float16)
    return cpu_tensor.to(device=device, non_blocking=False)


def run_ffn_gemm_once(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    skip_activation: bool,
) -> torch.Tensor:
    gate = F.linear(x, gate_weight)
    up = F.linear(x, up_weight)
    hidden = up if skip_activation else F.silu(gate) * up
    return F.linear(hidden, down_weight)


def main() -> None:
    args = parse_args()
    check_args(args)
    configure_preferred_blas_library(args.preferred_blas)

    if args.print_reference:
        print("reference kernels:")
        for kernel in REFERENCE_TMA_KERNELS:
            print(f"  {kernel}")
        print("reference TMA opcode families:")
        for opcode in REFERENCE_TMA_OPCODES:
            print(f"  {opcode}")

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    x = make_tensor((1, args.batch_tokens, args.hidden_size), device, 0.01)
    gate_weight = make_tensor((args.intermediate_size, args.hidden_size), device, 0.02)
    up_weight = make_tensor((args.intermediate_size, args.hidden_size), device, 0.03)
    down_weight = make_tensor((args.hidden_size, args.intermediate_size), device, 0.04)

    torch.cuda.synchronize(device)
    with torch.no_grad():
        torch.cuda.nvtx.range_push("lightweight_tma_ffn_gemm")
        try:
            out = None
            for _ in range(args.iterations):
                out = run_ffn_gemm_once(x, gate_weight, up_weight, down_weight, args.skip_activation)
        finally:
            torch.cuda.nvtx.range_pop()

    torch.cuda.synchronize(device)
    assert out is not None
    first_value = float(out.detach().reshape(-1)[:1].cpu()[0])
    print(f"device: {device}")
    print(f"shapes: x={tuple(x.shape)} gate/up=({args.intermediate_size}, {args.hidden_size}) down=({args.hidden_size}, {args.intermediate_size})")
    print(f"iterations: {args.iterations}")
    print(f"output shape: {tuple(out.shape)} dtype={out.dtype}")
    print(f"first value: {first_value:.6f}")


if __name__ == "__main__":
    main()
