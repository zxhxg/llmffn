import argparse
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
CUTRACER_FFN_TRACE_DIR = SCRIPT_DIR.parent / "cutracer_ffn_trace"
if str(CUTRACER_FFN_TRACE_DIR) not in sys.path:
    sys.path.insert(0, str(CUTRACER_FFN_TRACE_DIR))

from common import (
    configure_preferred_blas_library,
    ensure_cuda_module,
    get_target_mlp,
    load_model_and_tokenizer,
    resolve_default_model_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay one target layer MLP call from a saved FFN input capture. "
            "This script is designed to be wrapped by CUTracer."
        ),
    )
    parser.add_argument("--capture", type=Path, required=True, help="Path to the saved capture .pt file.")
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Optional override for the model id stored in the capture file.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Optional override for the layer stored in the capture file.",
    )
    parser.add_argument(
        "--device-map",
        choices=["cuda", "auto"],
        default="cuda",
        help="Model placement strategy. Default forces the whole model onto cuda:0.",
    )
    parser.add_argument(
        "--preferred-blas",
        choices=["default", "cublas", "cublaslt"],
        default="cublas",
        help="Preferred CUDA BLAS backend requested through torch.backends.cuda.preferred_blas_library.",
    )
    return parser.parse_args()


def replay_target_mlp_once(target_mlp, input_tensor: torch.Tensor) -> torch.Tensor:
    return target_mlp(input_tensor)


def replay_single_ffn_mlp(args: argparse.Namespace) -> torch.Tensor:
    configure_preferred_blas_library(args.preferred_blas)
    payload = torch.load(args.capture, map_location="cpu", weights_only=True)
    model_id = args.model_id or payload.get("model_id") or resolve_default_model_id()
    layer = args.layer if args.layer is not None else int(payload["layer"])

    load_mode = args.device_map
    required_cuda_module = f"model.layers.{layer}.mlp"
    if args.device_map == "auto":
        # Replay only needs one MLP, so loading the full model on CPU and then moving
        # the target MLP to CUDA is more reliable than asking accelerate to pin only
        # one nested module to GPU.
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

    input_vector = payload["ffn_input"]
    if not isinstance(input_vector, torch.Tensor):
        raise RuntimeError("Capture payload did not contain a tensor field named 'ffn_input'.")
    if input_vector.dim() != 1:
        raise RuntimeError(f"Expected 1D ffn_input vector, got {tuple(input_vector.shape)}.")

    input_dtype = next(target_mlp.parameters()).dtype
    input_tensor = input_vector.to(device=target_device, dtype=input_dtype).reshape(1, 1, -1)

    with torch.no_grad():
        nvtx_range = f"ffn_replay_layer_{layer}"
        torch.cuda.nvtx.range_push(nvtx_range)
        try:
            output = replay_target_mlp_once(target_mlp, input_tensor)
        finally:
            torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize(target_device)
    return output


def main() -> None:
    args = parse_args()
    output = replay_single_ffn_mlp(args)
    print(f"capture: {args.capture}")
    print(f"replayed output shape: {tuple(output.shape)}")
    print(f"replayed output dtype: {output.dtype}")


if __name__ == "__main__":
    main()
