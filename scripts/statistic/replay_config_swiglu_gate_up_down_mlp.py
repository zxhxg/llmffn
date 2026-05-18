import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
CUTRACER_FFN_TRACE_DIR = SCRIPT_DIR.parent / "cutracer_ffn_trace"
if str(CUTRACER_FFN_TRACE_DIR) not in sys.path:
    sys.path.insert(0, str(CUTRACER_FFN_TRACE_DIR))

from common import configure_preferred_blas_library


DEFAULT_CONFIG_PATH = Path(
    "/HOME/pxyai/pxyaih_0028/Performance01/wlh/llmffn/models/"
    "Meta-Llama-3.1-70B/config.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay one Llama-style SwiGLU MLP from config-derived shapes only. "
            "This avoids loading checkpoint weights while preserving the dense "
            "MLP access pattern for CUTracer."
        ),
    )
    parser.add_argument(
        "--capture",
        type=Path,
        default=None,
        help=(
            "Optional saved capture .pt file. If omitted, this script synthesizes "
            "a constant input tensor with shape (batch_size, seq_len, hidden_size)."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the model config.json used to derive the MLP shape.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Optional layer id used only for the NVTX range name. Defaults to capture['layer'] or 0.",
    )
    parser.add_argument(
        "--device-map",
        choices=["cuda", "auto"],
        default="cuda",
        help=(
            "Compatibility with replay_single_ffn_mlp.py. Since this script only "
            "creates one MLP, both modes place it on cuda:0."
        ),
    )
    parser.add_argument(
        "--preferred-blas",
        choices=["default", "cublas", "cublaslt"],
        default="cublas",
        help="Preferred CUDA BLAS backend requested through torch.backends.cuda.preferred_blas_library.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Dtype for the synthetic MLP weights and replay input.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Seed for synthetic random weights.")
    parser.add_argument(
        "--init-device",
        choices=["cuda", "cpu"],
        default="cuda",
        help=(
            "Where to initialize synthetic weights and synthetic inputs. CUDA is "
            "faster and uses the target GPU directly; CPU avoids CUDA random-init "
            "kernels before the replay range, then copies tensors to CUDA."
        ),
    )
    parser.add_argument(
        "--input-fill",
        type=float,
        default=0.01,
        help="Constant used when synthesizing an input tensor without --capture.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Synthetic input batch size used when --capture is omitted.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=1,
        help="Synthetic input sequence length used when --capture is omitted.",
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_mlp_config(config_path: Path) -> dict[str, object]:
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    required_keys = ("hidden_act", "hidden_size", "intermediate_size")
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise KeyError(f"Missing required config key(s): {', '.join(missing)}")

    return {
        "hidden_act": str(config["hidden_act"]),
        "hidden_size": int(config["hidden_size"]),
        "intermediate_size": int(config["intermediate_size"]),
        "initializer_range": float(config.get("initializer_range", 0.02)),
        "mlp_bias": bool(config.get("mlp_bias", False)),
    }


def make_random_weight(
    shape: tuple[int, int],
    dtype: torch.dtype,
    initializer_range: float,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    weight = torch.empty(shape, device=device, dtype=dtype)
    try:
        return weight.normal_(mean=0.0, std=initializer_range, generator=generator)
    except RuntimeError as exc:
        message = str(exc).lower()
        unsupported_cpu_half_normal = "normal" in message and (
            "not implemented" in message or "unsupported" in message
        )
        if (
            device.type != "cpu"
            or dtype not in (torch.float16, torch.bfloat16)
            or not unsupported_cpu_half_normal
        ):
            raise
        del weight
        weight_fp32 = torch.empty(shape, device="cpu", dtype=torch.float32)
        weight_fp32.normal_(mean=0.0, std=initializer_range, generator=generator)
        return weight_fp32.to(dtype=dtype)


class RandomNoBiasLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype: torch.dtype,
        initializer_range: float,
        generator: torch.Generator,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        weight = make_random_weight(
            (out_features, in_features),
            dtype=dtype,
            initializer_range=initializer_range,
            generator=generator,
            device=device,
        )
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


class LlamaSwiGLUGateUpDownMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        dtype: torch.dtype,
        initializer_range: float,
        seed: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        if hidden_act != "silu":
            raise ValueError(
                "This replay script implements the Llama SwiGLU path and expects "
                f"hidden_act='silu', got {hidden_act!r}."
            )

        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = RandomNoBiasLinear(
            hidden_size,
            intermediate_size,
            dtype=dtype,
            initializer_range=initializer_range,
            generator=generator,
            device=device,
        )
        self.up_proj = RandomNoBiasLinear(
            hidden_size,
            intermediate_size,
            dtype=dtype,
            initializer_range=initializer_range,
            generator=generator,
            device=device,
        )
        self.down_proj = RandomNoBiasLinear(
            intermediate_size,
            hidden_size,
            dtype=dtype,
            initializer_range=initializer_range,
            generator=generator,
            device=device,
        )
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def replay_target_mlp_once(target_mlp: nn.Module, input_tensor: torch.Tensor) -> torch.Tensor:
    return target_mlp(input_tensor)


def ensure_cuda_available() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("Replay requires CUDA, but torch.cuda.is_available() is False.")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    return device


def load_optional_capture(capture_path: Path | None) -> dict[str, object] | None:
    if capture_path is None:
        return None
    return torch.load(capture_path, map_location="cpu", weights_only=True)


def resolve_layer(layer_arg: int | None, payload: dict[str, object] | None) -> int:
    if layer_arg is not None:
        return layer_arg
    if payload is not None and "layer" in payload:
        return int(payload["layer"])
    return 0


def build_input_tensor(
    payload: dict[str, object] | None,
    hidden_size: int,
    dtype: torch.dtype,
    target_device: torch.device,
    init_device: torch.device,
    input_fill: float,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    if payload is not None:
        input_vector = payload["ffn_input"]
        if not isinstance(input_vector, torch.Tensor):
            raise RuntimeError("Capture payload did not contain a tensor field named 'ffn_input'.")
        if input_vector.dim() != 1:
            raise RuntimeError(f"Expected 1D ffn_input vector, got {tuple(input_vector.shape)}.")
        if input_vector.numel() != hidden_size:
            raise RuntimeError(
                "Capture ffn_input width does not match config hidden_size: "
                f"{input_vector.numel()} != {hidden_size}."
            )
        return input_vector.to(device=target_device, dtype=dtype).reshape(1, 1, -1)

    if batch_size <= 0 or seq_len <= 0:
        raise ValueError("--batch-size and --seq-len must be positive.")

    input_tensor = torch.full(
        (batch_size, seq_len, hidden_size),
        input_fill,
        device=init_device,
        dtype=dtype,
    )
    if init_device != target_device:
        input_tensor = input_tensor.to(device=target_device)
    return input_tensor


def replay_config_swiglu_gate_up_down_mlp(args: argparse.Namespace) -> torch.Tensor:
    configure_preferred_blas_library(args.preferred_blas)
    target_device = ensure_cuda_available()
    dtype = dtype_from_name(args.dtype)

    mlp_config = load_mlp_config(args.config)
    payload = load_optional_capture(args.capture)
    layer = resolve_layer(args.layer, payload)
    init_device = target_device if args.init_device == "cuda" else torch.device("cpu")

    target_mlp = LlamaSwiGLUGateUpDownMLP(
        hidden_size=int(mlp_config["hidden_size"]),
        intermediate_size=int(mlp_config["intermediate_size"]),
        hidden_act=str(mlp_config["hidden_act"]),
        dtype=dtype,
        initializer_range=float(mlp_config["initializer_range"]),
        seed=args.seed,
        device=init_device,
    ).eval()
    if init_device != target_device:
        target_mlp = target_mlp.to(device=target_device)

    input_tensor = build_input_tensor(
        payload=payload,
        hidden_size=int(mlp_config["hidden_size"]),
        dtype=dtype,
        target_device=target_device,
        init_device=init_device,
        input_fill=args.input_fill,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
    )

    torch.cuda.synchronize(target_device)
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
    output = replay_config_swiglu_gate_up_down_mlp(args)
    print(f"capture: {args.capture if args.capture is not None else '<synthetic input>'}")
    print(f"config: {args.config}")
    print(f"device-map: {args.device_map} -> cuda:0")
    print(f"init-device: {args.init_device}")
    print(f"replayed output shape: {tuple(output.shape)}")
    print(f"replayed output dtype: {output.dtype}")


if __name__ == "__main__":
    main()
