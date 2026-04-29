from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
STATISTIC_DIR = SCRIPT_DIR.parent / "statistic"
RUN_FP16_PATH = STATISTIC_DIR / "run_fp16.py"
THIRD_PARTY_DIR = REPO_ROOT / "third_party"
DEFAULT_CUTRACER_ROOT = THIRD_PARTY_DIR / "CUTracer"

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"
DEFAULT_CAPTURE_DIR = DEFAULT_OUTPUT_DIR / "captures"
DEFAULT_RAW_TRACE_DIR = DEFAULT_OUTPUT_DIR / "raw_trace"
DEFAULT_PROCESSED_DIR = DEFAULT_OUTPUT_DIR / "processed"
DEFAULT_RUNS_DIR = DEFAULT_OUTPUT_DIR / "runs"

_RUN_FP16_MODULE: Any | None = None
_RUN_FP16_ATTEMPTED = False


def _load_run_fp16_module() -> Any | None:
    global _RUN_FP16_MODULE, _RUN_FP16_ATTEMPTED
    if _RUN_FP16_ATTEMPTED:
        return _RUN_FP16_MODULE
    _RUN_FP16_ATTEMPTED = True

    if not RUN_FP16_PATH.is_file():
        return None

    spec = importlib.util.spec_from_file_location("cutracer_ffn_run_fp16", RUN_FP16_PATH)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _RUN_FP16_MODULE = module
    return _RUN_FP16_MODULE


def resolve_default_model_id() -> str:
    module = _load_run_fp16_module()
    return getattr(module, "MODEL_ID", "")


def ensure_parent_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def configure_preferred_blas_library(preferred: str | None = "cublas") -> str | None:
    if preferred in (None, "", "default"):
        return None

    import torch

    setter = getattr(torch.backends.cuda, "preferred_blas_library", None)
    if setter is None:
        print(
            "preferred_blas_library: unavailable in this PyTorch build; "
            f"requested={preferred}"
        )
        return None

    before = setter()
    setter(preferred)
    after = setter()
    print(f"preferred_blas_library: before={before} requested={preferred} after={after}")
    return str(after)


def default_capture_path(layer: int) -> Path:
    return DEFAULT_CAPTURE_DIR / f"layer_{layer}_first_generated_token_capture.pt"


def default_processed_path(stem: str) -> Path:
    safe_stem = stem.replace(" ", "_")
    return DEFAULT_PROCESSED_DIR / f"{safe_stem}_ffn_mem_sequence.jsonl"


def get_runtime_device(model: Any) -> torch.device:
    run_fp16_module = _load_run_fp16_module()
    if run_fp16_module is not None and hasattr(run_fp16_module, "_get_runtime_device"):
        return run_fp16_module._get_runtime_device(model)

    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    raise RuntimeError("No non-meta parameter device was found.")


def _cpu_max_memory_bytes() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().available * 0.9)
    except Exception:
        return 64 * 1024**3


def _force_module_to_cuda(device_map: dict[str, Any], module_key: str) -> dict[str, Any]:
    updated: dict[str, Any] = {}
    for key, value in device_map.items():
        if key == module_key:
            continue
        if key.startswith(f"{module_key}.") or module_key.startswith(f"{key}."):
            if key.startswith(f"{module_key}."):
                continue
        updated[key] = value
    updated[module_key] = 0
    return updated


def infer_target_module_device_map(model_id: str, module_key: str) -> dict[str, Any]:
    import torch
    from accelerate import infer_auto_device_map, init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    if not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA device is required to force the target layer onto GPU while using "
            f"--device-map auto (requested module {module_key}). In the current process "
            f"torch.cuda.is_available()={torch.cuda.is_available()} and "
            f"torch.cuda.device_count()={torch.cuda.device_count()}."
        )

    config = AutoConfig.from_pretrained(model_id)
    with init_empty_weights():
        empty_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)

    gpu_budget = int(torch.cuda.get_device_properties(0).total_memory * 0.92)
    device_map = infer_auto_device_map(
        empty_model,
        max_memory={0: gpu_budget, "cpu": _cpu_max_memory_bytes()},
        no_split_module_classes=["LlamaDecoderLayer"],
        dtype=torch.float16,
    )
    return _force_module_to_cuda(device_map, module_key)


def build_device_map(
    device_map_mode: str, model_id: str, required_cuda_module: str | None = None
) -> dict[str, Any] | str:
    import torch

    if device_map_mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device-map cuda was requested, but torch.cuda.is_available() is False."
            )
        return {"": "cuda:0"}
    if device_map_mode == "cpu":
        return {"": "cpu"}
    if device_map_mode == "auto":
        if required_cuda_module is not None:
            return infer_target_module_device_map(model_id, required_cuda_module)
        return "auto"
    raise ValueError(f"Unsupported device-map mode: {device_map_mode}")


def load_model_and_tokenizer(
    model_id: str, device_map_mode: str, required_cuda_module: str | None = None
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not model_id:
        raise ValueError(
            "No model id was provided and scripts/statistic/run_fp16.py did not expose MODEL_ID."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map=build_device_map(
                device_map_mode,
                model_id=model_id,
                required_cuda_module=required_cuda_module,
            ),
        )
    except torch.OutOfMemoryError as exc:
        if device_map_mode == "cuda":
            raise RuntimeError(
                "Model loading ran out of CUDA memory while using --device-map cuda. "
                "If you are on a smaller GPU, retry with --device-map auto so Transformers "
                "can offload non-target layers."
            ) from exc
        raise
    model.eval()
    return model, tokenizer


def get_transformer_layers(model: Any) -> Any:
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError(
            "Expected model.model.layers to exist. This helper currently supports Llama-style models."
        )
    return layers


def get_target_mlp(model: Any, layer: int) -> Any:
    layers = get_transformer_layers(model)
    if layer < 0 or layer >= len(layers):
        raise ValueError(f"--layer must be in [0, {len(layers) - 1}], got {layer}.")
    target_mlp = getattr(layers[layer], "mlp", None)
    if target_mlp is None:
        raise RuntimeError(f"Layer {layer} does not expose an .mlp module.")
    return target_mlp


def ensure_cuda_module(module: Any, module_name: str) -> torch.device:
    for param in module.parameters():
        if param.device.type == "cuda":
            return param.device
        if param.device.type != "meta":
            raise RuntimeError(
                f"{module_name} is on device={param.device}, but this workflow requires CUDA. "
                "Try rerunning with --device-map cuda."
            )
    raise RuntimeError(f"Could not find a concrete parameter device for {module_name}.")


def last_token_vector(hidden: torch.Tensor) -> torch.Tensor:
    if hidden.dim() == 1:
        return hidden
    if hidden.shape[-1] <= 0:
        raise RuntimeError(f"Unexpected hidden shape: {tuple(hidden.shape)}")
    flat = hidden.reshape(-1, hidden.shape[-1])
    if flat.shape[0] == 0:
        raise RuntimeError(f"Cannot capture last token from empty hidden state: {tuple(hidden.shape)}")
    return flat[-1]
