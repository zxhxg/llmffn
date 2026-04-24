import argparse
import json
import math
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from run_fp16 import MODEL_ID, _get_runtime_device


DEFAULT_PROMPT = "Explain briefly what the FFN layer does in a transformer."
DEFAULT_LAYER = 0
DEFAULT_TOKEN_PHASE = "decode"
DEFAULT_TOKEN_INDEX = 1
DEFAULT_MAX_NEW_TOKENS = 1
DEFAULT_OUTPUT_TILE = 128
DEFAULT_REDUCTION_TILE = 128
DEFAULT_VECTOR_TILE = 256
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().with_name("ffn_dense_address_trace_output")

_TRACE_MODEL = None
_TRACE_TOKENIZER = None
_TRACE_MODEL_KEY = None


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Capture one dense FFN execution on GPU and export a logical memory-address "
            "trace for gate_proj, up_proj, silu*mul, and down_proj."
        ),
    )
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument(
        "--device-map",
        choices=["cuda", "auto"],
        default="cuda",
        help="Model placement strategy. Default forces the whole model onto cuda:0.",
    )
    parser.add_argument(
        "--quantization",
        choices=["none", "8bit", "4bit"],
        default="none",
        help="Optional bitsandbytes quantization strategy used while loading the model.",
    )
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER, help="Target FFN layer index.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument(
        "--token-phase",
        choices=["prefill", "decode"],
        default=DEFAULT_TOKEN_PHASE,
        help="Choose whether to trace one prompt token or one generated token.",
    )
    parser.add_argument(
        "--token-index",
        type=int,
        default=DEFAULT_TOKEN_INDEX,
        help="1-based token index inside the chosen phase.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Number of decode tokens to generate. Must cover the chosen decode token.",
    )
    parser.add_argument("--output-tile", type=int, default=DEFAULT_OUTPUT_TILE)
    parser.add_argument("--reduction-tile", type=int, default=DEFAULT_REDUCTION_TILE)
    parser.add_argument("--vector-tile", type=int, default=DEFAULT_VECTOR_TILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_trace_model(model_id, device_map_mode, quantization_mode):
    global _TRACE_MODEL, _TRACE_TOKENIZER, _TRACE_MODEL_KEY

    cache_key = (model_id, device_map_mode, quantization_mode)
    if (
        _TRACE_MODEL is not None
        and _TRACE_TOKENIZER is not None
        and _TRACE_MODEL_KEY == cache_key
    ):
        return _TRACE_MODEL, _TRACE_TOKENIZER

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device_map_mode == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device-map cuda was requested, but torch.cuda.is_available() is False."
            )
        device_map = {"": "cuda:0"}
    else:
        device_map = "auto"

    quantization_config = None
    model_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": device_map,
    }
    if quantization_mode == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization_mode == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        # Quantized bitsandbytes models do not support a later `.to(device)` dispatch.
        # Let Transformers/Accelerate place them automatically.
        if device_map_mode == "cuda":
            model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    model.eval()

    _TRACE_MODEL = model
    _TRACE_TOKENIZER = tokenizer
    _TRACE_MODEL_KEY = cache_key
    return _TRACE_MODEL, _TRACE_TOKENIZER


def token_count_of_hidden(hidden):
    if hidden.dim() == 1:
        return 1
    return math.prod(hidden.shape[:-1])


def token_view_from_flat_index(hidden, flat_index):
    if hidden.dim() == 1:
        if flat_index != 0:
            raise IndexError(f"flat_index must be 0 for 1D hidden state, got {flat_index}.")
        return hidden

    leading_shape = list(hidden.shape[:-1])
    total = math.prod(leading_shape)
    if flat_index < 0 or flat_index >= total:
        raise IndexError(f"flat_index must be in [0, {total - 1}], got {flat_index}.")

    coords = [0] * len(leading_shape)
    remaining = flat_index
    for dim in range(len(leading_shape) - 1, -1, -1):
        size = leading_shape[dim]
        coords[dim] = remaining % size
        remaining //= size
    return hidden[tuple(coords)]


def make_capture_state(token_phase, token_index):
    return {
        "token_phase": token_phase,
        "token_index": token_index,
        "prompt_token_count": 0,
        "prefill_seen": 0,
        "decode_seen": 0,
        "current_local_index": None,
        "captured_phase": None,
        "captured_index": None,
        "captured_source": None,
        "hook_call_count": 0,
        "ffn_input": None,
        "gate_linear": None,
        "up_linear": None,
        "intermediate": None,
        "ffn_output": None,
    }


def resolve_local_index(state, token_count):
    remaining_prefill = max(state["prompt_token_count"] - state["prefill_seen"], 0)
    prefill_count = min(token_count, remaining_prefill)
    decode_count = token_count - prefill_count

    local_index = None
    phase = None

    if state["token_phase"] == "prefill" and prefill_count > 0:
        start = state["prefill_seen"] + 1
        end = state["prefill_seen"] + prefill_count
        if start <= state["token_index"] <= end:
            local_index = state["token_index"] - start
            phase = "prefill"
            source = "prefill"
        else:
            source = None
    else:
        source = None

    if state["token_phase"] == "decode" and decode_count > 0 and local_index is None:
        start = state["decode_seen"] + 2
        end = state["decode_seen"] + decode_count + 1
        if start <= state["token_index"] <= end:
            local_index = prefill_count + (state["token_index"] - start)
            phase = "decode"
            source = "decode_forward"

    if (
        state["token_phase"] == "decode"
        and state["token_index"] == 1
        and prefill_count > 0
        and local_index is None
    ):
        local_index = prefill_count - 1
        phase = "decode"
        source = "prefill_last_token"

    state["prefill_seen"] += prefill_count
    state["decode_seen"] += decode_count
    state["current_local_index"] = local_index

    if local_index is not None:
        state["captured_phase"] = phase
        state["captured_index"] = state["token_index"]
        state["captured_source"] = source

    return local_index


def capture_vector_if_needed(state, key, tensor):
    local_index = state["current_local_index"]
    if local_index is None or state[key] is not None:
        return

    vector = token_view_from_flat_index(tensor.detach(), local_index)
    if vector.dim() != 1:
        raise RuntimeError(f"Expected captured tensor {key} to become 1D, got shape={tuple(vector.shape)}.")
    state[key] = vector


def ensure_cuda_tensor(name, tensor):
    if tensor.device.type != "cuda":
        raise RuntimeError(
            f"{name} was captured on device={tensor.device}, but this script is intended for GPU traces."
        )


def tensor_meta(name, tensor):
    return {
        "name": name,
        "device": str(tensor.device),
        "dtype": str(tensor.dtype),
        "python_type": type(tensor).__name__,
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "storage_offset": int(tensor.storage_offset()),
        "element_size": int(tensor.element_size()),
        "numel": int(tensor.numel()),
        "nbytes": int(tensor.numel() * tensor.element_size()),
        "data_ptr": int(tensor.data_ptr()),
        "is_contiguous": bool(tensor.is_contiguous()),
    }


def vector_event(op_name, access, tensor_name, tensor, start, length):
    return {
        "op": op_name,
        "access": access,
        "tensor": tensor_name,
        "layout": "strided_vector",
        "index_range": [int(start), int(start + length)],
        "start_addr": int(tensor.data_ptr() + start * tensor.stride(0) * tensor.element_size()),
        "count": int(length),
        "stride_bytes": int(tensor.stride(0) * tensor.element_size()),
        "element_size": int(tensor.element_size()),
        "dtype": str(tensor.dtype),
    }


def matrix_tile_event(op_name, access, tensor_name, tensor, row_start, row_len, col_start, col_len):
    return {
        "op": op_name,
        "access": access,
        "tensor": tensor_name,
        "layout": "matrix_tile",
        "row_range": [int(row_start), int(row_start + row_len)],
        "col_range": [int(col_start), int(col_start + col_len)],
        "first_addr": int(
            tensor.data_ptr()
            + (
                row_start * tensor.stride(0) + col_start * tensor.stride(1)
            ) * tensor.element_size()
        ),
        "rows": int(row_len),
        "cols": int(col_len),
        "row_stride_bytes": int(tensor.stride(0) * tensor.element_size()),
        "col_stride_bytes": int(tensor.stride(1) * tensor.element_size()),
        "element_size": int(tensor.element_size()),
        "dtype": str(tensor.dtype),
    }


class TraceWriter:
    def __init__(self, path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("w", encoding="utf-8")
        self.event_count = 0

    def write(self, event):
        self.event_count += 1
        payload = {"step": self.event_count, **event}
        self._f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def close(self):
        self._f.close()


def emit_linear_trace(writer, op_name, input_tensor, weight_tensor, output_tensor, output_tile, reduction_tile):
    output_dim = output_tensor.shape[0]
    reduction_dim = input_tensor.shape[0]

    for out_start in range(0, output_dim, output_tile):
        out_len = min(output_tile, output_dim - out_start)
        for k_start in range(0, reduction_dim, reduction_tile):
            k_len = min(reduction_tile, reduction_dim - k_start)
            writer.write(vector_event(op_name, "read", f"{op_name}.input", input_tensor, k_start, k_len))
            writer.write(
                matrix_tile_event(
                    op_name,
                    "read",
                    f"{op_name}.weight",
                    weight_tensor,
                    out_start,
                    out_len,
                    k_start,
                    k_len,
                )
            )
        writer.write(vector_event(op_name, "write", f"{op_name}.output", output_tensor, out_start, out_len))


def emit_elementwise_trace(writer, op_name, gate_tensor, up_tensor, output_tensor, vector_tile):
    hidden_dim = output_tensor.shape[0]
    for start in range(0, hidden_dim, vector_tile):
        tile_len = min(vector_tile, hidden_dim - start)
        writer.write(vector_event(op_name, "read", f"{op_name}.gate_linear", gate_tensor, start, tile_len))
        writer.write(vector_event(op_name, "read", f"{op_name}.up_linear", up_tensor, start, tile_len))
        writer.write(vector_event(op_name, "write", f"{op_name}.output", output_tensor, start, tile_len))


def write_meta(meta_path, payload):
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def capture_dense_ffn_trace(args):
    if args.token_index <= 0:
        raise ValueError("--token-index must be >= 1.")
    if args.token_phase == "decode" and args.max_new_tokens < args.token_index:
        raise ValueError("--max-new-tokens must cover the selected decode token.")
    if args.output_tile <= 0 or args.reduction_tile <= 0 or args.vector_tile <= 0:
        raise ValueError("All tile sizes must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "torch.cuda.is_available() is False. This script needs a visible CUDA device because "
            "it exports FFN traces from real GPU tensor addresses."
        )

    model, tokenizer = load_trace_model(args.model_id, args.device_map, args.quantization)
    num_layers = len(model.model.layers)
    if args.layer < 0 or args.layer >= num_layers:
        raise ValueError(f"--layer must be in [0, {num_layers - 1}], got {args.layer}.")

    state = make_capture_state(args.token_phase, args.token_index)
    target_mlp = model.model.layers[args.layer].mlp

    def mlp_pre_hook(_module, inputs):
        state["hook_call_count"] += 1
        hidden = inputs[0]
        local_index = resolve_local_index(state, token_count_of_hidden(hidden))
        if local_index is not None:
            capture_vector_if_needed(state, "ffn_input", hidden)

    def gate_hook(_module, _inputs, output):
        capture_vector_if_needed(state, "gate_linear", output)

    def up_hook(_module, _inputs, output):
        capture_vector_if_needed(state, "up_linear", output)

    def down_pre_hook(_module, inputs):
        capture_vector_if_needed(state, "intermediate", inputs[0])

    def down_hook(_module, _inputs, output):
        capture_vector_if_needed(state, "ffn_output", output)

    handles = [
        target_mlp.register_forward_pre_hook(mlp_pre_hook),
        target_mlp.gate_proj.register_forward_hook(gate_hook),
        target_mlp.up_proj.register_forward_hook(up_hook),
        target_mlp.down_proj.register_forward_pre_hook(down_pre_hook),
        target_mlp.down_proj.register_forward_hook(down_hook),
    ]

    generated_text = None
    try:
        inputs = tokenizer(args.prompt, return_tensors="pt")
        runtime_device = _get_runtime_device(model)
        target_device = target_mlp.gate_proj.weight.device
        if target_device.type != "cuda":
            raise RuntimeError(
                f"Target layer {args.layer} is on device={target_device}, but a GPU trace requires CUDA. "
                f"Try rerunning with --device-map cuda if your GPU has enough memory."
            )
        inputs = {key: value.to(runtime_device) for key, value in inputs.items()}
        state["prompt_token_count"] = int(inputs["input_ids"].shape[-1])

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    finally:
        for handle in handles:
            handle.remove()

    required = ["ffn_input", "gate_linear", "up_linear", "intermediate", "ffn_output"]
    missing = [name for name in required if state[name] is None]
    if missing:
        raise RuntimeError(
            f"Did not capture all FFN tensors for phase={args.token_phase} token={args.token_index}. "
            f"Missing={missing}, seen_prefill={state['prefill_seen']}, seen_decode={state['decode_seen']}."
        )

    tensors = {
        "ffn_input": state["ffn_input"],
        "gate_linear": state["gate_linear"],
        "up_linear": state["up_linear"],
        "intermediate": state["intermediate"],
        "ffn_output": state["ffn_output"],
        "gate_weight": target_mlp.gate_proj.weight.detach(),
        "up_weight": target_mlp.up_proj.weight.detach(),
        "down_weight": target_mlp.down_proj.weight.detach(),
    }

    for name, tensor in tensors.items():
        ensure_cuda_tensor(name, tensor)

    prefix = f"layer_{args.layer}_{state['captured_phase']}_token_{state['captured_index']}"
    meta_path = args.output_dir / f"{prefix}_meta.json"
    trace_path = args.output_dir / f"{prefix}_trace.jsonl"

    writer = TraceWriter(trace_path)
    try:
        emit_linear_trace(
            writer,
            "gate_proj",
            tensors["ffn_input"],
            tensors["gate_weight"],
            tensors["gate_linear"],
            args.output_tile,
            args.reduction_tile,
        )
        emit_linear_trace(
            writer,
            "up_proj",
            tensors["ffn_input"],
            tensors["up_weight"],
            tensors["up_linear"],
            args.output_tile,
            args.reduction_tile,
        )
        emit_elementwise_trace(
            writer,
            "silu_mul",
            tensors["gate_linear"],
            tensors["up_linear"],
            tensors["intermediate"],
            args.vector_tile,
        )
        emit_linear_trace(
            writer,
            "down_proj",
            tensors["intermediate"],
            tensors["down_weight"],
            tensors["ffn_output"],
            args.output_tile,
            args.reduction_tile,
        )
    finally:
        writer.close()

    meta = {
        "model_id": args.model_id,
        "device_map_mode": args.device_map,
        "quantization_mode": args.quantization,
        "layer": int(args.layer),
        "prompt": args.prompt,
        "generated_text": generated_text,
        "token_phase": state["captured_phase"],
        "token_index": int(state["captured_index"]),
        "capture_source": state["captured_source"],
        "prompt_token_count": int(state["prompt_token_count"]),
        "prefill_seen": int(state["prefill_seen"]),
        "decode_seen": int(state["decode_seen"]),
        "hook_call_count": int(state["hook_call_count"]),
        "tile_config": {
            "output_tile": int(args.output_tile),
            "reduction_tile": int(args.reduction_tile),
            "vector_tile": int(args.vector_tile),
        },
        "trace_granularity": "logical_tile_sequence",
        "trace_semantics": (
            "This is a logical FFN memory-access sequence derived from the captured CUDA tensor "
            "addresses and a deterministic tile traversal order, not a hardware profiler dump."
        ),
        "kernel_order": ["gate_proj", "up_proj", "silu_mul", "down_proj"],
        "tensors": {name: tensor_meta(name, tensor) for name, tensor in tensors.items()},
        "trace_path": str(trace_path),
        "trace_event_count": int(writer.event_count),
    }
    write_meta(meta_path, meta)

    return meta_path, trace_path, writer.event_count


def main():
    args = parse_args()
    meta_path, trace_path, event_count = capture_dense_ffn_trace(args)
    print(f"saved meta to: {meta_path}")
    print(f"saved trace to: {trace_path}")
    print(f"trace events: {event_count}")


if __name__ == "__main__":
    main()
