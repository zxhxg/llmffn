import argparse
from pathlib import Path

import torch

from common import (
    configure_preferred_blas_library,
    default_capture_path,
    ensure_parent_dir,
    get_runtime_device,
    get_target_mlp,
    last_token_vector,
    load_model_and_tokenizer,
    positive_int,
    resolve_default_model_id,
)


DEFAULT_PROMPT = "Explain briefly what the FFN layer does in a transformer."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture the target layer FFN input that corresponds to the first generated token. "
            "This workflow treats the first generated token as the prefill last prompt token."
        ),
    )
    parser.add_argument("--model-id", type=str, default=resolve_default_model_id())
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=1,
        help=(
            "Capture batch size. With batch size 1, preserve the historical last-token "
            "1D capture. With batch size >1, run real batched prefill and capture the "
            "full target MLP input tensor [batch_size, prompt_tokens, hidden_size]."
        ),
    )
    parser.add_argument(
        "--device-map",
        choices=["cuda", "auto"],
        default="cuda",
        help="Model placement strategy. Default forces the whole model onto cuda:0.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Capture output path (.pt). Defaults to scripts/cutracer_ffn_trace/output/captures/...",
    )
    parser.add_argument(
        "--preferred-blas",
        choices=["default", "cublas", "cublaslt"],
        default="cublas",
        help="Preferred CUDA BLAS backend requested through torch.backends.cuda.preferred_blas_library.",
    )
    return parser.parse_args()


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output is None:
        return default_capture_path(args.layer)
    if args.output.suffix:
        return args.output
    return args.output / default_capture_path(args.layer).name


def capture_first_generated_ffn_input(args: argparse.Namespace) -> Path:
    configure_preferred_blas_library(args.preferred_blas)
    model, tokenizer = load_model_and_tokenizer(args.model_id, args.device_map)
    target_mlp = get_target_mlp(model, args.layer)

    state: dict[str, torch.Tensor | None] = {"ffn_input": None}

    def mlp_pre_hook(_module, inputs):
        if state["ffn_input"] is not None:
            return
        hidden = inputs[0].detach()
        if args.batch_size == 1:
            vector = last_token_vector(hidden)
            if vector.dim() != 1:
                raise RuntimeError(
                    f"Expected captured FFN input to be 1D after flattening, got {tuple(vector.shape)}."
                )
            state["ffn_input"] = vector.to(device="cpu", dtype=vector.dtype).clone()
            return

        if hidden.dim() != 3:
            raise RuntimeError(
                "Expected batched prefill FFN input to have shape "
                f"[batch_size, prompt_tokens, hidden_size], got {tuple(hidden.shape)}."
            )
        if int(hidden.shape[0]) != args.batch_size:
            raise RuntimeError(
                f"Expected captured batch dimension {args.batch_size}, got {int(hidden.shape[0])}."
            )
        if hidden.shape[-1] <= 0:
            raise RuntimeError(f"Unexpected batched FFN input shape: {tuple(hidden.shape)}")
        state["ffn_input"] = hidden.to(device="cpu", dtype=hidden.dtype).contiguous().clone()

    handle = target_mlp.register_forward_pre_hook(mlp_pre_hook)
    try:
        tokenizer_input = args.prompt if args.batch_size == 1 else [args.prompt] * args.batch_size
        inputs = tokenizer(
            tokenizer_input,
            return_tensors="pt",
            padding=True if args.batch_size > 1 else False,
        )
        runtime_device = get_runtime_device(model)
        inputs = {key: value.to(runtime_device) for key, value in inputs.items()}
        actual_batch_size = int(inputs["input_ids"].shape[0])
        if actual_batch_size != args.batch_size:
            raise RuntimeError(f"Tokenizer produced batch size {actual_batch_size}, expected {args.batch_size}.")
        prompt_token_count = int(inputs["input_ids"].shape[-1])

        with torch.no_grad():
            _ = model(**inputs, use_cache=True, return_dict=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    finally:
        handle.remove()

    if state["ffn_input"] is None:
        raise RuntimeError(
            "The target layer MLP hook never captured an FFN input during prompt prefill."
        )

    captured = state["ffn_input"]
    assert captured is not None
    token_semantics = (
        "first_generated_token_from_prefill_last_prompt_token"
        if args.batch_size == 1
        else "batched_prefill_full_prompt_tokens"
    )
    output_path = ensure_parent_dir(resolve_output_path(args))
    payload = {
        "ffn_input": captured,
        "layer": int(args.layer),
        "prompt": args.prompt,
        "batch_size": int(args.batch_size),
        "prompt_token_count": prompt_token_count,
        "model_id": args.model_id,
        "dtype": str(captured.dtype),
        "device": str(next(target_mlp.parameters()).device),
        "hidden_size": int(captured.shape[-1]),
        "ffn_input_shape": tuple(int(dim) for dim in captured.shape),
        "token_semantics": token_semantics,
    }
    torch.save(payload, output_path)
    return output_path


def main() -> None:
    args = parse_args()
    output_path = capture_first_generated_ffn_input(args)
    payload = torch.load(output_path, map_location="cpu", weights_only=True)
    print(f"saved capture to: {output_path}")
    print(f"layer: {payload['layer']}")
    print(f"batch_size: {payload['batch_size']}")
    print(f"prompt_token_count: {payload['prompt_token_count']}")
    print(f"ffn_input_shape: {tuple(payload['ffn_input'].shape)}")
    print(f"token_semantics: {payload['token_semantics']}")


if __name__ == "__main__":
    main()
