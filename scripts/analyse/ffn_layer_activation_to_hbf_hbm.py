import argparse
import json
import sys
from pathlib import Path

import torch


THIS_DIR = Path(__file__).resolve().parent
STATISTIC_DIR = THIS_DIR.parent / "statistic"
if str(STATISTIC_DIR) not in sys.path:
    sys.path.insert(0, str(STATISTIC_DIR))

from hbf_hbm_ffn_simulator import (  # noqa: E402
    FFNConfig,
    MemoryConfig,
    pretty_print,
    results_to_jsonable,
    simulate,
)
import ffn_choose_single_layer_profile as profile_mod  # noqa: E402


DEFAULT_HBM_BANDWIDTH_GBPS = 2500.0
DEFAULT_HBM_STARTUP_NS = 50.0
DEFAULT_HBM_SMALL_RUN_PENALTY = 0.65
DEFAULT_HBF_BANDWIDTH_GBPS = 350.0
DEFAULT_HBF_PAGE_SIZE = 4096
DEFAULT_HBF_STARTUP_NS = 300.0
DEFAULT_HBF_SMALL_RUN_PENALTY = 0.45
DEFAULT_OUTPUT_DIR = THIS_DIR / "ffn_layer_hbf_hbm_analysis_output"
DEFAULT_TOKEN_PHASE = "decode"
DEFAULT_TOKEN_INDEX = 1
ACTIVE_IDS_SAMPLE_SIZE = 64


def parse_args():
    parser = argparse.ArgumentParser(
        description="Capture one token's FFN activations for a chosen layer and run HBM/HBF simulation.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=profile_mod.DEFAULT_TARGET_LAYER,
        help="Layer index to analyse.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Override the default prompt used during generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override the default generation length.",
    )
    parser.add_argument(
        "--token-phase",
        choices=["prefill", "decode"],
        default=DEFAULT_TOKEN_PHASE,
        help="Whether to capture a prompt token or a generated token.",
    )
    parser.add_argument(
        "--token-index",
        type=int,
        default=DEFAULT_TOKEN_INDEX,
        help="1-based index within the chosen phase. Default captures the first decode token.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the JSON analysis output.",
    )
    parser.add_argument("--hbm-bandwidth-gbps", type=float, default=DEFAULT_HBM_BANDWIDTH_GBPS)
    parser.add_argument("--hbm-startup-ns", type=float, default=DEFAULT_HBM_STARTUP_NS)
    parser.add_argument("--hbm-small-run-penalty", type=float, default=DEFAULT_HBM_SMALL_RUN_PENALTY)
    parser.add_argument("--hbf-bandwidth-gbps", type=float, default=DEFAULT_HBF_BANDWIDTH_GBPS)
    parser.add_argument("--hbf-page-size", type=int, default=DEFAULT_HBF_PAGE_SIZE)
    parser.add_argument("--hbf-startup-ns", type=float, default=DEFAULT_HBF_STARTUP_NS)
    parser.add_argument("--hbf-small-run-penalty", type=float, default=DEFAULT_HBF_SMALL_RUN_PENALTY)
    return parser.parse_args()


def memory_configs_from_args(args):
    hbm = MemoryConfig(
        name="HBM",
        bandwidth_GBps=args.hbm_bandwidth_gbps,
        page_size_B=0,
        startup_ns=args.hbm_startup_ns,
        small_run_bw_penalty=args.hbm_small_run_penalty,
    )
    hbf = MemoryConfig(
        name="HBF",
        bandwidth_GBps=args.hbf_bandwidth_gbps,
        page_size_B=args.hbf_page_size,
        startup_ns=args.hbf_startup_ns,
        small_run_bw_penalty=args.hbf_small_run_penalty,
    )
    return hbm, hbf


def infer_ffn_config(model, layer_idx):
    mlp = model.model.layers[layer_idx].mlp
    return FFNConfig(
        d=int(mlp.gate_proj.in_features),
        m=int(mlp.gate_proj.out_features),
        bytes_per_weight=float(mlp.gate_proj.weight.element_size()),
    )


def config_to_dict(cfg):
    return {
        "name": cfg.name,
        "bandwidth_GBps": float(cfg.bandwidth_GBps),
        "page_size_B": int(cfg.page_size_B),
        "startup_ns": float(cfg.startup_ns),
        "small_run_bw_penalty": float(cfg.small_run_bw_penalty),
    }


def ffn_config_to_dict(cfg):
    return {
        "d": int(cfg.d),
        "m": int(cfg.m),
        "bytes_per_weight": float(cfg.bytes_per_weight),
    }


def write_json_output(output_dir, payload, layer_idx, phase, token_index):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"layer_{layer_idx}_{phase}_token_{token_index}_hbf_hbm_analysis.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return output_path


def token_capture_state(token_phase, token_index):
    return {
        "token_phase": token_phase,
        "token_index": token_index,
        "prompt_token_count": 0,
        "prefill_seen": 0,
        "decode_seen": 0,
        "selected_idx": None,
        "selected_score": None,
        "selected_count": None,
        "captured_phase": None,
        "captured_index": None,
        "hook_call_count": 0,
    }


def capture_one_token(layer_idx, prompt_override, max_new_tokens_override, token_phase, token_index):
    if token_index <= 0:
        raise ValueError("--token-index must be >= 1.")

    original_prompt = profile_mod.PROMPT
    original_max_new_tokens = profile_mod.MAX_NEW_TOKENS
    if prompt_override is not None:
        profile_mod.PROMPT = prompt_override
    if max_new_tokens_override is not None:
        profile_mod.MAX_NEW_TOKENS = max_new_tokens_override

    try:
        model, tokenizer = profile_mod.load_fp16_model()
        num_layers = len(model.model.layers)
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(f"--layer must be in [0, {num_layers - 1}], got {layer_idx}.")

        runtime_device = profile_mod._get_runtime_device(model)
        state = token_capture_state(token_phase, token_index)
        target_mlp = model.model.layers[layer_idx].mlp

        def hook(_module, inputs):
            state["hook_call_count"] += 1
            intermediate = inputs[0]
            flat = intermediate.detach().to(torch.float32).reshape(-1, intermediate.shape[-1])
            token_count = flat.shape[0]

            if state["prompt_token_count"] == 0:
                raise RuntimeError("prompt_token_count must be set before generation starts.")

            remaining_prefill = max(state["prompt_token_count"] - state["prefill_seen"], 0)
            prefill_count = min(token_count, remaining_prefill)
            decode_count = token_count - prefill_count

            if prefill_count > 0:
                start = state["prefill_seen"] + 1
                end = state["prefill_seen"] + prefill_count
                if state["token_phase"] == "prefill" and state["selected_idx"] is None:
                    if start <= state["token_index"] <= end:
                        local_pos = state["token_index"] - start
                        score = flat[local_pos : local_pos + 1].abs()
                        idx, score_values, count = profile_mod.select_activated_neurons(
                            score,
                            profile_mod.CONTRIBUTION_RATIO,
                        )
                        state["selected_idx"] = idx
                        state["selected_score"] = score_values
                        state["selected_count"] = count
                        state["captured_phase"] = "prefill"
                        state["captured_index"] = state["token_index"]
                state["prefill_seen"] += prefill_count

            if decode_count > 0:
                start = state["decode_seen"] + 1
                end = state["decode_seen"] + decode_count
                if state["token_phase"] == "decode" and state["selected_idx"] is None:
                    if start <= state["token_index"] <= end:
                        local_pos = prefill_count + (state["token_index"] - start)
                        score = flat[local_pos : local_pos + 1].abs()
                        idx, score_values, count = profile_mod.select_activated_neurons(
                            score,
                            profile_mod.CONTRIBUTION_RATIO,
                        )
                        state["selected_idx"] = idx
                        state["selected_score"] = score_values
                        state["selected_count"] = count
                        state["captured_phase"] = "decode"
                        state["captured_index"] = state["token_index"]
                state["decode_seen"] += decode_count

        handle = target_mlp.down_proj.register_forward_pre_hook(hook)

        try:
            inputs = profile_mod.prepare_inputs(tokenizer, runtime_device)
            state["prompt_token_count"] = int(inputs["input_ids"].shape[-1])
            profile_mod.set_seed(profile_mod.SEED)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=profile_mod.MAX_NEW_TOKENS,
                    do_sample=profile_mod.DO_SAMPLE,
                    temperature=profile_mod.TEMPERATURE,
                    top_p=profile_mod.TOP_P,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        finally:
            handle.remove()

        if state["selected_idx"] is None or state["selected_count"] is None:
            raise RuntimeError(
                f"Did not capture token phase={token_phase} index={token_index}. "
                f"Seen prefill={state['prefill_seen']} decode={state['decode_seen']}."
            )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        active_ids = state["selected_idx"].to(torch.int64).tolist()
        return {
            "model": model,
            "tokenizer": tokenizer,
            "active_ids": active_ids,
            "selected_idx": state["selected_idx"],
            "selected_score": state["selected_score"],
            "selected_count": state["selected_count"],
            "captured_phase": state["captured_phase"],
            "captured_index": state["captured_index"],
            "hook_call_count": state["hook_call_count"],
            "generated_text": generated_text,
            "prompt_token_count": state["prompt_token_count"],
            "max_new_tokens": profile_mod.MAX_NEW_TOKENS,
        }
    finally:
        profile_mod.PROMPT = original_prompt
        profile_mod.MAX_NEW_TOKENS = original_max_new_tokens


def main():
    args = parse_args()
    hbm, hbf = memory_configs_from_args(args)

    capture = capture_one_token(
        args.layer,
        args.prompt,
        args.max_new_tokens,
        args.token_phase,
        args.token_index,
    )
    active_ids = capture["active_ids"]
    ffn_cfg = infer_ffn_config(capture["model"], args.layer)
    simulation_results = simulate(ffn_cfg, active_ids, hbm, hbf)

    print(f"target_layer={args.layer}")
    print(f"captured_phase={capture['captured_phase']}")
    print(f"captured_index={capture['captured_index']}")
    print(f"active_ids_count={len(active_ids)}")
    print(f"ffn_config={ffn_config_to_dict(ffn_cfg)}")
    pretty_print(simulation_results)

    json_payload = {
        "target_layer": int(args.layer),
        "captured_phase": capture["captured_phase"],
        "captured_index": int(capture["captured_index"]),
        "active_ids_count": int(len(active_ids)),
        "active_ids_sample": active_ids[:ACTIVE_IDS_SAMPLE_SIZE],
        "selected_count_tensor": capture["selected_count"].to(torch.int64).tolist(),
        "selected_score_sample": [
            float(x) for x in capture["selected_score"][:ACTIVE_IDS_SAMPLE_SIZE].tolist()
        ],
        "ffn_config": ffn_config_to_dict(ffn_cfg),
        "hbm_config": config_to_dict(hbm),
        "hbf_config": config_to_dict(hbf),
        "simulation_results": results_to_jsonable(simulation_results),
        "prompt": args.prompt if args.prompt is not None else profile_mod.PROMPT,
        "prompt_token_count": int(capture["prompt_token_count"]),
        "max_new_tokens": int(capture["max_new_tokens"]),
        "hook_call_count": int(capture["hook_call_count"]),
        "generated_text": capture["generated_text"],
    }
    output_path = write_json_output(
        args.output_dir,
        json_payload,
        args.layer,
        capture["captured_phase"],
        capture["captured_index"],
    )
    print(f"saved analysis json to: {output_path}")


if __name__ == "__main__":
    main()
