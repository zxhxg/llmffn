import argparse
import time
from pathlib import Path

import torch

from run_fp16 import _get_runtime_device, load_fp16_model


PROMPT = "Explain briefly what the FFN layer does in a transformer."
CONTRIBUTION_RATIO = 0.8
MAX_NEW_TOKENS = 32
DO_SAMPLE = True
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K_SUMMARY = 5
SEED = 1234
DEFAULT_TARGET_LAYER = 0
OUTPUT_DIR = Path(__file__).resolve().with_name("ffn_choose_single_layer_profile_output")
DIAGNOSTIC_LOG_EVERY_TOKEN = True
ADJACENCY_DTYPE = torch.float32

_LOG_FILE = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile FFN activations for one chosen layer in a single generate pass.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=DEFAULT_TARGET_LAYER,
        help="Layer index to profile.",
    )
    return parser.parse_args()


def open_log_file(log_path):
    global _LOG_FILE
    if _LOG_FILE is not None and not _LOG_FILE.closed:
        _LOG_FILE.close()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = log_path.open("w", encoding="utf-8", buffering=1)
    return log_path


def close_log_file():
    global _LOG_FILE
    if _LOG_FILE is not None and not _LOG_FILE.closed:
        _LOG_FILE.close()
    _LOG_FILE = None


def log_message(message):
    print(message, flush=True)
    if _LOG_FILE is not None and not _LOG_FILE.closed:
        _LOG_FILE.write(f"{message}\n")
        _LOG_FILE.flush()


def format_bytes(num_bytes):
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{value:.2f}TB"


def log_path_for_layer(output_dir, layer_idx):
    return output_dir / f"ffn_choose_single_layer_profile_layer_{layer_idx}.log"


def init_profile_state(intermediate_size):
    adjacency = torch.zeros((intermediate_size, intermediate_size), dtype=ADJACENCY_DTYPE)
    return {
        "activation_counts": torch.zeros(intermediate_size, dtype=torch.int32),
        "adjacency": adjacency,
        "prompt_token_count": 0,
        "prefill_tokens_seen": 0,
        "decoding_tokens_seen": 0,
        "max_new_tokens": 0,
        "hook_call_count": 0,
        "last_phase": "prefill",
        "pair_updates_total": 0,
    }


def select_activated_neurons(score, contribution_ratio):
    flat_score = score.detach().to(torch.float32).reshape(-1, score.shape[-1])

    if flat_score.numel() == 0:
        empty_idx = torch.empty((0,), dtype=torch.int32)
        empty_score = torch.empty((0,), dtype=torch.float32)
        empty_count = torch.empty((0,), dtype=torch.int32)
        return empty_idx, empty_score, empty_count

    sorted_score, sorted_idx = torch.sort(flat_score, dim=-1, descending=True)
    cumsum = torch.cumsum(sorted_score, dim=-1)
    threshold = contribution_ratio * cumsum[:, -1]

    selected_count = torch.zeros_like(threshold, dtype=torch.int64)
    positive_mask = threshold > 0
    if positive_mask.any():
        positive_cumsum = cumsum[positive_mask]
        positive_threshold = threshold[positive_mask].unsqueeze(-1)
        positive_count = torch.searchsorted(
            positive_cumsum,
            positive_threshold,
            right=False,
        ).squeeze(-1) + 1
        selected_count[positive_mask] = positive_count

    selected_count = torch.clamp(selected_count, max=flat_score.shape[-1])
    position = torch.arange(sorted_idx.shape[-1], device=sorted_idx.device).unsqueeze(0)
    selected_mask = position < selected_count.unsqueeze(-1)

    selected_idx = sorted_idx[selected_mask].to(torch.int32).cpu()
    selected_score = sorted_score[selected_mask].to(torch.float32).cpu()
    selected_count = selected_count.to(torch.int32).cpu()
    return selected_idx, selected_score, selected_count


def update_activation_counts(counts, selected_idx):
    if selected_idx.numel() == 0:
        return

    bincount = torch.bincount(selected_idx.to(torch.int64), minlength=counts.numel())
    counts += bincount.to(torch.int32)


def update_dense_adjacency(adjacency, selected_idx, selected_score, selected_count):
    if selected_idx.numel() == 0:
        return 0

    offset = 0
    pair_updates = 0
    for count in selected_count.tolist():
        if count <= 1:
            offset += count
            continue

        chosen_idx = selected_idx[offset : offset + count].to(torch.int64)
        chosen_score = selected_score[offset : offset + count].to(torch.float32)
        offset += count

        outer = torch.outer(chosen_score, chosen_score)
        outer.fill_diagonal_(0)
        adjacency[chosen_idx[:, None], chosen_idx[None, :]] += outer
        pair_updates += count * (count - 1) // 2

    return pair_updates


def top_activation_summary(counts, top_k=TOP_K_SUMMARY):
    top_values, top_indices = torch.topk(counts, k=min(top_k, counts.numel()))
    return [
        {"index": int(idx), "count": int(value)}
        for value, idx in zip(top_values.tolist(), top_indices.tolist())
        if value > 0
    ]


def top_edge_summary(_adjacency, _top_k=TOP_K_SUMMARY):
    return "omitted in dense adjacency experiment"


def prepare_inputs(tokenizer, runtime_device):
    inputs = tokenizer(PROMPT, return_tensors="pt")
    return {key: value.to(runtime_device) for key, value in inputs.items()}


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def report_progress(state, token_count):
    remaining_prefill = max(state["prompt_token_count"] - state["prefill_tokens_seen"], 0)
    prefill_count = min(token_count, remaining_prefill)
    decoding_count = token_count - prefill_count
    phase = "prefill" if decoding_count == 0 else "decode"

    for _ in range(prefill_count):
        state["prefill_tokens_seen"] += 1
        log_message(
            f"[progress] prefilling token {state['prefill_tokens_seen']}/{state['prompt_token_count']}"
        )

    for _ in range(decoding_count):
        state["decoding_tokens_seen"] += 1
        log_message(
            f"[progress] decoding token {state['decoding_tokens_seen']}/{state['max_new_tokens']}"
        )

    state["last_phase"] = phase


def log_hook_diagnostics(
    state,
    token_count,
    selected_count,
    score_elapsed,
    select_elapsed,
    count_elapsed,
    edge_elapsed,
):
    selected_count_i64 = selected_count.to(torch.int64)
    selected_total = int(selected_count_i64.sum().item())
    selected_max = int(selected_count_i64.max().item()) if selected_count_i64.numel() > 0 else 0
    selected_min = int(selected_count_i64.min().item()) if selected_count_i64.numel() > 0 else 0
    selected_avg = (
        float(selected_count_i64.to(torch.float32).mean().item()) if selected_count_i64.numel() > 0 else 0.0
    )
    estimated_pairs = int(
        ((selected_count_i64 * (selected_count_i64 - 1)) // 2).sum().item()
    ) if selected_count_i64.numel() > 0 else 0

    log_message(
        "[diag] "
        f"hook={state['hook_call_count']}, "
        f"phase={state['last_phase']}, "
        f"token_count={token_count}, "
        f"selected_total={selected_total}, "
        f"selected_min={selected_min}, "
        f"selected_avg={selected_avg:.1f}, "
        f"selected_max={selected_max}, "
        f"estimated_pairs={estimated_pairs}, "
        f"pair_updates_total={state['pair_updates_total']}, "
        f"timing_ms(score={score_elapsed * 1000:.1f}, "
        f"select={select_elapsed * 1000:.1f}, "
        f"counts={count_elapsed * 1000:.1f}, "
        f"edges={edge_elapsed * 1000:.1f})"
    )


def make_down_proj_pre_hook(state):
    def hook(_module, inputs):
        state["hook_call_count"] += 1
        intermediate = inputs[0]
        token_count = intermediate.reshape(-1, intermediate.shape[-1]).shape[0]
        report_progress(state, token_count)

        score_start = time.perf_counter()
        score = intermediate.detach().to(torch.float32).abs()
        score_elapsed = time.perf_counter() - score_start

        select_start = time.perf_counter()
        selected_idx, selected_score, selected_count = select_activated_neurons(
            score,
            CONTRIBUTION_RATIO,
        )
        select_elapsed = time.perf_counter() - select_start

        count_start = time.perf_counter()
        update_activation_counts(state["activation_counts"], selected_idx)
        count_elapsed = time.perf_counter() - count_start

        edge_start = time.perf_counter()
        pair_updates = update_dense_adjacency(
            state["adjacency"],
            selected_idx,
            selected_score,
            selected_count,
        )
        state["pair_updates_total"] += pair_updates
        edge_elapsed = time.perf_counter() - edge_start

        if DIAGNOSTIC_LOG_EVERY_TOKEN:
            log_hook_diagnostics(
                state,
                token_count,
                selected_count,
                score_elapsed,
                select_elapsed,
                count_elapsed,
                edge_elapsed,
            )

    return hook


def run_profile_pass(model, tokenizer, runtime_device, target_layer, state):
    target_mlp = model.model.layers[target_layer].mlp
    handle = target_mlp.down_proj.register_forward_pre_hook(make_down_proj_pre_hook(state))

    try:
        inputs = prepare_inputs(tokenizer, runtime_device)
        state["prompt_token_count"] = int(inputs["input_ids"].shape[-1])
        state["max_new_tokens"] = MAX_NEW_TOKENS
        set_seed(SEED)
        log_message(
            f"[diag] starting generate: target_layer={target_layer}, "
            f"prompt_tokens={state['prompt_token_count']}, max_new_tokens={MAX_NEW_TOKENS}"
        )
        log_message(
            f"[diag] dense adjacency shape={tuple(state['adjacency'].shape)}, "
            f"dtype={state['adjacency'].dtype}, "
            f"size={format_bytes(state['adjacency'].numel() * state['adjacency'].element_size())}"
        )
        generate_start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generate_elapsed = time.perf_counter() - generate_start
        log_message(
            f"[diag] generate finished in {generate_elapsed:.2f}s, "
            f"hooks_seen={state['hook_call_count']}, "
            f"prefill_seen={state['prefill_tokens_seen']}, "
            f"decode_seen={state['decoding_tokens_seen']}, "
            f"pair_updates_total={state['pair_updates_total']}"
        )
    finally:
        handle.remove()

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def cleanup_output_dir(output_dir, target_layer):
    patterns = [
        f"activation_counts_layer_{target_layer}.pt",
        f"adjacency_layer_{target_layer}.pt",
        f"ffn_choose_single_layer_profile_layer_{target_layer}.log",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for path in output_dir.glob(pattern):
            path.unlink()


def save_outputs(output_dir, target_layer, state):
    counts_path = output_dir / f"activation_counts_layer_{target_layer}.pt"
    adjacency_path = output_dir / f"adjacency_layer_{target_layer}.pt"

    adjacency = state["adjacency"]
    log_message(
        f"[diag] saving dense adjacency directly, "
        f"shape={tuple(adjacency.shape)}, dtype={adjacency.dtype}, "
        f"size={format_bytes(adjacency.numel() * adjacency.element_size())}"
    )

    save_counts_start = time.perf_counter()
    torch.save(state["activation_counts"], counts_path)
    save_counts_elapsed = time.perf_counter() - save_counts_start
    log_message(
        f"[diag] saved activation counts in {save_counts_elapsed:.2f}s -> {counts_path.name}"
    )

    save_adj_start = time.perf_counter()
    torch.save(adjacency, adjacency_path)
    save_adj_elapsed = time.perf_counter() - save_adj_start
    log_message(
        f"[diag] saved dense adjacency in {save_adj_elapsed:.2f}s -> {adjacency_path.name}"
    )
    return counts_path, adjacency_path, adjacency


def main():
    args = parse_args()

    model, tokenizer = load_fp16_model()
    num_layers = len(model.model.layers)
    if args.layer < 0 or args.layer >= num_layers:
        raise ValueError(f"--layer must be in [0, {num_layers - 1}], got {args.layer}.")

    runtime_device = _get_runtime_device(model)
    intermediate_size = model.model.layers[args.layer].mlp.intermediate_size
    state = init_profile_state(intermediate_size)

    cleanup_output_dir(OUTPUT_DIR, args.layer)
    log_path = open_log_file(log_path_for_layer(OUTPUT_DIR, args.layer))
    log_message(f"[diag] writing logs to: {log_path}")

    try:
        generated_text = run_profile_pass(model, tokenizer, runtime_device, args.layer, state)
        counts_path, adjacency_path, adjacency = save_outputs(OUTPUT_DIR, args.layer, state)

        active_neuron_count = int((state["activation_counts"] > 0).sum().item())
        adjacency_nnz = int(torch.count_nonzero(adjacency).item())
        log_message(
            f"target_layer={args.layer}, "
            f"active_neurons={active_neuron_count}, "
            f"pair_updates_total={state['pair_updates_total']}, "
            f"adjacency_nnz={adjacency_nnz}, "
            f"activation_counts_path={counts_path.name}, "
            f"adjacency_path={adjacency_path.name}, "
            f"top_activated_neurons={top_activation_summary(state['activation_counts'])}, "
            f"top_edges={top_edge_summary(adjacency)}"
        )
        log_message(generated_text)
        log_message(f"saved profiling outputs to: {OUTPUT_DIR}")
    finally:
        close_log_file()


if __name__ == "__main__":
    main()



