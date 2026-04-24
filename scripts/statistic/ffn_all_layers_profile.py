import time
from pathlib import Path

import numpy as np
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
OUTPUT_DIR = Path(__file__).resolve().with_name("ffn_all_layers_profile_output")
TEMP_DIR = OUTPUT_DIR / "tmp_dense_memmaps"
LOG_PATH = OUTPUT_DIR / "ffn_all_layers_profile.log"
DIAGNOSTIC_LOG_EVERY_TOKEN = True
ADJACENCY_DTYPE = np.float32

_LOG_FILE = None


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


def cleanup_output_dir(output_dir, temp_dir):
    patterns = [
        "activation_counts_layer_*.pt",
        "adjacency_layer_*.pt",
        "ffn_all_layers_profile.log",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for path in output_dir.glob(pattern):
            path.unlink()

    if temp_dir.exists():
        for path in temp_dir.glob("*.mmap"):
            path.unlink()
    temp_dir.mkdir(parents=True, exist_ok=True)


def init_progress_state():
    return {
        "prompt_token_count": 0,
        "prefill_tokens_seen": 0,
        "decoding_tokens_seen": 0,
        "max_new_tokens": 0,
    }


def init_layer_state(layer_idx, intermediate_size, temp_dir):
    mmap_path = temp_dir / f"adjacency_layer_{layer_idx}.mmap"
    adjacency = np.memmap(
        mmap_path,
        mode="w+",
        dtype=ADJACENCY_DTYPE,
        shape=(intermediate_size, intermediate_size),
    )
    adjacency[:] = 0

    return {
        "layer_idx": layer_idx,
        "activation_counts": torch.zeros(intermediate_size, dtype=torch.int32),
        "adjacency": adjacency,
        "adjacency_mmap_path": mmap_path,
        "hook_call_count": 0,
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

        chosen_idx = selected_idx[offset : offset + count].numpy().astype(np.int64, copy=False)
        chosen_score = selected_score[offset : offset + count].numpy().astype(np.float32, copy=False)
        offset += count

        outer = np.outer(chosen_score, chosen_score)
        np.fill_diagonal(outer, 0.0)
        adjacency[np.ix_(chosen_idx, chosen_idx)] += outer
        pair_updates += count * (count - 1) // 2

    return pair_updates


def top_activation_summary(counts, top_k=TOP_K_SUMMARY):
    top_values, top_indices = torch.topk(counts, k=min(top_k, counts.numel()))
    return [
        {"index": int(idx), "count": int(value)}
        for value, idx in zip(top_values.tolist(), top_indices.tolist())
        if value > 0
    ]


def prepare_inputs(tokenizer, runtime_device):
    inputs = tokenizer(PROMPT, return_tensors="pt")
    return {key: value.to(runtime_device) for key, value in inputs.items()}


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def report_progress(progress_state, token_count):
    remaining_prefill = max(
        progress_state["prompt_token_count"] - progress_state["prefill_tokens_seen"],
        0,
    )
    prefill_count = min(token_count, remaining_prefill)
    decoding_count = token_count - prefill_count

    for _ in range(prefill_count):
        progress_state["prefill_tokens_seen"] += 1
        log_message(
            f"[progress] prefilling token "
            f"{progress_state['prefill_tokens_seen']}/{progress_state['prompt_token_count']}"
        )

    for _ in range(decoding_count):
        progress_state["decoding_tokens_seen"] += 1
        log_message(
            f"[progress] decoding token "
            f"{progress_state['decoding_tokens_seen']}/{progress_state['max_new_tokens']}"
        )


def log_progress_diagnostics(progress_state, token_count, token_layer_summaries, elapsed_ms):
    summary_str = ", ".join(
        [
            (
                f"layer={item['layer_idx']}:selected_avg={item['selected_avg']:.1f},"
                f"selected_max={item['selected_max']},pairs={item['estimated_pairs']},"
                f"edge_ms={item['edge_elapsed_ms']:.1f}"
            )
            for item in token_layer_summaries
        ]
    )
    log_message(
        "[diag] "
        f"token_count={token_count}, "
        f"prefill_seen={progress_state['prefill_tokens_seen']}, "
        f"decode_seen={progress_state['decoding_tokens_seen']}, "
        f"hook_batch_elapsed_ms={elapsed_ms:.1f}, "
        f"layer_samples=[{summary_str}]"
    )


def make_down_proj_pre_hook(layer_state, progress_state, layer_summaries):
    def hook(_module, inputs):
        layer_state["hook_call_count"] += 1
        intermediate = inputs[0]
        token_count = intermediate.reshape(-1, intermediate.shape[-1]).shape[0]

        if layer_state["layer_idx"] == 0:
            report_progress(progress_state, token_count)
            layer_summaries.clear()
            layer_summaries.append({"token_count": token_count, "start_time": time.perf_counter()})

        score = intermediate.detach().to(torch.float32).abs()
        selected_idx, selected_score, selected_count = select_activated_neurons(
            score,
            CONTRIBUTION_RATIO,
        )
        update_activation_counts(layer_state["activation_counts"], selected_idx)

        edge_start = time.perf_counter()
        pair_updates = update_dense_adjacency(
            layer_state["adjacency"],
            selected_idx,
            selected_score,
            selected_count,
        )
        edge_elapsed_ms = (time.perf_counter() - edge_start) * 1000
        layer_state["pair_updates_total"] += pair_updates

        if DIAGNOSTIC_LOG_EVERY_TOKEN:
            selected_count_i64 = selected_count.to(torch.int64)
            selected_avg = (
                float(selected_count_i64.to(torch.float32).mean().item())
                if selected_count_i64.numel() > 0
                else 0.0
            )
            selected_max = int(selected_count_i64.max().item()) if selected_count_i64.numel() > 0 else 0
            estimated_pairs = (
                int(((selected_count_i64 * (selected_count_i64 - 1)) // 2).sum().item())
                if selected_count_i64.numel() > 0
                else 0
            )
            layer_summaries.append(
                {
                    "layer_idx": layer_state["layer_idx"],
                    "selected_avg": selected_avg,
                    "selected_max": selected_max,
                    "estimated_pairs": estimated_pairs,
                    "edge_elapsed_ms": edge_elapsed_ms,
                }
            )

        if DIAGNOSTIC_LOG_EVERY_TOKEN and layer_state["layer_idx"] == progress_state.get("last_layer_idx", -1):
            elapsed_ms = (time.perf_counter() - layer_summaries[0]["start_time"]) * 1000
            log_progress_diagnostics(
                progress_state,
                layer_summaries[0]["token_count"],
                layer_summaries[1:],
                elapsed_ms,
            )

    return hook


def register_all_layer_hooks(model, progress_state, layer_states):
    handles = []
    layer_summaries = []
    progress_state["last_layer_idx"] = len(layer_states) - 1
    for layer_state in layer_states:
        layer = model.model.layers[layer_state["layer_idx"]]
        handle = layer.mlp.down_proj.register_forward_pre_hook(
            make_down_proj_pre_hook(layer_state, progress_state, layer_summaries)
        )
        handles.append(handle)
    return handles


def run_profile_pass(model, tokenizer, runtime_device, progress_state, layer_states):
    handles = register_all_layer_hooks(model, progress_state, layer_states)

    try:
        inputs = prepare_inputs(tokenizer, runtime_device)
        progress_state["prompt_token_count"] = int(inputs["input_ids"].shape[-1])
        progress_state["max_new_tokens"] = MAX_NEW_TOKENS
        set_seed(SEED)
        log_message(
            f"[diag] starting generate: layers={len(layer_states)}, "
            f"prompt_tokens={progress_state['prompt_token_count']}, max_new_tokens={MAX_NEW_TOKENS}"
        )
        total_dense_bytes = sum(
            state["adjacency"].size * state["adjacency"].dtype.itemsize for state in layer_states
        )
        log_message(
            f"[diag] total dense adjacency backing store={format_bytes(total_dense_bytes)}, "
            f"temp_dir={TEMP_DIR}"
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
            f"prefill_seen={progress_state['prefill_tokens_seen']}, "
            f"decode_seen={progress_state['decoding_tokens_seen']}"
        )
    finally:
        for handle in handles:
            handle.remove()

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def save_layer_outputs(output_dir, layer_state):
    layer_idx = layer_state["layer_idx"]
    counts_path = output_dir / f"activation_counts_layer_{layer_idx}.pt"
    adjacency_path = output_dir / f"adjacency_layer_{layer_idx}.pt"

    adjacency = layer_state["adjacency"]
    adjacency.flush()
    log_message(
        f"[diag] saving layer {layer_idx}: "
        f"shape={adjacency.shape}, dtype={adjacency.dtype}, "
        f"size={format_bytes(adjacency.size * adjacency.dtype.itemsize)}"
    )

    save_counts_start = time.perf_counter()
    torch.save(layer_state["activation_counts"], counts_path)
    save_counts_elapsed = time.perf_counter() - save_counts_start
    log_message(
        f"[diag] saved activation counts in {save_counts_elapsed:.2f}s -> {counts_path.name}"
    )

    save_adj_start = time.perf_counter()
    adjacency_tensor = torch.from_numpy(adjacency)
    torch.save(adjacency_tensor, adjacency_path)
    save_adj_elapsed = time.perf_counter() - save_adj_start
    log_message(
        f"[diag] saved dense adjacency in {save_adj_elapsed:.2f}s -> {adjacency_path.name}"
    )
    del adjacency_tensor

    if hasattr(adjacency, "_mmap") and adjacency._mmap is not None:
        adjacency._mmap.close()
    if layer_state["adjacency_mmap_path"].exists():
        layer_state["adjacency_mmap_path"].unlink()

    return counts_path, adjacency_path


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_output_dir(OUTPUT_DIR, TEMP_DIR)
    log_path = open_log_file(LOG_PATH)
    log_message(f"[diag] writing logs to: {log_path}")

    try:
        model, tokenizer = load_fp16_model()
        runtime_device = _get_runtime_device(model)
        num_layers = len(model.model.layers)

        layer_states = []
        for layer_idx in range(num_layers):
            intermediate_size = model.model.layers[layer_idx].mlp.intermediate_size
            layer_states.append(init_layer_state(layer_idx, intermediate_size, TEMP_DIR))

        progress_state = init_progress_state()
        generated_text = run_profile_pass(model, tokenizer, runtime_device, progress_state, layer_states)

        for layer_state in layer_states:
            counts_path, adjacency_path = save_layer_outputs(OUTPUT_DIR, layer_state)
            active_neuron_count = int((layer_state["activation_counts"] > 0).sum().item())
            log_message(
                f"target_layer={layer_state['layer_idx']}, "
                f"active_neurons={active_neuron_count}, "
                f"pair_updates_total={layer_state['pair_updates_total']}, "
                f"activation_counts_path={counts_path.name}, "
                f"adjacency_path={adjacency_path.name}, "
                f"top_activated_neurons={top_activation_summary(layer_state['activation_counts'])}"
            )

        log_message(generated_text)
        log_message(f"saved profiling outputs to: {OUTPUT_DIR}")
    finally:
        close_log_file()


if __name__ == "__main__":
    main()
