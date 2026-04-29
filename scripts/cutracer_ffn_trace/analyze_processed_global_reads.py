from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_GLOBAL_ADDR_MIN = 1 << 20
DEFAULT_DTYPE_BYTES = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream a postprocessed CUTracer FFN trace and estimate unique global-memory "
            "read addresses. This is intended for runs whose raw trace was deleted and "
            "whose processed events no longer contain SASS."
        )
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Run directory containing processed_ffn_mem_sequence.jsonl.",
    )
    parser.add_argument(
        "--processed-name",
        default="processed_ffn_mem_sequence.jsonl",
        help="Processed JSONL filename inside run_dir.",
    )
    parser.add_argument(
        "--global-addr-min",
        type=int,
        default=DEFAULT_GLOBAL_ADDR_MIN,
        help=(
            "Addresses below this value are treated as shared-memory/local offsets. "
            "Default: 1 MiB."
        ),
    )
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=14336)
    parser.add_argument("--dtype-bytes", type=int, default=DEFAULT_DTYPE_BYTES)
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=8,
        help="Number of example addresses to keep per kernel source.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Defaults to stdout only.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("rb") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                yield line_no, json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Malformed JSON at {path}:{line_no}: {exc}") from exc


def source_name(record: dict[str, Any]) -> str:
    value = record.get("source_trace") or record.get("unmangled_name") or "<unknown>"
    return str(value).rsplit("/", 1)[-1]


def kernel_kind(record: dict[str, Any]) -> str:
    text = f"{record.get('unmangled_name') or ''} {record.get('source_trace') or ''}"
    if "gemm" in text or "gemv" in text or "cublas" in text:
        return "linear_matmul"
    if "silu" in text:
        return "silu_elementwise"
    if "MulFunctor" in text:
        return "mul_elementwise"
    return "other"


def expected_mlp_bytes(hidden_size: int, intermediate_size: int, dtype_bytes: int) -> dict[str, int]:
    projection_bytes = hidden_size * intermediate_size * dtype_bytes
    return {
        "one_projection_weight_bytes": projection_bytes,
        "gate_proj_weight_bytes": projection_bytes,
        "up_proj_weight_bytes": projection_bytes,
        "down_proj_weight_bytes": projection_bytes,
        "three_projection_weight_bytes": projection_bytes * 3,
        "ffn_input_bytes": hidden_size * dtype_bytes,
        "intermediate_vector_bytes": intermediate_size * dtype_bytes,
        "output_vector_bytes": hidden_size * dtype_bytes,
    }


def make_counter() -> dict[str, Any]:
    return {
        "events": 0,
        "addr_items": 0,
        "global_addr_items": 0,
        "shared_or_small_addr_items": 0,
        "unique_global_addrs": set(),
        "min_global_addr": None,
        "max_global_addr": None,
        "sample_global_addrs": [],
        "pcs": Counter(),
    }


def add_global_addr(stats: dict[str, Any], addr: int, sample_limit: int) -> None:
    stats["global_addr_items"] += 1
    stats["unique_global_addrs"].add(addr)
    stats["min_global_addr"] = (
        addr if stats["min_global_addr"] is None else min(stats["min_global_addr"], addr)
    )
    stats["max_global_addr"] = (
        addr if stats["max_global_addr"] is None else max(stats["max_global_addr"], addr)
    )
    samples = stats["sample_global_addrs"]
    if len(samples) < sample_limit and addr not in samples:
        samples.append(addr)


def finalize_stats(stats: dict[str, Any], dtype_bytes: int) -> dict[str, Any]:
    unique_global = stats.pop("unique_global_addrs")
    pcs = stats.pop("pcs")
    result = dict(stats)
    result["unique_global_addr_count"] = len(unique_global)
    result["unique_global_bytes_if_element_addresses"] = len(unique_global) * dtype_bytes
    result["top_pcs"] = [{"pc": pc, "events": count} for pc, count in pcs.most_common(10)]
    if result["min_global_addr"] is not None and result["max_global_addr"] is not None:
        result["global_addr_span"] = result["max_global_addr"] - result["min_global_addr"] + 1
    else:
        result["global_addr_span"] = 0
    return result


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    processed_path = args.run_dir / args.processed_name
    if not processed_path.is_file():
        raise FileNotFoundError(f"Processed trace not found: {processed_path}")

    overall = make_counter()
    by_source: dict[str, dict[str, Any]] = defaultdict(make_counter)
    by_kind: dict[str, dict[str, Any]] = defaultdict(make_counter)
    source_kind: dict[str, str] = {}

    for _line_no, record in iter_jsonl(processed_path):
        addrs = record.get("addrs") or []
        src = source_name(record)
        kind = kernel_kind(record)
        source_kind[src] = kind

        for bucket in (overall, by_source[src], by_kind[kind]):
            bucket["events"] += 1
            bucket["addr_items"] += len(addrs)
            pc = record.get("pc")
            if pc is not None:
                bucket["pcs"][str(pc)] += 1

        for raw_addr in addrs:
            addr = int(raw_addr)
            if addr >= args.global_addr_min:
                for bucket in (overall, by_source[src], by_kind[kind]):
                    add_global_addr(bucket, addr, args.sample_limit)
            else:
                for bucket in (overall, by_source[src], by_kind[kind]):
                    bucket["shared_or_small_addr_items"] += 1

    expected = expected_mlp_bytes(args.hidden_size, args.intermediate_size, args.dtype_bytes)
    overall_final = finalize_stats(overall, args.dtype_bytes)

    return {
        "processed_path": str(processed_path),
        "global_addr_min": args.global_addr_min,
        "classification_note": (
            "Processed events in this run have sass=null and raw_trace was deleted, so global "
            "reads are estimated by treating addresses >= global_addr_min as global-like. "
            "For exact load/store classification, keep raw_trace and map pc/opcode_id to SASS."
        ),
        "model_shape": {
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "dtype_bytes": args.dtype_bytes,
        },
        "expected_mlp_bytes": expected,
        "overall": overall_final,
        "coverage_vs_three_projection_weights": (
            overall_final["unique_global_bytes_if_element_addresses"]
            / expected["three_projection_weight_bytes"]
        ),
        "by_kind": {
            key: finalize_stats(value, args.dtype_bytes)
            for key, value in sorted(by_kind.items())
        },
        "by_source": {
            key: {
                "kind": source_kind.get(key, "other"),
                **finalize_stats(value, args.dtype_bytes),
            }
            for key, value in sorted(
                by_source.items(),
                key=lambda item: len(item[1]["unique_global_addrs"]),
                reverse=True,
            )
        },
    }


def main() -> None:
    args = parse_args()
    result = analyze(args)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
