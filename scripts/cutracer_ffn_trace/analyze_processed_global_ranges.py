from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_GLOBAL_ADDR_MIN = 1 << 20
DEFAULT_GLOBAL_ADDR_MAX = 1 << 48


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze global-like address ranges in a processed CUTracer JSONL trace. "
            "This avoids storing every unique address, so it is suitable for very large "
            "4070S GEMV traces."
        )
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--processed-name", default="processed_ffn_mem_sequence.jsonl")
    parser.add_argument("--global-addr-min", type=int, default=DEFAULT_GLOBAL_ADDR_MIN)
    parser.add_argument(
        "--global-addr-max",
        type=int,
        default=DEFAULT_GLOBAL_ADDR_MAX,
        help="Ignore implausibly large values above this threshold.",
    )
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=14336)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    parser.add_argument("--max-intervals-per-source", type=int, default=5_000_000)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def source_name(record: dict[str, Any]) -> str:
    return str(record.get("source_trace") or "<unknown>").rsplit("/", 1)[-1]


def kernel_kind(record: dict[str, Any]) -> str:
    text = f"{record.get('unmangled_name') or ''} {record.get('source_trace') or ''}"
    if "gemv" in text:
        return "gemv_linear"
    if "gemm" in text or "cublas" in text:
        return "gemm_linear"
    if "silu" in text:
        return "silu_elementwise"
    if "MulFunctor" in text:
        return "mul_elementwise"
    return "other"


def expected_bytes(hidden_size: int, intermediate_size: int, dtype_bytes: int) -> dict[str, int]:
    one = hidden_size * intermediate_size * dtype_bytes
    return {
        "one_projection_weight_bytes": one,
        "three_projection_weight_bytes": one * 3,
        "gate_or_up_output_vector_bytes": intermediate_size * dtype_bytes,
        "down_output_vector_bytes": hidden_size * dtype_bytes,
    }


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals.sort()
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            if end > prev_end:
                merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def covered_bytes(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start for start, end in intervals)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    processed_path = args.run_dir / args.processed_name
    if not processed_path.is_file():
        raise FileNotFoundError(processed_path)

    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "kind": None,
            "events": 0,
            "addr_items": 0,
            "global_addr_items": 0,
            "small_addr_items": 0,
            "ignored_huge_addr_items": 0,
            "patterns": Counter(),
            "pcs": Counter(),
            "min_global_addr": None,
            "max_global_addr": None,
            "intervals": [],
            "intervals_truncated": False,
            "samples": [],
        }
    )
    overall = {
        "events": 0,
        "addr_items": 0,
        "global_addr_items": 0,
        "small_addr_items": 0,
        "ignored_huge_addr_items": 0,
        "patterns": Counter(),
    }

    with processed_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            src = source_name(record)
            entry = stats[src]
            entry["kind"] = entry["kind"] or kernel_kind(record)
            entry["events"] += 1
            overall["events"] += 1
            pc = record.get("pc")
            if pc is not None:
                entry["pcs"][str(pc)] += 1

            addrs = [int(value) for value in (record.get("addrs") or [])]
            entry["addr_items"] += len(addrs)
            overall["addr_items"] += len(addrs)
            global_addrs: list[int] = []
            for addr in addrs:
                if addr < args.global_addr_min:
                    entry["small_addr_items"] += 1
                    overall["small_addr_items"] += 1
                elif addr > args.global_addr_max:
                    entry["ignored_huge_addr_items"] += 1
                    overall["ignored_huge_addr_items"] += 1
                else:
                    global_addrs.append(addr)

            if not global_addrs:
                continue

            entry["global_addr_items"] += len(global_addrs)
            overall["global_addr_items"] += len(global_addrs)
            mn = min(global_addrs)
            mx = max(global_addrs)
            entry["min_global_addr"] = (
                mn if entry["min_global_addr"] is None else min(entry["min_global_addr"], mn)
            )
            entry["max_global_addr"] = (
                mx if entry["max_global_addr"] is None else max(entry["max_global_addr"], mx)
            )
            if len(entry["samples"]) < 5:
                entry["samples"].append(global_addrs[:8])

            if len(set(global_addrs)) == 1:
                pattern = "all_same"
                interval = (global_addrs[0], global_addrs[0] + args.dtype_bytes)
            elif all(
                global_addrs[index + 1] - global_addrs[index] == args.dtype_bytes
                for index in range(len(global_addrs) - 1)
            ):
                pattern = f"stride_{args.dtype_bytes}"
                interval = (global_addrs[0], global_addrs[-1] + args.dtype_bytes)
            else:
                pattern = "other"
                interval = None

            entry["patterns"][pattern] += 1
            overall["patterns"][pattern] += 1
            if interval is not None and not entry["intervals_truncated"]:
                if len(entry["intervals"]) < args.max_intervals_per_source:
                    entry["intervals"].append(interval)
                else:
                    entry["intervals_truncated"] = True

    finalized_sources = {}
    by_kind: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "events": 0,
            "addr_items": 0,
            "global_addr_items": 0,
            "small_addr_items": 0,
            "ignored_huge_addr_items": 0,
            "covered_bytes_from_mergeable_intervals": 0,
            "sources": 0,
        }
    )
    for src, entry in sorted(stats.items(), key=lambda item: item[1]["global_addr_items"], reverse=True):
        merged = merge_intervals(entry["intervals"])
        merged_bytes = covered_bytes(merged)
        payload = {
            key: value
            for key, value in entry.items()
            if key not in {"intervals", "patterns", "pcs"}
        }
        payload["patterns"] = dict(entry["patterns"].most_common())
        payload["top_pcs"] = [{"pc": pc, "events": count} for pc, count in entry["pcs"].most_common(8)]
        payload["mergeable_interval_count"] = len(entry["intervals"])
        payload["merged_interval_count"] = len(merged)
        payload["covered_bytes_from_mergeable_intervals"] = merged_bytes
        payload["covered_fp16_values_from_mergeable_intervals"] = merged_bytes // args.dtype_bytes
        payload["first_merged_intervals"] = merged[:10]
        finalized_sources[src] = payload

        kind_entry = by_kind[payload["kind"]]
        kind_entry["sources"] += 1
        for key in (
            "events",
            "addr_items",
            "global_addr_items",
            "small_addr_items",
            "ignored_huge_addr_items",
            "covered_bytes_from_mergeable_intervals",
        ):
            kind_entry[key] += payload[key]

    result = {
        "processed_path": str(processed_path),
        "classification_note": (
            "This script estimates global-like addresses from processed JSONL by address "
            "thresholds because raw_trace was deleted and processed events have sass=null. "
            "For 4070S GEMV traces, preview/global patterns are mostly contiguous fp16 "
            "byte addresses with stride 2, so merged interval coverage is a useful sanity check."
        ),
        "global_addr_min": args.global_addr_min,
        "global_addr_max": args.global_addr_max,
        "model_shape": {
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "dtype_bytes": args.dtype_bytes,
        },
        "expected_bytes": expected_bytes(args.hidden_size, args.intermediate_size, args.dtype_bytes),
        "overall": {
            **{k: v for k, v in overall.items() if k != "patterns"},
            "patterns": dict(overall["patterns"].most_common()),
        },
        "by_kind": by_kind,
        "by_source": finalized_sources,
    }
    return result


def main() -> None:
    args = parse_args()
    result = analyze(args)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
