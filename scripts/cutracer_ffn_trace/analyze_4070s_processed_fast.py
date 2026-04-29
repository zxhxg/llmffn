from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


GLOBAL_ADDR_MIN = 1 << 20
GLOBAL_ADDR_MAX = 1 << 48
DTYPE_BYTES = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fast range/coverage analysis for the 4070S processed FFN trace. "
            "It avoids full json.loads on the 17GB JSONL file and estimates unique "
            "global fp16 addresses with a compact bitmap per source kernel."
        )
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--processed-name", default="processed_ffn_mem_sequence.jsonl")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=14336)
    return parser.parse_args()


def source_key(line: str) -> str:
    if "kernel_cd181a8ecb276135_iter0" in line:
        return "gemv_iter0"
    if "kernel_cd181a8ecb276135_iter1" in line:
        return "gemv_iter1"
    if "kernel_cd181a8ecb276135_iter2" in line:
        return "gemv_iter2"
    if "kernel_3ed1ff7a7c68d7d" in line:
        return "mul_elementwise"
    if "kernel_c4726ee7c8e6b75e" in line:
        return "silu_elementwise"
    return "other"


def iter_addrs(line: str):
    marker = '"addrs": ['
    start = line.find(marker)
    if start < 0:
        return
    start += len(marker)
    end = line.find("]", start)
    if end < 0:
        return
    for part in line[start:end].split(","):
        part = part.strip()
        if part:
            yield int(part)


def expected_bytes(hidden: int, intermediate: int) -> dict[str, int]:
    one = hidden * intermediate * DTYPE_BYTES
    return {
        "one_projection_weight_bytes": one,
        "three_projection_weight_bytes": one * 3,
        "gate_or_up_output_vector_bytes": intermediate * DTYPE_BYTES,
        "down_output_vector_bytes": hidden * DTYPE_BYTES,
    }


def make_stats() -> dict:
    return {
        "events": 0,
        "addr_items": 0,
        "global_addr_items": 0,
        "small_addr_items": 0,
        "ignored_huge_addr_items": 0,
        "min_global_addr": None,
        "max_global_addr": None,
        "patterns": Counter(),
        "samples": [],
    }


def add_minmax(stats: dict, addr: int) -> None:
    stats["min_global_addr"] = (
        addr if stats["min_global_addr"] is None else min(stats["min_global_addr"], addr)
    )
    stats["max_global_addr"] = (
        addr if stats["max_global_addr"] is None else max(stats["max_global_addr"], addr)
    )


def classify_pattern(addrs: list[int]) -> str:
    if not addrs:
        return "no_global"
    if len(set(addrs)) == 1:
        return "all_same"
    if all(addrs[index + 1] - addrs[index] == DTYPE_BYTES for index in range(len(addrs) - 1)):
        return "stride_2"
    return "other"


def first_pass(path: Path) -> dict[str, dict]:
    stats_by_source: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            key = source_key(line)
            stats = stats_by_source.setdefault(key, make_stats())
            stats["events"] += 1
            global_addrs = []
            for addr in iter_addrs(line):
                stats["addr_items"] += 1
                if addr < GLOBAL_ADDR_MIN:
                    stats["small_addr_items"] += 1
                elif addr > GLOBAL_ADDR_MAX:
                    stats["ignored_huge_addr_items"] += 1
                else:
                    stats["global_addr_items"] += 1
                    global_addrs.append(addr)
                    add_minmax(stats, addr)
            stats["patterns"][classify_pattern(global_addrs)] += 1
            if global_addrs and len(stats["samples"]) < 5:
                stats["samples"].append(global_addrs[:8])
    return stats_by_source


def allocate_bitmaps(stats_by_source: dict[str, dict]) -> dict[str, bytearray]:
    bitmaps: dict[str, bytearray] = {}
    for key, stats in stats_by_source.items():
        mn = stats["min_global_addr"]
        mx = stats["max_global_addr"]
        if mn is None or mx is None:
            continue
        slots = ((mx - mn) // DTYPE_BYTES) + 1
        # One byte per fp16 address slot. This is intentionally simple and still
        # modest for these traces: each projection is roughly 56M slots.
        bitmaps[key] = bytearray(slots)
    return bitmaps


def second_pass(path: Path, stats_by_source: dict[str, dict], bitmaps: dict[str, bytearray]) -> None:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            key = source_key(line)
            bitmap = bitmaps.get(key)
            if bitmap is None:
                continue
            base = stats_by_source[key]["min_global_addr"]
            for addr in iter_addrs(line):
                if GLOBAL_ADDR_MIN <= addr <= GLOBAL_ADDR_MAX:
                    index = (addr - base) // DTYPE_BYTES
                    if 0 <= index < len(bitmap):
                        bitmap[index] = 1


def finalize(stats_by_source: dict[str, dict], bitmaps: dict[str, bytearray], args: argparse.Namespace) -> dict:
    expected = expected_bytes(args.hidden_size, args.intermediate_size)
    result_sources = {}
    total_unique = 0
    for key, stats in sorted(stats_by_source.items()):
        unique_count = sum(bitmaps[key]) if key in bitmaps else 0
        total_unique += unique_count
        span = (
            0
            if stats["min_global_addr"] is None
            else stats["max_global_addr"] - stats["min_global_addr"] + 1
        )
        payload = {
            "events": stats["events"],
            "addr_items": stats["addr_items"],
            "global_addr_items": stats["global_addr_items"],
            "small_addr_items": stats["small_addr_items"],
            "ignored_huge_addr_items": stats["ignored_huge_addr_items"],
            "min_global_addr": stats["min_global_addr"],
            "max_global_addr": stats["max_global_addr"],
            "global_addr_span_bytes": span,
            "unique_global_fp16_addr_count": unique_count,
            "unique_global_bytes": unique_count * DTYPE_BYTES,
            "coverage_vs_one_projection_weight": (
                (unique_count * DTYPE_BYTES) / expected["one_projection_weight_bytes"]
                if expected["one_projection_weight_bytes"]
                else None
            ),
            "patterns": dict(stats["patterns"].most_common()),
            "samples": stats["samples"],
        }
        result_sources[key] = payload

    gemv_unique = sum(
        result_sources[key]["unique_global_fp16_addr_count"]
        for key in result_sources
        if key.startswith("gemv_")
    )
    return {
        "classification_note": (
            "raw_trace was deleted and processed events have sass=null. This analysis treats "
            "plausible addresses in [1MiB, 2^48] as global-like. For this 4070S GEMV trace, "
            "preview shows stride-2 fp16 byte addresses, so bitmap coverage is a useful sanity check."
        ),
        "global_addr_min": GLOBAL_ADDR_MIN,
        "global_addr_max": GLOBAL_ADDR_MAX,
        "model_shape": {
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "dtype_bytes": DTYPE_BYTES,
        },
        "expected_bytes": expected,
        "gemv_unique_global_fp16_addr_count": gemv_unique,
        "gemv_unique_global_bytes": gemv_unique * DTYPE_BYTES,
        "gemv_coverage_vs_three_projection_weights": (
            (gemv_unique * DTYPE_BYTES) / expected["three_projection_weight_bytes"]
        ),
        "by_source": result_sources,
    }


def main() -> None:
    args = parse_args()
    processed_path = args.run_dir / args.processed_name
    if not processed_path.is_file():
        raise FileNotFoundError(processed_path)
    stats_by_source = first_pass(processed_path)
    bitmaps = allocate_bitmaps(stats_by_source)
    second_pass(processed_path, stats_by_source, bitmaps)
    result = finalize(stats_by_source, bitmaps, args)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
