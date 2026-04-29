import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


PAGE_BYTES = 4096
FP16_BYTES = 2
SLOTS_PER_PAGE = PAGE_BYTES // FP16_BYTES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze raw CUTracer output from replay_single_ffn_weight_scan.py. "
            "Counts unique fp16 global addresses emitted by scan_fp16_weight_kernel."
        )
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=14336)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def iter_json_records(path: Path):
    with path.open("rb") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def is_global_load_sass(sass: str) -> bool:
    return bool(re.search(r"\bLDG|\bLD\.GLOBAL|ld\.global", sass, re.IGNORECASE))


def add_addr_to_pages(pages: dict[int, bytearray], addr: int) -> None:
    if addr <= 0:
        return
    page = addr // PAGE_BYTES
    slot = (addr % PAGE_BYTES) // FP16_BYTES
    if slot < 0 or slot >= SLOTS_PER_PAGE:
        return
    bitmap = pages.get(page)
    if bitmap is None:
        bitmap = bytearray(SLOTS_PER_PAGE // 8)
        pages[page] = bitmap
    bitmap[slot // 8] |= 1 << (slot % 8)


def count_pages(pages: dict[int, bytearray]) -> int:
    total = 0
    for bitmap in pages.values():
        total += sum(byte.bit_count() for byte in bitmap)
    return total


def analyze_kernel(path: Path) -> dict | None:
    records = iter_json_records(path)
    metadata = next(records, None)
    if not metadata:
        return None
    kernel = metadata.get("unmangled_name") or metadata.get("mangled_name") or ""
    if "scan_fp16_weight_kernel" not in kernel:
        return None

    instructions = metadata.get("instructions", {})
    pages: dict[int, bytearray] = {}
    opcode_events = defaultdict(int)
    load_events = 0
    load_addr_items = 0
    min_addr = None
    max_addr = None
    samples = []

    for record in records:
        if record.get("type") != "mem_addr_trace":
            continue
        opcode = str(record.get("opcode_id"))
        sass = instructions.get(opcode, {}).get("sass") or ""
        if not is_global_load_sass(sass):
            continue
        addrs = [int(value) for value in (record.get("addrs") or []) if int(value) > 0]
        if not addrs:
            continue
        load_events += 1
        load_addr_items += len(addrs)
        opcode_events[opcode] += 1
        if len(samples) < 5:
            samples.append(addrs[:16])
        mn = min(addrs)
        mx = max(addrs)
        min_addr = mn if min_addr is None else min(min_addr, mn)
        max_addr = mx if max_addr is None else max(max_addr, mx)
        for addr in addrs:
            add_addr_to_pages(pages, addr)

    unique_fp16 = count_pages(pages)
    return {
        "file": str(path),
        "kernel": kernel,
        "grid": metadata.get("grid"),
        "block": metadata.get("block"),
        "load_events": load_events,
        "load_addr_items": load_addr_items,
        "load_opcodes": dict(opcode_events),
        "min_addr": min_addr,
        "max_addr": max_addr,
        "unique_fp16_addresses": unique_fp16,
        "unique_bytes": unique_fp16 * FP16_BYTES,
        "samples": samples,
    }


def main() -> None:
    args = parse_args()
    raw_trace_dir = args.run_dir / "raw_trace"
    kernels = []
    for path in sorted(raw_trace_dir.glob("kernel_*.ndjson")):
        result = analyze_kernel(path)
        if result is not None:
            kernels.append(result)

    expected_one = args.hidden_size * args.intermediate_size * FP16_BYTES
    expected_three = expected_one * 3
    total_unique_bytes = sum(kernel["unique_bytes"] for kernel in kernels)

    result = {
        "run_dir": str(args.run_dir),
        "expected": {
            "one_projection_weight_bytes": expected_one,
            "three_projection_weight_bytes": expected_three,
        },
        "summary": {
            "scan_kernel_count": len(kernels),
            "total_unique_bytes_summed_by_kernel": total_unique_bytes,
            "ratio_to_three_projection_weight_bytes": (
                total_unique_bytes / expected_three if expected_three else None
            ),
        },
        "kernels": kernels,
    }

    output = args.output or (args.run_dir / "weight_scan_global_reads.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
