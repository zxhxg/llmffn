import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_HIDDEN_SIZE = 4096
DEFAULT_INTERMEDIATE_SIZE = 14336
DEFAULT_DTYPE_BYTES = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Hopper/H100 CUTracer raw GEMM traces and check whether recorded "
            "global-load address events cover the expected Llama MLP projection weights."
        )
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--intermediate-size", type=int, default=DEFAULT_INTERMEDIATE_SIZE)
    parser.add_argument("--dtype-bytes", type=int, default=DEFAULT_DTYPE_BYTES)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def classify_sass(sass: str) -> str:
    if "UTMALDG" in sass:
        return "UTMALDG"
    if re.search(r"\bLDGSTS\b", sass):
        return "LDGSTS"
    if re.search(r"\bLDG", sass):
        return "LDG"
    if re.search(r"\bSTG", sass):
        return "STG"
    if re.search(r"\bLDS", sass):
        return "LDS"
    if re.search(r"\bSTS", sass):
        return "STS"
    if "SYNCS" in sass:
        return "SYNCS"
    return "OTHER"


def iter_json_records(path: Path):
    with path.open("rb") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def analyze_kernel(path: Path) -> dict:
    records = iter_json_records(path)
    metadata = next(records)
    instructions = metadata.get("instructions", {})

    class_events = Counter()
    class_addr_items = Counter()
    class_samples = {}
    opcode_events = Counter()
    opcode_addr_items = Counter()

    load_addr_items = 0
    load_unique = set()
    load_min = None
    load_max = None
    load_event_opcodes = Counter()

    metadata_instr_classes = Counter(
        classify_sass((entry.get("sass") or "")) for entry in instructions.values()
    )

    for record in records:
        if record.get("type") != "mem_addr_trace":
            continue
        opcode = str(record.get("opcode_id"))
        sass = instructions.get(opcode, {}).get("sass") or ""
        cls = classify_sass(sass)
        addrs = [int(value) for value in (record.get("addrs") or [])]

        class_events[cls] += 1
        class_addr_items[cls] += len(addrs)
        opcode_events[opcode] += 1
        opcode_addr_items[opcode] += len(addrs)
        class_samples.setdefault(
            cls,
            {
                "opcode_id": opcode,
                "sass": sass,
                "addrs": addrs[:16],
            },
        )

        if cls in {"UTMALDG", "LDGSTS", "LDG"}:
            load_event_opcodes[opcode] += 1
            load_addr_items += len(addrs)
            for addr in addrs:
                load_unique.add(addr)
                load_min = addr if load_min is None else min(load_min, addr)
                load_max = addr if load_max is None else max(load_max, addr)

    opcode_details = []
    for opcode, count in opcode_events.most_common():
        sass = instructions.get(opcode, {}).get("sass") or ""
        opcode_details.append(
            {
                "opcode_id": opcode,
                "sass": sass,
                "class": classify_sass(sass),
                "events": count,
                "addr_items": opcode_addr_items[opcode],
            }
        )

    metadata_load_instr = []
    for opcode, entry in sorted(instructions.items(), key=lambda item: int(item[0])):
        sass = entry.get("sass") or ""
        if classify_sass(sass) in {"UTMALDG", "LDGSTS", "LDG"}:
            metadata_load_instr.append({"opcode_id": opcode, "sass": sass})

    return {
        "file": str(path),
        "kernel": metadata.get("unmangled_name") or metadata.get("mangled_name"),
        "grid": metadata.get("grid"),
        "block": metadata.get("block"),
        "metadata_instruction_classes": dict(metadata_instr_classes),
        "metadata_load_instructions": metadata_load_instr,
        "event_classes": {
            cls: {
                "events": class_events[cls],
                "addr_items": class_addr_items[cls],
                "sample": class_samples.get(cls),
            }
            for cls in sorted(class_events)
        },
        "opcode_details_top": opcode_details[:40],
        "load_like": {
            "classes_counted": ["UTMALDG", "LDGSTS", "LDG"],
            "event_opcodes": dict(load_event_opcodes),
            "addr_items": load_addr_items,
            "unique_addr_count": len(load_unique),
            "unique_128bit_bytes": len(load_unique) * 16,
            "min_addr": load_min,
            "max_addr": load_max,
        },
    }


def main() -> None:
    args = parse_args()
    raw_trace = args.run_dir / "raw_trace"
    paths = sorted(raw_trace.glob("kernel_*gemm*.ndjson"))
    if not paths:
        raise FileNotFoundError(f"No GEMM raw trace files found under {raw_trace}")

    expected_one_projection_bytes = (
        args.hidden_size * args.intermediate_size * args.dtype_bytes
    )
    expected_three_projection_bytes = expected_one_projection_bytes * 3

    kernels = [analyze_kernel(path) for path in paths]
    total_unique_128bit_bytes = sum(
        kernel["load_like"]["unique_128bit_bytes"] for kernel in kernels
    )
    total_load_addr_items = sum(kernel["load_like"]["addr_items"] for kernel in kernels)
    total_load_events = sum(
        sum(kernel["load_like"]["event_opcodes"].values()) for kernel in kernels
    )

    result = {
        "run_dir": str(args.run_dir),
        "expected": {
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "dtype_bytes": args.dtype_bytes,
            "one_projection_weight_bytes": expected_one_projection_bytes,
            "three_projection_weight_bytes": expected_three_projection_bytes,
        },
        "summary": {
            "gemm_kernel_count": len(kernels),
            "total_load_like_events": total_load_events,
            "total_load_like_addr_items": total_load_addr_items,
            "total_load_like_unique_128bit_bytes_summed_by_kernel": total_unique_128bit_bytes,
            "ratio_to_three_projection_weight_bytes": (
                total_unique_128bit_bytes / expected_three_projection_bytes
            ),
            "note": (
                "On this H100 trace, metadata contains Hopper TMA/load instructions, "
                "but mem_addr_trace events are dominated by SYNCS/LDS/STS. UTMALDG "
                "instructions do not emit address events in the recorded files."
            ),
        },
        "kernels": kernels,
    }

    output = args.output or (args.run_dir / "h100_raw_global_completeness.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
