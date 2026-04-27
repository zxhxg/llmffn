from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "output" / "runs"
DEFAULT_CUTRACER_SO = REPO_ROOT / "third_party" / "CUTracer" / "lib" / "cutracer.so"
KERNEL_EVENT_TYPES = {"kernel_metadata", "kernel_launch"}
MEM_RECORD_TYPES = {"mem_trace", "mem_addr_trace"}
DEFAULT_CALLSTACK_MARKERS = ("target_matmul_once",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a 128x128 CUDA matmul under CUTracer and summarize the traced memory addresses."
        ),
    )
    parser.add_argument("--size", type=int, default=128, help="Matrix side length. Default: 128.")
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Input dtype used by matmul_128.py.",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup matmul count before target call.")
    parser.add_argument("--cutracer-so", type=Path, default=None, help="Path to cutracer.so.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable to use.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--no-data-timeout-s", type=int, default=120)
    parser.add_argument("--trace-size-limit-mb", type=int, default=0)
    parser.add_argument(
        "--no-dump-cubin",
        action="store_true",
        help="Forward --no-dump-cubin to CUTracer and set CUTRACER_DUMP_CUBIN=0.",
    )
    parser.add_argument(
        "--callstack-marker",
        action="append",
        default=None,
        help="Kernel CPU callstack marker to include in address stats. May be used multiple times.",
    )
    parser.add_argument(
        "--include-all-kernels",
        action="store_true",
        help="Ignore callstack markers and summarize all memory trace kernels.",
    )
    parser.add_argument(
        "--preview-events",
        type=int,
        default=20,
        help="Number of matching memory events copied into addr_preview.jsonl.",
    )
    return parser.parse_args()


def timestamped_run_dir(args: argparse.Namespace) -> Path:
    if args.run_name:
        return args.output_root / args.run_name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return args.output_root / f"matmul_{args.size}_{args.dtype}_{stamp}"


def resolve_cutracer_so(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    env_value = os.environ.get("CUTRACER_SO")
    if env_value:
        candidates.append(Path(env_value))
    candidates.append(DEFAULT_CUTRACER_SO)

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find cutracer.so. Pass --cutracer-so or set CUTRACER_SO. "
        f"Searched: {searched}"
    )


def run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    print("[command]")
    print(shlex.join(command))
    result = subprocess.run(command, cwd=str(cwd), env=env, text=True, capture_output=True)
    if result.stdout:
        print("[stdout]")
        print(result.stdout.rstrip())
    if result.stderr:
        print("[stderr]")
        print(result.stderr.rstrip())
    return result


def iter_trace_paths(raw_trace_dir: Path) -> list[Path]:
    paths = sorted(path.resolve() for path in raw_trace_dir.glob("*.ndjson") if path.is_file())
    if not paths:
        raise FileNotFoundError(f"No .ndjson files found in {raw_trace_dir}")
    return paths


def iter_json_records(path: Path):
    skipped_decode = 0
    skipped_json = 0
    with path.open("rb") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                skipped_decode += 1
                line = raw_line.decode("utf-8", errors="replace")
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped_json += 1
                continue
            if isinstance(record, dict):
                yield line_no, record
    if skipped_decode or skipped_json:
        print(
            f"[warn] skipped malformed records in {path}: "
            f"decode_errors={skipped_decode}, json_errors={skipped_json}",
            file=sys.stderr,
        )


def record_type(record: dict[str, Any]) -> str | None:
    return record.get("type") or record.get("message_type")


def launch_id(record: dict[str, Any]) -> int | None:
    value = record.get("grid_launch_id", record.get("kernel_launch_id"))
    return None if value is None else int(value)


def callstack_text(record: dict[str, Any]) -> str:
    callstack = record.get("cpu_callstack")
    if callstack is None:
        return ""
    if not isinstance(callstack, list):
        return str(callstack)
    parts = []
    for entry in callstack:
        if isinstance(entry, str):
            parts.append(entry)
        else:
            parts.append(json.dumps(entry, ensure_ascii=False, sort_keys=True))
    return "\n".join(parts)


def kernel_name(record: dict[str, Any]) -> str | None:
    return record.get("unmangled_name") or record.get("kernel_name") or record.get("mangled_name")


def collect_kernel_metadata(
    paths: list[Path],
    markers: tuple[str, ...],
    include_all_kernels: bool,
) -> tuple[dict[int, dict[str, Any]], str]:
    all_metadata: dict[int, dict[str, Any]] = {}
    matched_metadata: dict[int, dict[str, Any]] = {}

    for path in paths:
        for line_no, record in iter_json_records(path):
            if record_type(record) not in KERNEL_EVENT_TYPES:
                continue
            kid = launch_id(record)
            if kid is None:
                continue
            entry = {
                "kernel_launch_id": kid,
                "mangled_name": record.get("mangled_name"),
                "unmangled_name": kernel_name(record),
                "source_trace": str(path),
                "source_line": line_no,
            }
            all_metadata[kid] = entry
            if any(marker in callstack_text(record) for marker in markers):
                matched_metadata[kid] = entry

    if include_all_kernels:
        return all_metadata, "include_all_kernels"
    if matched_metadata:
        return matched_metadata, "callstack_marker"
    if all_metadata:
        return all_metadata, "fallback_all_kernel_metadata_no_marker_match"
    raise RuntimeError("No kernel metadata records were found. Use CUTracer with --kernel-events full.")


def classify_addrs(addrs: list[int]) -> str:
    if len(addrs) <= 1:
        return "short"
    unique_count = len(set(addrs))
    diffs = [addrs[index + 1] - addrs[index] for index in range(len(addrs) - 1)]
    if unique_count == 1:
        return "all_same"
    if all(diff == 2 for diff in diffs):
        return "stride_2"
    if all(diff == 4 for diff in diffs):
        return "stride_4"
    if any(addr == 0 for addr in addrs):
        return "contains_zero"
    return "other"


def short_kernel_name(name: str | None, limit: int = 160) -> str:
    if not name:
        return "<unknown>"
    return name if len(name) <= limit else name[: limit - 3] + "..."


def summarize_memory_addresses(
    raw_trace_dir: Path,
    stats_path: Path,
    preview_path: Path,
    *,
    markers: tuple[str, ...],
    include_all_kernels: bool,
    preview_events: int,
) -> dict[str, Any]:
    paths = iter_trace_paths(raw_trace_dir)
    metadata_by_launch_id, selection_mode = collect_kernel_metadata(
        paths,
        markers,
        include_all_kernels,
    )
    target_ids = set(metadata_by_launch_id)

    total_events = 0
    total_addresses = 0
    unique_addresses: set[int] = set()
    addr_min: int | None = None
    addr_max: int | None = None
    length_hist: Counter[int] = Counter()
    unique_per_event_hist: Counter[int] = Counter()
    pattern_hist: Counter[str] = Counter()
    kernel_hist: Counter[str] = Counter()
    pc_hist: Counter[str] = Counter()
    per_kernel: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, dict[str, Any]] = {}

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_written = 0
    with preview_path.open("w", encoding="utf-8") as preview_handle:
        for path in paths:
            for _line_no, record in iter_json_records(path):
                if record_type(record) not in MEM_RECORD_TYPES:
                    continue
                kid = launch_id(record)
                if kid not in target_ids:
                    continue
                raw_addrs = record.get("addrs") or []
                addrs = [int(addr) for addr in raw_addrs]
                pattern = classify_addrs(addrs)
                meta = metadata_by_launch_id[kid]
                name = short_kernel_name(meta.get("unmangled_name"))
                pc = str(record.get("pc"))

                total_events += 1
                total_addresses += len(addrs)
                length_hist[len(addrs)] += 1
                unique_per_event_hist[len(set(addrs))] += 1
                pattern_hist[pattern] += 1
                kernel_hist[name] += 1
                pc_hist[f"{name} @ {pc}"] += 1
                per_kernel[name]["events"] += 1
                per_kernel[name][f"pattern_{pattern}"] += 1

                for addr in addrs:
                    unique_addresses.add(addr)
                    addr_min = addr if addr_min is None else min(addr_min, addr)
                    addr_max = addr if addr_max is None else max(addr_max, addr)

                examples.setdefault(
                    pattern,
                    {
                        "kernel_launch_id": kid,
                        "trace_index": record.get("trace_index"),
                        "timestamp": record.get("timestamp"),
                        "kernel": name,
                        "pc": record.get("pc"),
                        "unique_addr_count": len(set(addrs)),
                        "first_16_addrs": addrs[:16],
                    },
                )
                if preview_written < preview_events:
                    payload = {
                        "event_index": total_events,
                        "kernel_launch_id": kid,
                        "trace_index": record.get("trace_index"),
                        "timestamp": record.get("timestamp"),
                        "kernel": name,
                        "pc": record.get("pc"),
                        "pattern": pattern,
                        "addrs": addrs,
                    }
                    preview_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    preview_written += 1

    if total_events == 0:
        raise RuntimeError("No memory-address records matched the selected kernels.")

    stats = {
        "raw_trace_dir": str(raw_trace_dir),
        "trace_files_scanned": len(paths),
        "selection_mode": selection_mode,
        "callstack_markers": list(markers),
        "matched_kernel_launches": len(metadata_by_launch_id),
        "total_memory_events": total_events,
        "total_addresses": total_addresses,
        "unique_addresses": len(unique_addresses),
        "address_min": addr_min,
        "address_max": addr_max,
        "addr_list_length_histogram": dict(sorted(length_hist.items())),
        "unique_addresses_per_event_histogram": dict(sorted(unique_per_event_hist.items())),
        "address_pattern_histogram": dict(pattern_hist.most_common()),
        "top_kernels": [{"kernel": key, "events": value} for key, value in kernel_hist.most_common(20)],
        "top_kernel_pcs": [{"kernel_pc": key, "events": value} for key, value in pc_hist.most_common(20)],
        "per_kernel": {
            key: dict(value)
            for key, value in sorted(per_kernel.items(), key=lambda item: item[1]["events"], reverse=True)
        },
        "examples": examples,
        "preview_path": str(preview_path),
        "preview_events_written": preview_written,
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def raw_trace_has_data(raw_trace_dir: Path) -> bool:
    return any(raw_trace_dir.glob("kernel_*.ndjson")) and any(
        raw_trace_dir.glob("cutracer_kernel_events_*.ndjson")
    )


def main() -> None:
    args = parse_args()
    if args.size <= 0:
        raise ValueError("--size must be positive.")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative.")
    if args.preview_events < 0:
        raise ValueError("--preview-events must be non-negative.")

    cutracer_so = resolve_cutracer_so(args.cutracer_so)
    cutracer_bin = shutil.which("cutracer")
    if not cutracer_bin:
        raise RuntimeError("Could not find `cutracer` in PATH.")

    run_dir = timestamped_run_dir(args).resolve()
    raw_trace_dir = run_dir / "raw_trace"
    stats_path = run_dir / "addr_stats.json"
    preview_path = run_dir / "addr_preview.jsonl"
    summary_path = run_dir / "run_summary.json"
    raw_trace_dir.mkdir(parents=True, exist_ok=True)

    matmul_script = SCRIPT_DIR / "matmul_128.py"
    trace_command = [
        cutracer_bin,
        "trace",
        "--cutracer-so",
        str(cutracer_so),
        "-i",
        "mem_addr_trace",
        "--trace-format",
        "ndjson",
        "--kernel-events",
        "full",
        "--cpu-callstack",
        "auto",
        "--no-data-timeout-s",
        str(args.no_data_timeout_s),
    ]
    if args.trace_size_limit_mb > 0:
        trace_command.extend(["--trace-size-limit-mb", str(args.trace_size_limit_mb)])
    if args.no_dump_cubin:
        trace_command.append("--no-dump-cubin")
    trace_command.extend(
        [
            "--output-dir",
            str(raw_trace_dir),
            "--",
            args.python,
            str(matmul_script),
            "--size",
            str(args.size),
            "--dtype",
            args.dtype,
            "--warmup",
            str(args.warmup),
        ]
    )

    trace_env = os.environ.copy()
    trace_environment_overrides: dict[str, str] = {}
    if args.no_dump_cubin:
        trace_environment_overrides["CUTRACER_DUMP_CUBIN"] = "0"
    trace_env.update(trace_environment_overrides)

    trace_result = run_command(trace_command, cwd=REPO_ROOT, env=trace_env)
    if trace_result.returncode not in {0, 143, -15} or not raw_trace_has_data(raw_trace_dir):
        raise RuntimeError(
            "CUTracer did not produce the expected raw trace artifacts. "
            f"Exit code: {trace_result.returncode}"
        )

    markers = tuple(args.callstack_marker or DEFAULT_CALLSTACK_MARKERS)
    stats = summarize_memory_addresses(
        raw_trace_dir,
        stats_path,
        preview_path,
        markers=markers,
        include_all_kernels=args.include_all_kernels,
        preview_events=args.preview_events,
    )

    summary = {
        "size": args.size,
        "dtype": args.dtype,
        "warmup": args.warmup,
        "cutracer_so": str(cutracer_so),
        "run_dir": str(run_dir),
        "raw_trace_dir": str(raw_trace_dir),
        "addr_stats": str(stats_path),
        "addr_preview": str(preview_path),
        "trace_command": shlex.join(trace_command),
        "trace_environment_overrides": trace_environment_overrides,
        "trace_return_code": trace_result.returncode,
        "stats_digest": {
            "selection_mode": stats["selection_mode"],
            "matched_kernel_launches": stats["matched_kernel_launches"],
            "total_memory_events": stats["total_memory_events"],
            "total_addresses": stats["total_addresses"],
            "unique_addresses": stats["unique_addresses"],
            "address_pattern_histogram": stats["address_pattern_histogram"],
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("[done]")
    print(f"run_dir: {run_dir}")
    print(f"addr_stats: {stats_path}")
    print(f"addr_preview: {preview_path}")
    print(f"run_summary: {summary_path}")


if __name__ == "__main__":
    main()
