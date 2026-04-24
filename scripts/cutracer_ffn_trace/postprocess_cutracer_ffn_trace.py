import argparse
import heapq
import json
from pathlib import Path
from typing import Any

from common import default_processed_path, ensure_parent_dir


MEM_RECORD_TYPES = {"mem_trace", "mem_addr_trace"}
KERNEL_EVENT_TYPES = {"kernel_metadata", "kernel_launch"}
DEFAULT_CALLSTACK_MARKERS = ("replay_target_mlp_once", "replay_single_ffn_mlp.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter CUTracer raw traces down to the FFN replay memory-access sequence "
            "triggered by replay_single_ffn_mlp.py."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Trace files or directories containing CUTracer .ndjson trace files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Processed output path (.jsonl). Defaults to scripts/cutracer_ffn_trace/output/processed/...",
    )
    parser.add_argument(
        "--callstack-marker",
        action="append",
        default=None,
        help=(
            "Substring that must appear in kernel_metadata.cpu_callstack to mark the replay kernels. "
            "May be passed multiple times."
        ),
    )
    return parser.parse_args()


def iter_trace_paths(inputs: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for item in inputs:
        if item.is_dir():
            expanded.extend(sorted(path for path in item.glob("*.ndjson") if path.is_file()))
            continue
        if item.is_file():
            expanded.append(item)
            continue
        raise FileNotFoundError(f"Trace input does not exist: {item}")

    deduped = sorted({path.resolve() for path in expanded})
    if not deduped:
        raise FileNotFoundError("No CUTracer .ndjson trace files were found.")
    return deduped


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


def resolve_output_path(paths: list[Path], explicit_output: Path | None) -> Path:
    if explicit_output is not None:
        if explicit_output.suffix:
            return explicit_output
        return explicit_output / "ffn_mem_sequence.jsonl"

    first = paths[0]
    stem = first.parent.name if len(paths) > 1 else first.stem
    return default_processed_path(stem)


def metadata_matches(record: dict[str, Any], markers: tuple[str, ...]) -> bool:
    text = callstack_text(record)
    return bool(text) and any(marker in text for marker in markers)


def launch_name(record: dict[str, Any]) -> str | None:
    return record.get("unmangled_name") or record.get("kernel_name") or record.get("mangled_name")


def collect_matching_metadata(
    paths: list[Path], markers: tuple[str, ...]
) -> dict[int, dict[str, Any]]:
    matched: dict[int, dict[str, Any]] = {}
    fallback: dict[int, dict[str, Any]] = {}
    metadata_seen = 0

    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if record_type(record) not in KERNEL_EVENT_TYPES:
                    continue
                metadata_seen += 1
                kid = launch_id(record)
                if kid is None:
                    continue
                entry = {
                    "kernel_launch_id": kid,
                    "mangled_name": record.get("mangled_name"),
                    "unmangled_name": launch_name(record),
                    "cpu_callstack": record.get("cpu_callstack"),
                    "source_trace": str(path),
                    "source_line": line_no,
                }
                fallback[kid] = entry
                if metadata_matches(record, markers):
                    matched[kid] = entry

    if matched:
        return matched
    if metadata_seen == 0:
        raise RuntimeError("No kernel metadata or kernel launch records were found in the provided trace files.")
    if fallback:
        return fallback
    if not matched:
        marker_text = ", ".join(markers)
        raise RuntimeError(
            "No kernel event records matched the requested CPU callstack markers "
            f"({marker_text}). Make sure CUTracer was run with --kernel-events full "
            "and --cpu-callstack auto."
        )
    return matched


def iter_matching_mem_events_for_path(
    path: Path, metadata_by_launch_id: dict[int, dict[str, Any]]
):
    target_ids = set(metadata_by_launch_id)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record_type(record) not in MEM_RECORD_TYPES:
                continue
            kid = launch_id(record)
            if kid not in target_ids:
                continue

            meta = metadata_by_launch_id[kid]
            yield {
                "kernel_launch_id": kid,
                "trace_index": int(record["trace_index"]),
                "timestamp": int(record["timestamp"]),
                "mangled_name": meta.get("mangled_name"),
                "unmangled_name": meta.get("unmangled_name"),
                "cta": record.get("cta"),
                "warp": record.get("warp"),
                "pc": record.get("pc"),
                "sass": record.get("sass"),
                "addrs": record.get("addrs"),
                "source_trace": str(path),
            }


def merge_mem_event_streams(
    paths: list[Path], metadata_by_launch_id: dict[int, dict[str, Any]]
):
    heap: list[tuple[Any, ...]] = []
    iterators = []

    for index, path in enumerate(paths):
        iterator = iter_matching_mem_events_for_path(path, metadata_by_launch_id)
        iterators.append(iterator)
        first = next(iterator, None)
        if first is None:
            continue
        heapq.heappush(
            heap,
            (
                first["timestamp"],
                first["trace_index"],
                first["kernel_launch_id"],
                first["source_trace"],
                index,
                first,
            ),
        )

    while heap:
        _timestamp, _trace_index, _kid, _source_trace, index, event = heapq.heappop(heap)
        yield event
        nxt = next(iterators[index], None)
        if nxt is None:
            continue
        heapq.heappush(
            heap,
            (
                nxt["timestamp"],
                nxt["trace_index"],
                nxt["kernel_launch_id"],
                nxt["source_trace"],
                index,
                nxt,
            ),
        )


def write_events(event_iter, output_path: Path) -> tuple[Path, int]:
    ensure_parent_dir(output_path)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for index, event in enumerate(event_iter, start=1):
            payload = {"sequence_index": index, **event}
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count = index
    return output_path, count


def main() -> None:
    args = parse_args()
    paths = iter_trace_paths(args.inputs)
    markers = tuple(args.callstack_marker or DEFAULT_CALLSTACK_MARKERS)
    metadata_by_launch_id = collect_matching_metadata(paths, markers)
    event_iter = merge_mem_event_streams(paths, metadata_by_launch_id)
    output_path, event_count = write_events(event_iter, resolve_output_path(paths, args.output))
    if event_count == 0:
        raise RuntimeError("No memory-address records matched the selected replay kernels.")

    print(f"trace files scanned: {len(paths)}")
    print(f"matched kernel launches: {len(metadata_by_launch_id)}")
    print(f"memory events written: {event_count}")
    print(f"saved processed trace to: {output_path}")


if __name__ == "__main__":
    main()
