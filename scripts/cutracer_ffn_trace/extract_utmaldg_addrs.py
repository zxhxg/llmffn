from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


TARGET_MNEMONICS = {"UTMALDG.4D", "UTMALDG.4D.MULTICAST"}
DEFAULT_ADDRS_OUTPUT = "utmaldg_addrs.ndjson"
DEFAULT_SUMMARY_OUTPUT = "summary.json"


@dataclass
class ParseStats:
    skipped_decode_errors: int = 0
    skipped_json_errors: int = 0
    skipped_non_object_records: int = 0


@dataclass
class InstructionStats:
    file: str
    launch_id: int | None
    kernel_checksum: str | None
    opcode_id: int
    mnemonic: str
    sass: str
    record_count: int = 0
    addr_item_count: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "launch_id": self.launch_id,
            "kernel_checksum": self.kernel_checksum,
            "opcode_id": self.opcode_id,
            "mnemonic": self.mnemonic,
            "sass": self.sass,
            "record_count": self.record_count,
            "addr_item_count": self.addr_item_count,
        }


@dataclass
class KernelFileStats:
    path: Path
    kernel_checksum: str | None = None
    launch_id: int | None = None
    target_instructions: dict[int, InstructionStats] = field(default_factory=dict)
    record_count: int = 0
    addr_item_count: int = 0
    skipped_matching_records_without_addrs: int = 0
    skipped_matching_records_with_non_array_addrs: int = 0
    parse_stats: ParseStats = field(default_factory=ParseStats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract addrs arrays for UTMALDG.4D and UTMALDG.4D.MULTICAST "
            "records from CUTracer kernel*.ndjson files."
        )
    )
    parser.add_argument(
        "raw_trace_dir",
        type=Path,
        help="Directory containing CUTracer kernel*.ndjson files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output NDJSON path. Each line is one addrs array. Defaults to "
            "<raw_trace_dir>/utmaldg_addrs.ndjson."
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Summary JSON path. Defaults to <raw_trace_dir>/summary.json.",
    )
    return parser.parse_args()


def iter_json_objects(path: Path, stats: ParseStats):
    with path.open("rb") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                stats.skipped_decode_errors += 1
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats.skipped_json_errors += 1
                continue

            if not isinstance(record, dict):
                stats.skipped_non_object_records += 1
                continue

            yield line_no, record


def record_type(record: dict[str, Any]) -> str | None:
    value = record.get("type") or record.get("message_type")
    return str(value) if value is not None else None


def parse_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def launch_id(record: dict[str, Any]) -> int | None:
    return parse_int(record.get("grid_launch_id"))


def opcode_id(record: dict[str, Any]) -> int | None:
    return parse_int(record.get("opcode_id"))


def sass_mnemonic(sass: str) -> str:
    parts = sass.strip().split(None, 1)
    if not parts:
        return ""
    return parts[0].rstrip(";")


def extract_target_instructions(
    metadata: dict[str, Any], relative_file: str
) -> dict[int, InstructionStats]:
    instructions = metadata.get("instructions") or {}
    if not isinstance(instructions, dict):
        return {}

    kernel_checksum = metadata.get("kernel_checksum")
    targets: dict[int, InstructionStats] = {}
    for raw_opcode, entry in instructions.items():
        if not isinstance(entry, dict):
            continue
        oid = parse_int(raw_opcode)
        if oid is None:
            continue
        sass = entry.get("sass") or ""
        if not isinstance(sass, str):
            continue
        mnemonic = sass_mnemonic(sass)
        if mnemonic not in TARGET_MNEMONICS:
            continue
        targets[oid] = InstructionStats(
            file=relative_file,
            launch_id=None,
            kernel_checksum=str(kernel_checksum) if kernel_checksum is not None else None,
            opcode_id=oid,
            mnemonic=mnemonic,
            sass=sass,
        )
    return targets


def iter_kernel_paths(raw_trace_dir: Path) -> list[Path]:
    paths = sorted(
        path for path in raw_trace_dir.glob("kernel*.ndjson") if path.is_file()
    )
    if not paths:
        raise FileNotFoundError(f"No kernel*.ndjson files found under {raw_trace_dir}")
    return paths


def ensure_outputs_do_not_overwrite_inputs(
    input_paths: list[Path], output_path: Path, summary_output: Path
) -> None:
    resolved_inputs = {path.resolve() for path in input_paths}
    resolved_output = output_path.resolve()
    resolved_summary = summary_output.resolve()
    if resolved_output in resolved_inputs:
        raise ValueError(f"Refusing to overwrite input trace file: {output_path}")
    if resolved_summary in resolved_inputs:
        raise ValueError(f"Refusing to overwrite input trace file: {summary_output}")
    if resolved_output == resolved_summary:
        raise ValueError("--output and --summary-output must be different paths")


def analyze_kernel_file(path: Path, raw_trace_dir: Path) -> KernelFileStats:
    relative_file = path.relative_to(raw_trace_dir).as_posix()
    file_stats = KernelFileStats(path=path)
    seen_metadata = False
    all_launch_ids: set[int] = set()
    matching_launch_ids: set[int] = set()

    for _line_no, record in iter_json_objects(path, file_stats.parse_stats):
        lid = launch_id(record)
        if lid is not None:
            all_launch_ids.add(lid)

        if not seen_metadata and record_type(record) == "kernel_metadata":
            seen_metadata = True
            kernel_checksum = record.get("kernel_checksum")
            file_stats.kernel_checksum = (
                str(kernel_checksum) if kernel_checksum is not None else None
            )
            file_stats.target_instructions = extract_target_instructions(record, relative_file)

        if not file_stats.target_instructions:
            continue

        oid = opcode_id(record)
        if oid not in file_stats.target_instructions:
            continue

        if "addrs" not in record:
            file_stats.skipped_matching_records_without_addrs += 1
            continue

        addrs = record.get("addrs")
        if not isinstance(addrs, list):
            file_stats.skipped_matching_records_with_non_array_addrs += 1
            continue

        if lid is None:
            raise ValueError(f"Target record in {path} is missing grid_launch_id")
        matching_launch_ids.add(lid)
        if len(matching_launch_ids) > 1:
            ids = ", ".join(str(value) for value in sorted(matching_launch_ids))
            raise ValueError(f"Target records in {path} contain multiple grid_launch_id values: {ids}")

        inst = file_stats.target_instructions[oid]
        inst.record_count += 1
        inst.addr_item_count += len(addrs)
        file_stats.record_count += 1
        file_stats.addr_item_count += len(addrs)

    if matching_launch_ids:
        file_stats.launch_id = next(iter(matching_launch_ids))
    elif len(all_launch_ids) == 1:
        file_stats.launch_id = next(iter(all_launch_ids))

    for inst in file_stats.target_instructions.values():
        inst.launch_id = file_stats.launch_id

    return file_stats


def sort_key(file_stats: KernelFileStats) -> tuple[int, str]:
    launch_sort = file_stats.launch_id if file_stats.launch_id is not None else sys.maxsize
    return (launch_sort, file_stats.path.name)


def write_addrs_output(file_stats: list[KernelFileStats], output_path: Path) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written_records = 0
    written_addr_items = 0

    with output_path.open("w", encoding="utf-8") as output:
        for stats in sorted(file_stats, key=sort_key):
            if not stats.target_instructions or stats.record_count == 0:
                continue

            target_opcodes = set(stats.target_instructions)
            parse_stats = ParseStats()
            for _line_no, record in iter_json_objects(stats.path, parse_stats):
                oid = opcode_id(record)
                if oid not in target_opcodes:
                    continue
                addrs = record.get("addrs")
                if not isinstance(addrs, list):
                    continue
                payload = json.dumps(addrs, ensure_ascii=False, separators=(",", ":"))
                output.write(payload + "\n")
                written_records += 1
                written_addr_items += len(addrs)

    return written_records, written_addr_items


def build_summary(
    raw_trace_dir: Path,
    output_path: Path,
    summary_output: Path,
    file_stats: list[KernelFileStats],
    written_records: int,
    written_addr_items: int,
) -> dict[str, Any]:
    instructions = []
    for stats in sorted(file_stats, key=sort_key):
        instructions.extend(
            inst.to_json()
            for inst in sorted(stats.target_instructions.values(), key=lambda item: item.opcode_id)
        )

    skipped_decode_errors = sum(stats.parse_stats.skipped_decode_errors for stats in file_stats)
    skipped_json_errors = sum(stats.parse_stats.skipped_json_errors for stats in file_stats)
    skipped_non_object_records = sum(
        stats.parse_stats.skipped_non_object_records for stats in file_stats
    )
    skipped_without_addrs = sum(stats.skipped_matching_records_without_addrs for stats in file_stats)
    skipped_non_array_addrs = sum(
        stats.skipped_matching_records_with_non_array_addrs for stats in file_stats
    )

    return {
        "input_dir": str(raw_trace_dir),
        "output": str(output_path),
        "summary_output": str(summary_output),
        "kernel_files_scanned": len(file_stats),
        "kernel_files_with_target_instructions": sum(
            1 for stats in file_stats if stats.target_instructions
        ),
        "records_written": written_records,
        "addr_items_written": written_addr_items,
        "skipped_decode_errors": skipped_decode_errors,
        "skipped_json_errors": skipped_json_errors,
        "skipped": {
            "decode_errors": skipped_decode_errors,
            "json_errors": skipped_json_errors,
            "non_object_records": skipped_non_object_records,
            "matching_records_without_addrs": skipped_without_addrs,
            "matching_records_with_non_array_addrs": skipped_non_array_addrs,
        },
        "instructions": instructions,
    }


def warn_if_skipped(summary: dict[str, Any]) -> None:
    skipped = summary["skipped"]
    skipped_total = sum(int(value) for value in skipped.values())
    if skipped_total == 0:
        return
    details = ", ".join(f"{key}={value}" for key, value in skipped.items() if value)
    print(f"[warn] skipped malformed or unusable records: {details}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    raw_trace_dir = args.raw_trace_dir.resolve()
    if not raw_trace_dir.is_dir():
        raise NotADirectoryError(f"Input is not a directory: {raw_trace_dir}")

    output_path = (args.output or raw_trace_dir / DEFAULT_ADDRS_OUTPUT).resolve()
    summary_output = (args.summary_output or raw_trace_dir / DEFAULT_SUMMARY_OUTPUT).resolve()
    paths = iter_kernel_paths(raw_trace_dir)
    ensure_outputs_do_not_overwrite_inputs(paths, output_path, summary_output)

    file_stats = [analyze_kernel_file(path, raw_trace_dir) for path in paths]
    expected_records = sum(stats.record_count for stats in file_stats)
    expected_addr_items = sum(stats.addr_item_count for stats in file_stats)

    written_records, written_addr_items = write_addrs_output(file_stats, output_path)
    if written_records != expected_records or written_addr_items != expected_addr_items:
        raise RuntimeError(
            "Output write count mismatch: "
            f"expected_records={expected_records}, written_records={written_records}, "
            f"expected_addr_items={expected_addr_items}, written_addr_items={written_addr_items}"
        )

    summary = build_summary(
        raw_trace_dir=raw_trace_dir,
        output_path=output_path,
        summary_output=summary_output,
        file_stats=file_stats,
        written_records=written_records,
        written_addr_items=written_addr_items,
    )
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    warn_if_skipped(summary)

    print(f"input_dir: {raw_trace_dir}")
    print(f"kernel_files_scanned: {len(paths)}")
    print(f"target_instruction_rows: {len(summary['instructions'])}")
    print(f"records_written: {written_records}")
    print(f"addr_items_written: {written_addr_items}")
    print(f"output: {output_path}")
    print(f"summary_output: {summary_output}")


if __name__ == "__main__":
    main()
