from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_MAX_BYTES = 1024 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read a processed_ffn_mem_sequence.jsonl file, extract the addrs field "
            "from each line, and split the output into multiple JSONL files."
        )
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to processed_ffn_mem_sequence.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("local.jsonl"),
        help=(
            "Base output JSONL path. The script writes numbered shard files such as "
            "local.part00001.jsonl."
        ),
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=DEFAULT_MAX_BYTES,
        help="Maximum size in bytes for each output shard. Defaults to 1 GiB.",
    )
    return parser.parse_args()


def shard_path(base_output: Path, shard_index: int) -> Path:
    suffix = "".join(base_output.suffixes)
    stem = base_output.name[: -len(suffix)] if suffix else base_output.name
    filename = f"{stem}.part{shard_index:05d}{suffix or '.jsonl'}"
    return base_output.with_name(filename)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if args.max_bytes <= 0:
        raise ValueError("--max-bytes must be a positive integer.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    line_count = 0
    shard_index = 1
    shard_paths: list[Path] = []
    current_path = shard_path(output_path, shard_index)
    current_size = 0
    dst = current_path.open("w", encoding="utf-8")
    shard_paths.append(current_path)

    with input_path.open("r", encoding="utf-8") as src:
        for raw_line in src:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            record = json.loads(raw_line)
            payload = json.dumps(record.get("addrs", []), ensure_ascii=False) + "\n"
            payload_size = len(payload.encode("utf-8"))

            if current_size > 0 and current_size + payload_size > args.max_bytes:
                dst.close()
                shard_index += 1
                current_path = shard_path(output_path, shard_index)
                current_size = 0
                dst = current_path.open("w", encoding="utf-8")
                shard_paths.append(current_path)

            dst.write(payload)
            current_size += payload_size
            line_count += 1

    dst.close()

    print(f"input: {input_path}")
    print(f"output_base: {output_path}")
    print(f"shards_written: {len(shard_paths)}")
    print(f"written_lines: {line_count}")
    for path in shard_paths:
        print(f"shard: {path} size_bytes={path.stat().st_size}")


if __name__ == "__main__":
    main()
