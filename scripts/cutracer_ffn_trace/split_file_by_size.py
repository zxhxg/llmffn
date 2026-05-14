from __future__ import annotations

import argparse
import re
from pathlib import Path


DEFAULT_MAX_SIZE = "1GiB"
DEFAULT_NAME_PATTERN = "{stem}.part{index:05d}{suffix}"


SIZE_UNITS = {
    "": 1,
    "b": 1,
    "byte": 1,
    "bytes": 1,
    "k": 1024,
    "kb": 1000,
    "kib": 1024,
    "m": 1024**2,
    "mb": 1000**2,
    "mib": 1024**2,
    "g": 1024**3,
    "gb": 1000**3,
    "gib": 1024**3,
    "t": 1024**4,
    "tb": 1000**4,
    "tib": 1024**4,
}


def parse_size(value: str) -> int:
    text = value.strip().lower()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([a-z]*)", text)
    if not match:
        raise argparse.ArgumentTypeError(
            f"Invalid size {value!r}. Examples: 1000000, 512MiB, 2GB."
        )

    number_text, unit = match.groups()
    if unit not in SIZE_UNITS:
        raise argparse.ArgumentTypeError(
            f"Invalid size unit {unit!r}. Supported units include B, KB, KiB, MB, MiB, GB, GiB."
        )

    size = int(float(number_text) * SIZE_UNITS[unit])
    if size <= 0:
        raise argparse.ArgumentTypeError("--max-size must be greater than zero.")
    return size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a large file into size-limited parts without breaking lines. "
            "Input bytes are copied as-is."
        )
    )
    parser.add_argument("input", type=Path, help="Input file to split.")
    parser.add_argument(
        "--max-size",
        type=parse_size,
        default=parse_size(DEFAULT_MAX_SIZE),
        help=(
            "Maximum target size for each output part. Supports bytes and units such as "
            "512MiB, 2GB, 1GiB. Defaults to 1GiB. If one line is larger than this, "
            "that line is written intact to a larger single part."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for split files. Defaults to the input file directory.",
    )
    parser.add_argument(
        "--name-pattern",
        default=DEFAULT_NAME_PATTERN,
        help=(
            "Output filename pattern. Available fields: {stem}, {suffix}, {name}, {index}. "
            "Defaults to '{stem}.part{index:05d}{suffix}'."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing split files that match generated output names.",
    )
    return parser.parse_args()


def input_stem_and_suffix(path: Path) -> tuple[str, str]:
    suffix = "".join(path.suffixes)
    if suffix:
        return path.name[: -len(suffix)], suffix
    return path.name, ""


def output_path(input_path: Path, output_dir: Path, pattern: str, index: int) -> Path:
    stem, suffix = input_stem_and_suffix(input_path)
    try:
        filename = pattern.format(
            stem=stem,
            suffix=suffix,
            name=input_path.name,
            index=index,
        )
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(f"Invalid --name-pattern {pattern!r}: {exc}") from exc

    if not filename:
        raise ValueError("--name-pattern produced an empty filename.")

    candidate = output_dir / filename
    if candidate.name != filename:
        raise ValueError("--name-pattern must produce a filename, not a path.")
    return candidate


def open_part(
    input_path: Path,
    output_dir: Path,
    pattern: str,
    index: int,
    overwrite: bool,
):
    path = output_path(input_path, output_dir, pattern, index)
    if path.resolve() == input_path.resolve():
        raise ValueError(f"Refusing to overwrite input file: {path}")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {path}")
    return path, path.open("wb")


def split_file(
    input_path: Path,
    output_dir: Path,
    pattern: str,
    max_size: int,
    overwrite: bool,
) -> tuple[list[Path], int, int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)

    part_index = 1
    part_size = 0
    part_paths: list[Path] = []
    current_path, current_file = open_part(
        input_path, output_dir, pattern, part_index, overwrite
    )
    part_paths.append(current_path)

    line_count = 0
    oversized_line_count = 0
    total_bytes = 0

    try:
        with input_path.open("rb") as src:
            for line in src:
                line_size = len(line)
                if line_size > max_size:
                    oversized_line_count += 1

                if part_size > 0 and part_size + line_size > max_size:
                    current_file.close()
                    part_index += 1
                    current_path, current_file = open_part(
                        input_path, output_dir, pattern, part_index, overwrite
                    )
                    part_paths.append(current_path)
                    part_size = 0

                current_file.write(line)
                part_size += line_size
                total_bytes += line_size
                line_count += 1
    finally:
        current_file.close()

    if line_count == 0:
        part_paths[0].touch()

    return part_paths, line_count, total_bytes, oversized_line_count


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    output_dir = (args.output_dir or input_path.parent).resolve()
    part_paths, line_count, total_bytes, oversized_line_count = split_file(
        input_path=input_path,
        output_dir=output_dir,
        pattern=args.name_pattern,
        max_size=args.max_size,
        overwrite=args.overwrite,
    )

    print(f"input: {input_path}")
    print(f"output_dir: {output_dir}")
    print(f"max_size_bytes: {args.max_size}")
    print(f"parts_written: {len(part_paths)}")
    print(f"lines_written: {line_count}")
    print(f"bytes_written: {total_bytes}")
    print(f"oversized_lines: {oversized_line_count}")
    for path in part_paths:
        print(f"part: {path} size_bytes={path.stat().st_size}")


if __name__ == "__main__":
    main()
