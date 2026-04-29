from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_CUTRACER_ROOT,
    DEFAULT_RUNS_DIR,
    SCRIPT_DIR,
    ensure_parent_dir,
    resolve_default_model_id,
)


DEFAULT_PROMPT = "Explain briefly what the FFN layer does in a transformer."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full CUTracer FFN workflow end-to-end: capture the FFN input, "
            "replay the target MLP under CUTracer, and postprocess the raw trace."
        ),
    )
    parser.add_argument("--model-id", type=str, default=resolve_default_model_id())
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument(
        "--device-map",
        choices=["cuda", "auto"],
        default="auto",
        help=(
            "Model placement strategy used by both capture and replay. "
            "Default uses auto because the full model may not fit on smaller GPUs."
        ),
    )
    parser.add_argument(
        "--preferred-blas",
        choices=["default", "cublas", "cublaslt"],
        default="cublas",
        help=(
            "Preferred CUDA BLAS backend passed to capture/replay via "
            "torch.backends.cuda.preferred_blas_library. Default cublas tries "
            "to avoid cuBLASLt where PyTorch supports that override."
        ),
    )
    parser.add_argument(
        "--replay-mode",
        choices=["mlp", "weight-scan"],
        default="mlp",
        help=(
            "Replay implementation wrapped by CUTracer. 'mlp' runs the real target_mlp call; "
            "'weight-scan' runs explicit LDG scans over gate/up/down weights to record "
            "complete weight global-memory addresses."
        ),
    )
    parser.add_argument(
        "--weight-scan-blocks",
        type=int,
        default=4096,
        help="CUDA blocks per projection when --replay-mode weight-scan is used.",
    )
    parser.add_argument(
        "--cutracer-so",
        type=Path,
        default=None,
        help="Explicit path to cutracer.so. If omitted, the script tries CUTRACER_SO and common local build paths.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run directory name. Default is layer_<N>_<timestamp>.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Root directory where one-click run artifacts are stored.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to run the helper scripts.",
    )
    parser.add_argument(
        "--callstack-marker",
        action="append",
        default=None,
        help="Optional marker passed through to the postprocess script. May be used multiple times.",
    )
    parser.add_argument(
        "--no-data-timeout-s",
        type=int,
        default=15,
        help=(
            "Forwarded to cutracer trace as --no-data-timeout-s. "
            "This workflow tolerates timeout-based termination if raw trace files were already produced."
        ),
    )
    parser.add_argument(
        "--trace-size-limit-mb",
        type=int,
        default=0,
        help=(
            "Forwarded to cutracer trace as --trace-size-limit-mb when positive. "
            "Use this to cap very large mem_addr_trace outputs. Default 0 leaves CUTracer unlimited."
        ),
    )
    parser.add_argument(
        "--no-dump-cubin",
        action="store_true",
        help=(
            "Set CUTRACER_DUMP_CUBIN=0 for the trace step. "
            "This avoids huge kernel_*.cubin files on H100/sm90 runs."
        ),
    )
    parser.add_argument(
        "--processed-preview-lines",
        type=int,
        default=10000,
        help=(
            "Maximum number of lines copied from processed_ffn_mem_sequence.jsonl "
            "into processed_preview.jsonl."
        ),
    )
    parser.add_argument(
        "--delete-raw-trace-after-postprocess",
        action="store_true",
        help=(
            "Delete the raw_trace directory after postprocess succeeds. "
            "The run summary keeps the raw trace size/count statistics collected before deletion."
        ),
    )
    return parser.parse_args()


def repo_root() -> Path:
    return SCRIPT_DIR.parent.parent


def resolve_helper_script(name: str) -> Path:
    candidates = [
        SCRIPT_DIR / name,
        repo_root() / "scripts" / "statistic" / name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find helper script {name}. Searched: {searched}")


def build_run_dir(args: argparse.Namespace) -> Path:
    if args.run_name:
        return args.output_root / args.run_name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return args.output_root / f"layer_{args.layer}_{stamp}"


def build_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "run_dir": run_dir,
        "capture": run_dir / "capture.pt",
        "raw_trace_dir": run_dir / "raw_trace",
        "processed": run_dir / "processed_ffn_mem_sequence.jsonl",
        "processed_preview": run_dir / "processed_preview.jsonl",
        "summary": run_dir / "run_summary.json",
    }


def print_block(title: str, content: str) -> None:
    print(f"\n[{title}]")
    if content.strip():
        print(content.rstrip())
    else:
        print("(no output)")


def run_command(
    cmd: list[str],
    cwd: Path,
    step_name: str,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        env=env,
    )
    print_block(f"{step_name} command", shlex.join(cmd))
    if result.stdout:
        print_block(f"{step_name} stdout", result.stdout)
    if result.stderr:
        print_block(f"{step_name} stderr", result.stderr)
    return result


def build_child_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("TORCH_BLAS_PREFER_CUBLASLT", None)
    return env


def ensure_success(result: subprocess.CompletedProcess[str], step_name: str) -> None:
    if result.returncode != 0:
        raise RuntimeError(f"{step_name} failed with exit code {result.returncode}.")


def resolve_cutracer_so(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)

    env_path = os.environ.get("CUTRACER_SO")
    if env_path:
        candidates.append(Path(env_path))

    candidates.extend(
        [
            DEFAULT_CUTRACER_ROOT / "lib" / "cutracer.so",
            repo_root() / "lib" / "cutracer.so",
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find cutracer.so. Pass --cutracer-so or set CUTRACER_SO, "
        "or build CUTracer locally. Searched: "
        f"{searched}"
    )


def count_raw_ndjson_files(raw_trace_dir: Path) -> tuple[int, int]:
    kernel_files = list(raw_trace_dir.glob("kernel_*.ndjson"))
    event_files = list(raw_trace_dir.glob("cutracer_kernel_events_*.ndjson"))
    return len(kernel_files), len(event_files)


def cutracer_trace_succeeded(result: subprocess.CompletedProcess[str], raw_trace_dir: Path) -> bool:
    kernel_count, event_count = count_raw_ndjson_files(raw_trace_dir)
    if result.returncode == 0:
        return kernel_count > 0 and event_count > 0
    if result.returncode in {143, -15}:
        return kernel_count > 0 and event_count > 0
    return False


def file_size_or_none(path: Path) -> int | None:
    return path.stat().st_size if path.exists() else None


def write_processed_preview(processed_path: Path, preview_path: Path, max_lines: int) -> dict[str, Any]:
    if max_lines <= 0:
        raise ValueError("--processed-preview-lines must be a positive integer.")

    ensure_parent_dir(preview_path)
    lines_written = 0
    total_lines_seen = 0

    with processed_path.open("r", encoding="utf-8") as src, preview_path.open("w", encoding="utf-8") as dst:
        for line in src:
            total_lines_seen += 1
            if lines_written < max_lines:
                dst.write(line)
                lines_written += 1

    return {
        "preview_line_limit": max_lines,
        "preview_line_count": lines_written,
        "source_total_line_count": total_lines_seen,
        "truncated": total_lines_seen > lines_written,
    }


def summarize_raw_trace_dir(raw_trace_dir: Path) -> dict[str, Any]:
    summary = {
        "file_count": 0,
        "total_size_bytes": 0,
        "kernel_ndjson_count": 0,
        "kernel_ndjson_size_bytes": 0,
        "kernel_events_count": 0,
        "kernel_events_size_bytes": 0,
        "cubin_count": 0,
        "cubin_size_bytes": 0,
        "log_count": 0,
        "log_size_bytes": 0,
        "largest_files": [],
    }
    largest: list[tuple[int, str]] = []

    for path in sorted(raw_trace_dir.glob("*")):
        if not path.is_file():
            continue
        size = path.stat().st_size
        summary["file_count"] += 1
        summary["total_size_bytes"] += size
        largest.append((size, path.name))

        name = path.name
        if name.startswith("kernel_") and name.endswith(".ndjson"):
            summary["kernel_ndjson_count"] += 1
            summary["kernel_ndjson_size_bytes"] += size
        elif name.startswith("cutracer_kernel_events_") and name.endswith(".ndjson"):
            summary["kernel_events_count"] += 1
            summary["kernel_events_size_bytes"] += size
        elif name.endswith(".cubin"):
            summary["cubin_count"] += 1
            summary["cubin_size_bytes"] += size
        elif name.endswith(".log"):
            summary["log_count"] += 1
            summary["log_size_bytes"] += size

    summary["largest_files"] = [
        {"name": name, "size_bytes": size}
        for size, name in sorted(largest, reverse=True)[:10]
    ]
    return summary


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent_dir(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_memory_events_written(stdout: str) -> int | None:
    for line in stdout.splitlines():
        prefix = "memory events written: "
        if line.startswith(prefix):
            try:
                return int(line[len(prefix) :].strip())
            except ValueError:
                return None
    return None


def main() -> None:
    args = parse_args()
    cutracer_so = resolve_cutracer_so(args.cutracer_so)
    cutracer_bin = shutil.which("cutracer")
    if not cutracer_bin:
        raise RuntimeError("Could not find `cutracer` in PATH.")

    run_dir = build_run_dir(args)
    paths = build_paths(run_dir)
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    paths["raw_trace_dir"].mkdir(parents=True, exist_ok=True)

    capture_script = resolve_helper_script("capture_first_generated_ffn_input.py")
    replay_script_name = (
        "replay_single_ffn_weight_scan.py"
        if args.replay_mode == "weight-scan"
        else "replay_single_ffn_mlp.py"
    )
    replay_script = resolve_helper_script(replay_script_name)
    postprocess_script = resolve_helper_script("postprocess_cutracer_ffn_trace.py")

    capture_cmd = [
        args.python,
        str(capture_script),
        "--model-id",
        args.model_id,
        "--layer",
        str(args.layer),
        "--prompt",
        args.prompt,
        "--device-map",
        args.device_map,
        "--output",
        str(paths["capture"]),
        "--preferred-blas",
        args.preferred_blas,
    ]
    capture_env = build_child_env()
    capture_result = run_command(capture_cmd, repo_root(), "capture", env=capture_env)
    ensure_success(capture_result, "capture")

    compile_cmd: list[str] | None = None
    compile_result: subprocess.CompletedProcess[str] | None = None
    if args.replay_mode == "weight-scan":
        compile_cmd = [
            args.python,
            str(replay_script),
            "--compile-only",
            "--preferred-blas",
            args.preferred_blas,
        ]
        compile_result = run_command(compile_cmd, repo_root(), "weight-scan compile", env=build_child_env())
        ensure_success(compile_result, "weight-scan compile")

    trace_cmd = [
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
        trace_cmd.extend(["--trace-size-limit-mb", str(args.trace_size_limit_mb)])
    if args.no_dump_cubin:
        trace_cmd.append("--no-dump-cubin")
    trace_cmd.extend(
        [
            "--output-dir",
            str(paths["raw_trace_dir"]),
            "--",
            args.python,
            str(replay_script),
            "--capture",
            str(paths["capture"]),
            "--device-map",
            args.device_map,
            "--preferred-blas",
            args.preferred_blas,
        ]
    )
    if args.replay_mode == "weight-scan":
        trace_cmd.extend(["--blocks", str(args.weight_scan_blocks)])
    trace_env = build_child_env()
    trace_environment_overrides: dict[str, str] = {}
    if os.environ.get("TORCH_BLAS_PREFER_CUBLASLT") is not None:
        trace_environment_overrides["TORCH_BLAS_PREFER_CUBLASLT"] = "<unset>"
    if args.no_dump_cubin:
        trace_environment_overrides["CUTRACER_DUMP_CUBIN"] = "0"
    for key, value in trace_environment_overrides.items():
        if value == "<unset>":
            trace_env.pop(key, None)
        else:
            trace_env[key] = value

    trace_result = run_command(trace_cmd, repo_root(), "trace", env=trace_env)
    if not cutracer_trace_succeeded(trace_result, paths["raw_trace_dir"]):
        raise RuntimeError(
            "cutracer trace did not produce the expected raw trace artifacts. "
            f"Exit code was {trace_result.returncode}."
        )

    postprocess_cmd = [
        args.python,
        str(postprocess_script),
        str(paths["raw_trace_dir"]),
        "--output",
        str(paths["processed"]),
    ]
    for marker in args.callstack_marker or []:
        postprocess_cmd.extend(["--callstack-marker", marker])

    postprocess_result = run_command(postprocess_cmd, repo_root(), "postprocess")
    ensure_success(postprocess_result, "postprocess")
    processed_preview_summary = write_processed_preview(
        paths["processed"],
        paths["processed_preview"],
        args.processed_preview_lines,
    )
    raw_trace_summary = summarize_raw_trace_dir(paths["raw_trace_dir"])
    raw_trace_deleted = False
    if args.delete_raw_trace_after_postprocess:
        shutil.rmtree(paths["raw_trace_dir"])
        raw_trace_deleted = True

    summary = {
        "model_id": args.model_id,
        "layer": args.layer,
        "prompt": args.prompt,
        "device_map": args.device_map,
        "preferred_blas": args.preferred_blas,
        "replay_mode": args.replay_mode,
        "weight_scan_blocks": args.weight_scan_blocks if args.replay_mode == "weight-scan" else None,
        "cutracer_so": str(cutracer_so),
        "paths": {key: str(value) for key, value in paths.items()},
        "commands": {
            "capture": shlex.join(capture_cmd),
            "weight_scan_compile": shlex.join(compile_cmd) if compile_cmd is not None else None,
            "trace": shlex.join(trace_cmd),
            "postprocess": shlex.join(postprocess_cmd),
        },
        "trace_environment_overrides": trace_environment_overrides,
        "return_codes": {
            "capture": capture_result.returncode,
            "weight_scan_compile": compile_result.returncode if compile_result is not None else None,
            "trace": trace_result.returncode,
            "postprocess": postprocess_result.returncode,
        },
        "artifacts": {
            "capture_size_bytes": file_size_or_none(paths["capture"]),
            "processed_size_bytes": file_size_or_none(paths["processed"]),
            "processed_preview_size_bytes": file_size_or_none(paths["processed_preview"]),
            "processed_preview_summary": processed_preview_summary,
            "raw_trace_summary": raw_trace_summary,
            "raw_trace_deleted_after_postprocess": raw_trace_deleted,
            "processed_event_count": parse_memory_events_written(postprocess_result.stdout),
        },
    }
    write_summary(paths["summary"], summary)

    print("\n[done]")
    print(f"run directory: {paths['run_dir']}")
    print(f"capture: {paths['capture']}")
    print(f"raw trace dir: {paths['raw_trace_dir']}")
    print(f"processed: {paths['processed']}")
    print(f"processed preview: {paths['processed_preview']}")
    print(f"summary: {paths['summary']}")


if __name__ == "__main__":
    main()
