
import math
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class MemoryConfig:
    name: str
    bandwidth_GBps: float
    page_size_B: int = 0
    startup_ns: float = 0.0
    small_run_bw_penalty: float = 1.0

@dataclass
class FFNConfig:
    d: int
    m: int
    bytes_per_weight: float

def _runs_from_sorted_indices(sorted_ids: List[int]) -> int:
    if not sorted_ids:
        return 0
    runs = 1
    prev = sorted_ids[0]
    for x in sorted_ids[1:]:
        if x != prev + 1:
            runs += 1
        prev = x
    return runs

def _page_aligned_chunk_bytes(chunk_bytes: float, page_size_B: int) -> int:
    if page_size_B <= 0:
        return int(math.ceil(chunk_bytes))
    return int(math.ceil(chunk_bytes / page_size_B) * page_size_B)

def dense_payload_bytes(cfg: FFNConfig) -> float:
    return 2.0 * cfg.d * cfg.m * cfg.bytes_per_weight

def sparse_payload_bytes_ideal(cfg: FFNConfig, active_count: int) -> float:
    return 2.0 * cfg.d * active_count * cfg.bytes_per_weight

def sparse_physical_bytes_dense_layout_hbf(cfg: FFNConfig, active_ids: List[int], page_size_B: int) -> Dict[str, float]:
    k = len(active_ids)
    s = cfg.bytes_per_weight
    d = cfg.d
    m = cfg.m
    P = page_size_B

    per_neuron_chunk = _page_aligned_chunk_bytes(d * s, P)
    down_bytes = k * per_neuron_chunk

    active_sorted = sorted(set(active_ids))
    down_runs = _runs_from_sorted_indices(active_sorted)

    row_bytes_up = m * s
    pages_per_row = math.ceil(row_bytes_up / P)
    if k == 0:
        pages_touched_per_row = 0.0
    else:
        pages_touched_per_row = pages_per_row * (1.0 - ((1.0 - 1.0 / pages_per_row) ** k))
    up_bytes = d * pages_touched_per_row * P
    up_runs = int(math.ceil(d * pages_touched_per_row))

    return {
        "up_bytes": up_bytes,
        "down_bytes": down_bytes,
        "total_bytes": up_bytes + down_bytes,
        "up_runs": up_runs,
        "down_runs": down_runs,
        "total_runs": up_runs + down_runs,
        "per_neuron_chunk_B": per_neuron_chunk,
        "pages_per_row_up": pages_per_row,
        "expected_pages_touched_per_row_up": pages_touched_per_row,
    }

def sparse_physical_bytes_relayout_hbf(cfg: FFNConfig, active_ids: List[int], page_size_B: int) -> Dict[str, float]:
    k = len(active_ids)
    P = page_size_B
    active_sorted = sorted(set(active_ids))
    runs = _runs_from_sorted_indices(active_sorted)
    per_neuron_chunk = _page_aligned_chunk_bytes(cfg.d * cfg.bytes_per_weight, P)
    total_bytes = 2 * k * per_neuron_chunk
    return {
        "total_bytes": total_bytes,
        "up_bytes": k * per_neuron_chunk,
        "down_bytes": k * per_neuron_chunk,
        "runs_each_matrix": runs,
        "total_runs": 2 * runs,
        "per_neuron_chunk_B": per_neuron_chunk,
    }

def time_seconds(payload_bytes: float, runs: int, mem: MemoryConfig, small_run: bool = False) -> float:
    bw = mem.bandwidth_GBps * 1e9
    if small_run:
        bw *= mem.small_run_bw_penalty
    return payload_bytes / bw + runs * mem.startup_ns * 1e-9

def simulate(cfg: FFNConfig, active_ids: List[int], hbm: MemoryConfig, hbf: MemoryConfig) -> Dict[str, Dict[str, float]]:
    k = len(active_ids)
    results: Dict[str, Dict[str, float]] = {}

    dense_bytes = dense_payload_bytes(cfg)

    results["HBM_dense"] = {
        "bytes": dense_bytes,
        "runs": 2,
        "time_s": time_seconds(dense_bytes, 2, hbm, small_run=False),
    }

    active_sorted = sorted(set(active_ids))
    runs = _runs_from_sorted_indices(active_sorted)
    sparse_ideal_bytes = sparse_payload_bytes_ideal(cfg, k)
    results["HBM_sparse_relayout_like"] = {
        "bytes": sparse_ideal_bytes,
        "runs": 2 * runs,
        "time_s": time_seconds(sparse_ideal_bytes, 2 * runs, hbm, small_run=True),
    }

    results["HBF_dense"] = {
        "bytes": dense_bytes,
        "runs": 2,
        "time_s": time_seconds(dense_bytes, 2, hbf, small_run=False),
    }

    dense_layout = sparse_physical_bytes_dense_layout_hbf(cfg, active_ids, hbf.page_size_B)
    results["HBF_sparse_dense_layout"] = {
        "bytes": dense_layout["total_bytes"],
        **dense_layout,
        "time_s": time_seconds(dense_layout["total_bytes"], int(dense_layout["total_runs"]), hbf, small_run=True),
    }

    relayout = sparse_physical_bytes_relayout_hbf(cfg, active_ids, hbf.page_size_B)
    results["HBF_sparse_relayout_like"] = {
        "bytes": relayout["total_bytes"],
        **relayout,
        "time_s": time_seconds(relayout["total_bytes"], int(relayout["total_runs"]), hbf, small_run=True),
    }

    return results

def pretty_print(results: Dict[str, Dict[str, float]]) -> None:
    order = [
        "HBM_dense",
        "HBM_sparse_relayout_like",
        "HBF_dense",
        "HBF_sparse_dense_layout",
        "HBF_sparse_relayout_like",
    ]
    for key in order:
        if key not in results:
            continue
        r = results[key]
        ms = r["time_s"] * 1e3
        gb = r["bytes"] / (1024**3)
        runs = r.get("runs", r.get("total_runs", 0))
        print(f"{key:28s}  time={ms:9.4f} ms   data={gb:8.4f} GiB   runs={runs}")


def results_to_jsonable(results: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    jsonable: Dict[str, Dict[str, float]] = {}
    for key, values in results.items():
        jsonable[key] = {}
        for inner_key, inner_value in values.items():
            if isinstance(inner_value, int):
                jsonable[key][inner_key] = int(inner_value)
            else:
                jsonable[key][inner_key] = float(inner_value)
    return jsonable

if __name__ == "__main__":
    cfg = FFNConfig(d=8192, m=28672, bytes_per_weight=2.0)

    hbm = MemoryConfig(
        name="HBM",
        bandwidth_GBps=2500.0,
        page_size_B=0,
        startup_ns=50.0,
        small_run_bw_penalty=0.65,
    )

    hbf = MemoryConfig(
        name="HBF",
        bandwidth_GBps=350.0,
        page_size_B=4096,
        startup_ns=300.0,
        small_run_bw_penalty=0.45,
    )

    active_ids = list(range(0, 32)) + list(range(1000, 1032)) + list(range(5000, 5032))
    results = simulate(cfg, active_ids, hbm, hbf)
    pretty_print(results)

    print("\\nNotes:")
    print("- HBF_sparse_dense_layout = case 3 sparse with active neurons known, but Wup still in dense layout.")
    print("- HBF_sparse_relayout_like = Wup relaid out neuron-major, like Wdown.")
    print("- startup_ns and small_run_bw_penalty are tunable sensitivity knobs, not vendor guarantees.")
