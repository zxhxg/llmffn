import argparse
import heapq
from pathlib import Path

import torch


DEFAULT_OUTPUT_DIR = Path("/home/wlh/llmffn/scripts/statistic/ffn_single_layer_profile_output")
DEFAULT_TOP_K = 10


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect activation counts and adjacency outputs from ffn_single_layer_profile.py.",
    )
    parser.add_argument("--layer", type=int, required=True, help="Layer index to inspect.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing activation_counts_layer_*.pt and adjacency_layer_*.pt.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of top neurons/edges to display in the summary.",
    )
    parser.add_argument(
        "--neurons",
        type=int,
        nargs="*",
        default=[],
        help="Specific neuron indices to inspect in activation_counts.",
    )
    parser.add_argument(
        "--edges",
        type=str,
        nargs="*",
        default=[],
        help="Specific edges to inspect, formatted as 'src,dst'.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        nargs="*",
        default=[],
        help="Specific adjacency rows to inspect; prints the top connected neighbors for each row.",
    )
    parser.add_argument(
        "--row-top-k",
        type=int,
        default=10,
        help="Number of neighbors to show for each row passed through --rows.",
    )
    return parser.parse_args()


def load_outputs(output_dir, layer):
    counts_path = output_dir / f"activation_counts_layer_{layer}.pt"
    adjacency_path = output_dir / f"adjacency_layer_{layer}.pt"

    if not counts_path.exists():
        raise FileNotFoundError(f"Missing counts file: {counts_path}")
    if not adjacency_path.exists():
        raise FileNotFoundError(f"Missing adjacency file: {adjacency_path}")

    counts = torch.load(counts_path, map_location="cpu")
    adjacency = torch.load(adjacency_path, map_location="cpu")
    return counts_path, adjacency_path, counts, adjacency


def tensor_size_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def format_bytes(num_bytes):
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{value:.2f}TB"


def top_activation_summary(counts, top_k):
    values, indices = torch.topk(counts, k=min(top_k, counts.numel()))
    return [
        {"index": int(index), "count": int(value)}
        for value, index in zip(values.tolist(), indices.tolist())
        if value > 0
    ]


def adjacency_nnz(adjacency):
    if adjacency.is_sparse:
        return int(adjacency._nnz())
    return int(torch.count_nonzero(adjacency).item())


def adjacency_value(adjacency, src, dst):
    return float(adjacency[src, dst].item())


def top_edges_dense(adjacency, top_k):
    heap = []
    size = adjacency.shape[0]
    for src in range(size):
        row = adjacency[src, src + 1 :]
        if row.numel() == 0:
            continue

        local_k = min(top_k, row.numel())
        values, offsets = torch.topk(row, k=local_k)
        for value, offset in zip(values.tolist(), offsets.tolist()):
            if value <= 0:
                continue
            dst = src + 1 + int(offset)
            item = (float(value), int(src), int(dst))
            if len(heap) < top_k:
                heapq.heappush(heap, item)
            elif value > heap[0][0]:
                heapq.heapreplace(heap, item)

    return [
        {"src": src, "dst": dst, "weight": weight}
        for weight, src, dst in sorted(heap, reverse=True)
    ]


def top_edges_sparse(adjacency, top_k):
    adjacency = adjacency.coalesce()
    indices = adjacency.indices()
    values = adjacency.values()
    heap = []
    for pos in range(values.numel()):
        src = int(indices[0, pos].item())
        dst = int(indices[1, pos].item())
        if src >= dst:
            continue

        weight = float(values[pos].item())
        item = (weight, src, dst)
        if len(heap) < top_k:
            heapq.heappush(heap, item)
        elif weight > heap[0][0]:
            heapq.heapreplace(heap, item)

    return [
        {"src": src, "dst": dst, "weight": weight}
        for weight, src, dst in sorted(heap, reverse=True)
    ]


def top_edge_summary(adjacency, top_k):
    if adjacency.is_sparse:
        return top_edges_sparse(adjacency, top_k)
    return top_edges_dense(adjacency, top_k)


def summarize_counts(counts, top_k):
    print("Counts Summary")
    print(f"  shape: {tuple(counts.shape)}")
    print(f"  dtype: {counts.dtype}")
    print(f"  nonzero_neurons: {int(torch.count_nonzero(counts).item())}")
    print(f"  total_activations: {int(counts.sum().item())}")
    print(f"  top_{top_k}_neurons: {top_activation_summary(counts, top_k)}")


def summarize_adjacency(adjacency, top_k):
    is_sparse = bool(adjacency.is_sparse)
    nnz = adjacency_nnz(adjacency)
    symmetric = bool(torch.equal(adjacency, adjacency.T))
    diagonal_nnz = int(torch.count_nonzero(torch.diag(adjacency)).item())

    print("Adjacency Summary")
    print(f"  shape: {tuple(adjacency.shape)}")
    print(f"  dtype: {adjacency.dtype}")
    print(f"  is_sparse: {is_sparse}")
    print(f"  storage_size: {format_bytes(tensor_size_bytes(adjacency))}")
    print(f"  nnz: {nnz}")
    print(f"  diagonal_nnz: {diagonal_nnz}")
    print(f"  symmetric: {symmetric}")
    print(f"  total_weight: {float(adjacency.sum().item()):.6f}")
    print(f"  top_{top_k}_edges: {top_edge_summary(adjacency, top_k)}")


def inspect_neurons(counts, neurons):
    if not neurons:
        return

    print("Neuron Values")
    for neuron_idx in neurons:
        if neuron_idx < 0 or neuron_idx >= counts.numel():
            print(f"  neuron[{neuron_idx}]: out of range")
            continue
        print(f"  neuron[{neuron_idx}] = {int(counts[neuron_idx].item())}")


def parse_edge(edge_text):
    parts = edge_text.split(",")
    if len(parts) != 2:
        raise ValueError(f"Invalid edge '{edge_text}', expected format 'src,dst'.")
    return int(parts[0]), int(parts[1])


def inspect_edges(adjacency, edges):
    if not edges:
        return

    print("Edge Values")
    size = adjacency.shape[0]
    for edge_text in edges:
        try:
            src, dst = parse_edge(edge_text)
        except ValueError as exc:
            print(f"  {exc}")
            continue

        if src < 0 or src >= size or dst < 0 or dst >= size:
            print(f"  edge[{src},{dst}]: out of range")
            continue
        print(f"  edge[{src},{dst}] = {adjacency_value(adjacency, src, dst):.6f}")


def inspect_rows(adjacency, rows, row_top_k):
    if not rows:
        return

    print("Row Neighbors")
    size = adjacency.shape[0]
    for row_idx in rows:
        if row_idx < 0 or row_idx >= size:
            print(f"  row[{row_idx}]: out of range")
            continue

        row = adjacency[row_idx].clone()
        row[row_idx] = 0
        k = min(row_top_k, row.numel())
        values, indices = torch.topk(row, k=k)
        neighbors = [
            {"dst": int(index), "weight": float(value)}
            for value, index in zip(values.tolist(), indices.tolist())
            if value > 0
        ]
        print(f"  row[{row_idx}] top_{row_top_k}: {neighbors}")


def main():
    args = parse_args()
    counts_path, adjacency_path, counts, adjacency = load_outputs(args.output_dir, args.layer)

    print(f"Layer: {args.layer}")
    print(f"Counts File: {counts_path}")
    print(f"Adjacency File: {adjacency_path}")
    print("")

    summarize_counts(counts, args.top_k)
    print("")
    summarize_adjacency(adjacency, args.top_k)
    print("")
    inspect_neurons(counts, args.neurons)
    if args.neurons:
        print("")
    inspect_edges(adjacency, args.edges)
    if args.edges:
        print("")
    inspect_rows(adjacency, args.rows, args.row_top_k)


if __name__ == "__main__":
    main()
