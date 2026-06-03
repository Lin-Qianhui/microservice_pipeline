#!/usr/bin/env python3
"""Cluster a call graph into microservice candidates.

Reads edge list CSV (from ``microservice-pipeline call-graph``) and outputs:
- cluster_assignments.csv: node -> cluster mapping with readable metadata
- cluster_members.csv: cluster members sorted by cluster and module
- cluster_summary.csv: cluster-level cohesion stats
- cluster_edges.csv: inter-cluster call flow
- cluster_report.md: readable Markdown summary
- clusters.json: combined payload

Leiden clustering is supported when ``igraph`` and ``leidenalg`` are installed.
Infomap clustering is supported when ``infomap`` is installed.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from microservice_pipeline.artifact_io import ensure_dir, write_csv_rows, write_json, write_markdown
except ImportError:  # pragma: no cover - supports direct script execution
    from artifact_io import ensure_dir, write_csv_rows, write_json, write_markdown  # type: ignore


@dataclass
class Edge:
    src: str
    dst: str
    weight: int = 1


def load_node_rows(nodes_csv: Path | None) -> Dict[str, dict]:
    if nodes_csv is None or not nodes_csv.exists():
        return {}
    node_rows: Dict[str, dict] = {}
    with nodes_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node = (row.get("id") or "").strip()
            if node:
                node_rows[node] = row
    return node_rows


def load_nodes(nodes_csv: Path | None) -> Set[str]:
    return set(load_node_rows(nodes_csv))


def load_edges(edges_csv: Path) -> Tuple[List[Edge], Set[str]]:
    edge_weights: Dict[Tuple[str, str], int] = defaultdict(int)
    nodes: Set[str] = set()

    with edges_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = (row.get("caller") or "").strip()
            dst = (row.get("callee") or "").strip()
            if not src or not dst:
                continue
            edge_weights[(src, dst)] += 1
            nodes.add(src)
            nodes.add(dst)

    edges = [Edge(src=s, dst=t, weight=w) for (s, t), w in edge_weights.items()]
    return edges, nodes


def build_directed_adjacency(edges: Iterable[Edge]) -> Tuple[Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
    out_adj: Dict[str, Dict[str, int]] = defaultdict(dict)
    in_adj: Dict[str, Dict[str, int]] = defaultdict(dict)

    for e in edges:
        out_adj[e.src][e.dst] = out_adj[e.src].get(e.dst, 0) + e.weight
        in_adj[e.dst][e.src] = in_adj[e.dst].get(e.src, 0) + e.weight

    return out_adj, in_adj


def build_undirected_adjacency(edges: Iterable[Edge], nodes: Iterable[str]) -> Dict[str, Dict[str, int]]:
    adj: Dict[str, Dict[str, int]] = {n: {} for n in nodes}
    for e in edges:
        adj[e.src][e.dst] = adj[e.src].get(e.dst, 0) + e.weight
        adj[e.dst][e.src] = adj[e.dst].get(e.src, 0) + e.weight
    return adj


def compute_node_degrees(nodes: Iterable[str], edges: Iterable[Edge]) -> Dict[str, dict]:
    incoming = defaultdict(int)
    outgoing = defaultdict(int)
    for e in edges:
        outgoing[e.src] += e.weight
        incoming[e.dst] += e.weight

    degree_map: Dict[str, dict] = {}
    for node in nodes:
        indeg = incoming[node]
        outdeg = outgoing[node]
        degree_map[node] = {
            "in_degree": indeg,
            "out_degree": outdeg,
            "total_degree": indeg + outdeg,
        }
    return degree_map


def cluster_wcc(nodes: Set[str], undirected_adj: Dict[str, Dict[str, int]]) -> Dict[str, str]:
    cluster_of: Dict[str, str] = {}
    visited: Set[str] = set()
    cid = 0

    for start in sorted(nodes):
        if start in visited:
            continue
        cid += 1
        cluster_id = f"C{cid:03d}"
        queue = deque([start])
        visited.add(start)
        while queue:
            cur = queue.popleft()
            cluster_of[cur] = cluster_id
            for nbr in undirected_adj.get(cur, {}):
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
    return cluster_of


def cluster_scc(nodes: Set[str], out_adj: Dict[str, Dict[str, int]], in_adj: Dict[str, Dict[str, int]]) -> Dict[str, str]:
    # Kosaraju
    visited: Set[str] = set()
    order: List[str] = []

    def dfs1(start: str) -> None:
        stack = [(start, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for nxt in out_adj.get(node, {}):
                if nxt not in visited:
                    stack.append((nxt, False))

    for node in sorted(nodes):
        if node not in visited:
            dfs1(node)

    cluster_of: Dict[str, str] = {}
    visited.clear()
    cid = 0

    for node in reversed(order):
        if node in visited:
            continue
        cid += 1
        cluster_id = f"C{cid:03d}"
        stack = [node]
        visited.add(node)
        while stack:
            cur = stack.pop()
            cluster_of[cur] = cluster_id
            for prv in in_adj.get(cur, {}):
                if prv not in visited:
                    visited.add(prv)
                    stack.append(prv)

    return cluster_of


def cluster_label_propagation(
    nodes: Set[str],
    undirected_adj: Dict[str, Dict[str, int]],
    max_iter: int,
    seed: int,
) -> Dict[str, str]:
    rng = random.Random(seed)
    labels: Dict[str, str] = {n: n for n in nodes}
    node_list = list(nodes)

    for _ in range(max_iter):
        changed = False
        rng.shuffle(node_list)
        for node in node_list:
            nbrs = undirected_adj.get(node, {})
            if not nbrs:
                continue

            scores: Dict[str, int] = defaultdict(int)
            for nbr, w in nbrs.items():
                scores[labels[nbr]] += w

            if not scores:
                continue

            best_score = max(scores.values())
            best_labels = [lab for lab, score in scores.items() if score == best_score]
            best_labels.sort()
            chosen = best_labels[0]

            if labels[node] != chosen:
                labels[node] = chosen
                changed = True

        if not changed:
            break

    # normalize labels to Cxxx ids ordered by (size desc, min node)
    groups: Dict[str, List[str]] = defaultdict(list)
    for n, lab in labels.items():
        groups[lab].append(n)

    sorted_groups = sorted(
        groups.values(),
        key=lambda members: (-len(members), min(members)),
    )

    cluster_of: Dict[str, str] = {}
    for idx, members in enumerate(sorted_groups, start=1):
        cluster_id = f"C{idx:03d}"
        for m in members:
            cluster_of[m] = cluster_id
    return cluster_of


def cluster_leiden(
    nodes: Set[str],
    undirected_adj: Dict[str, Dict[str, int]],
    resolution: float,
    seed: int,
) -> Dict[str, str]:
    try:
        import igraph as ig  # type: ignore
        import leidenalg  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Leiden clustering requires the optional dependencies 'igraph' and "
            "'leidenalg'. Install them before using --algorithm leiden."
        ) from exc

    node_list = sorted(nodes)
    node_index = {node: idx for idx, node in enumerate(node_list)}
    undirected_edges: List[Tuple[int, int]] = []
    weights: List[int] = []

    for src in node_list:
        for dst, weight in undirected_adj.get(src, {}).items():
            if src >= dst:
                continue
            undirected_edges.append((node_index[src], node_index[dst]))
            weights.append(weight)

    graph = ig.Graph(n=len(node_list), edges=undirected_edges, directed=False)
    if weights:
        graph.es["weight"] = weights

    if graph.vcount() == 0:
        return {}
    if graph.ecount() == 0:
        return {node: f"C{idx:03d}" for idx, node in enumerate(node_list, start=1)}

    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights=weights if weights else None,
        seed=seed,
        resolution_parameter=resolution,
    )

    groups: Dict[int, List[str]] = defaultdict(list)
    for node, membership in zip(node_list, partition.membership):
        groups[membership].append(node)

    ordered_groups = sorted(groups.values(), key=lambda members: (-len(members), min(members)))
    cluster_of: Dict[str, str] = {}
    for idx, members in enumerate(ordered_groups, start=1):
        cluster_id = f"C{idx:03d}"
        for member in members:
            cluster_of[member] = cluster_id
    return cluster_of


def cluster_infomap(
    nodes: Set[str],
    edges: List[Edge],
    seed: int,
) -> Dict[str, str]:
    try:
        from infomap import Infomap  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Infomap clustering requires the optional dependency 'infomap'. "
            "Install it before using --algorithm infomap."
        ) from exc

    node_list = sorted(nodes)
    if not node_list:
        return {}
    if not edges:
        return {node: f"C{idx:03d}" for idx, node in enumerate(node_list, start=1)}

    node_index = {node: idx for idx, node in enumerate(node_list, start=1)}
    reverse_index = {idx: node for node, idx in node_index.items()}

    infomap = Infomap(f"--two-level --directed --silent --seed {seed}")
    for edge in edges:
        infomap.add_link(node_index[edge.src], node_index[edge.dst], edge.weight)

    infomap.run()

    groups: Dict[int, List[str]] = defaultdict(list)
    for tree_node in infomap.tree:
        if not getattr(tree_node, "is_leaf", False):
            continue
        node_id = getattr(tree_node, "physical_id", None)
        if node_id is None:
            node_id = getattr(tree_node, "node_id", None)
        if node_id is None or node_id not in reverse_index:
            continue
        groups[int(tree_node.module_id)].append(reverse_index[node_id])

    assigned_nodes = {node for members in groups.values() for node in members}
    missing_nodes = [node for node in node_list if node not in assigned_nodes]
    next_group_id = (max(groups.keys()) + 1) if groups else 1
    for node in missing_nodes:
        groups[next_group_id] = [node]
        next_group_id += 1

    ordered_groups = sorted(groups.values(), key=lambda members: (-len(members), min(members)))
    cluster_of: Dict[str, str] = {}
    for idx, members in enumerate(ordered_groups, start=1):
        cluster_id = f"C{idx:03d}"
        for member in members:
            cluster_of[member] = cluster_id
    return cluster_of


def reindex_clusters(cluster_of: Dict[str, str]) -> Dict[str, str]:
    groups: Dict[str, List[str]] = defaultdict(list)
    for node, cid in cluster_of.items():
        groups[cid].append(node)

    ordered = sorted(groups.values(), key=lambda members: (-len(members), min(members)))
    remap: Dict[str, str] = {}
    for idx, members in enumerate(ordered, start=1):
        new_cid = f"C{idx:03d}"
        for n in members:
            remap[n] = new_cid
    return remap


def _cluster_members(cluster_of: Dict[str, str]) -> Dict[str, Set[str]]:
    members: Dict[str, Set[str]] = defaultdict(set)
    for node, cid in cluster_of.items():
        members[cid].add(node)
    return members


def _sorted_cluster_pair(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _cluster_pair_weights(cluster_of: Dict[str, str], edges: List[Edge]) -> Dict[Tuple[str, str], int]:
    weights: Dict[Tuple[str, str], int] = defaultdict(int)
    for e in edges:
        a = cluster_of.get(e.src)
        b = cluster_of.get(e.dst)
        if not a or not b or a == b:
            continue
        key = _sorted_cluster_pair(a, b)
        weights[key] += e.weight
    return weights


def _shared_prefix_len(a: str, b: str) -> int:
    pa = a.split(".")
    pb = b.split(".")
    length = 0
    for x, y in zip(pa, pb):
        if x != y:
            break
        length += 1
    return length


def _cluster_similarity(members: Dict[str, Set[str]], c1: str, c2: str) -> int:
    # Cheap lexical fallback when call-edge linkage is absent.
    # This often keeps methods from nearby namespaces together.
    n1 = min(members[c1])
    n2 = min(members[c2])
    return _shared_prefix_len(n1, n2)


def _merge_into(cluster_of: Dict[str, str], src_cluster: str, dst_cluster: str) -> None:
    for node, cid in list(cluster_of.items()):
        if cid == src_cluster:
            cluster_of[node] = dst_cluster


def reduce_cluster_count(
    cluster_of: Dict[str, str],
    edges: List[Edge],
    min_cluster_size: int = 1,
    max_clusters: int = 0,
) -> Dict[str, str]:
    cluster_of = dict(cluster_of)
    min_cluster_size = max(1, min_cluster_size)
    max_clusters = max(0, max_clusters)

    # Step 1: merge too-small clusters into their best neighboring cluster.
    changed = True
    while changed:
        changed = False
        members = _cluster_members(cluster_of)
        sizes = {cid: len(nodes) for cid, nodes in members.items()}
        if len(sizes) <= 1:
            break

        pair_w = _cluster_pair_weights(cluster_of, edges)
        small_clusters = [cid for cid, size in sizes.items() if size < min_cluster_size]
        small_clusters.sort(key=lambda cid: (sizes[cid], cid))

        for src in small_clusters:
            members = _cluster_members(cluster_of)
            sizes = {cid: len(nodes) for cid, nodes in members.items()}
            if src not in sizes or sizes[src] >= min_cluster_size or len(sizes) <= 1:
                continue

            best_target = None
            best_score = None
            for dst in sizes:
                if dst == src:
                    continue
                edge_score = pair_w.get(_sorted_cluster_pair(src, dst), 0)
                sim = _cluster_similarity(members, src, dst)
                score = (edge_score, sim, sizes[dst], dst)
                if best_score is None or score > best_score:
                    best_score = score
                    best_target = dst

            if best_target is not None:
                _merge_into(cluster_of, src, best_target)
                changed = True

    # Step 2: cap total cluster count if requested.
    while max_clusters > 0:
        members = _cluster_members(cluster_of)
        sizes = {cid: len(nodes) for cid, nodes in members.items()}
        if len(sizes) <= max_clusters or len(sizes) <= 1:
            break

        pair_w = _cluster_pair_weights(cluster_of, edges)
        cids = sorted(sizes)
        best_pair = None
        best_score = None

        for i, c1 in enumerate(cids):
            for c2 in cids[i + 1 :]:
                edge_score = pair_w.get((c1, c2), 0)
                sim = _cluster_similarity(members, c1, c2)
                merged_size = sizes[c1] + sizes[c2]
                # Prefer strong linkage first, then lexical proximity, then smaller merge.
                score = (edge_score, sim, -merged_size, c1, c2)
                if best_score is None or score > best_score:
                    best_score = score
                    best_pair = (c1, c2)

        if best_pair is None:
            break

        c1, c2 = best_pair
        # Merge smaller into larger for stable labels.
        if sizes[c1] <= sizes[c2]:
            src, dst = c1, c2
        else:
            src, dst = c2, c1
        _merge_into(cluster_of, src, dst)

    return reindex_clusters(cluster_of)


def compute_cluster_artifacts(
    cluster_of: Dict[str, str], edges: List[Edge], node_rows: Dict[str, dict]
) -> Tuple[List[dict], List[dict]]:
    members: Dict[str, List[str]] = defaultdict(list)
    for n, c in cluster_of.items():
        members[c].append(n)

    internal_weight: Dict[str, int] = defaultdict(int)
    outgoing_weight: Dict[str, int] = defaultdict(int)
    incoming_weight: Dict[str, int] = defaultdict(int)
    cluster_flow: Dict[Tuple[str, str], int] = defaultdict(int)

    for e in edges:
        c_src = cluster_of.get(e.src)
        c_dst = cluster_of.get(e.dst)
        if not c_src or not c_dst:
            continue
        if c_src == c_dst:
            internal_weight[c_src] += e.weight
        else:
            outgoing_weight[c_src] += e.weight
            incoming_weight[c_dst] += e.weight
            cluster_flow[(c_src, c_dst)] += e.weight

    summary_rows: List[dict] = []
    for cid, nodes in sorted(members.items()):
        internal = internal_weight[cid]
        outgoing = outgoing_weight[cid]
        incoming = incoming_weight[cid]
        total_touching = internal + outgoing + incoming
        cohesion = (internal / total_touching) if total_touching else 0.0
        module_counts: Dict[str, int] = defaultdict(int)
        kind_counts: Dict[str, int] = defaultdict(int)
        for node in nodes:
            row = node_rows.get(node, {})
            module_counts[(row.get("module") or "(unknown)").strip()] += 1
            kind_counts[(row.get("kind") or "(unknown)").strip()] += 1

        top_modules = ";".join(
            f"{module}({count})"
            for module, count in sorted(module_counts.items(), key=lambda item: (-item[1], item[0]))[:5]
        )
        kinds = ";".join(
            f"{kind}({count})"
            for kind, count in sorted(kind_counts.items(), key=lambda item: (-item[1], item[0]))
        )

        summary_rows.append(
            {
                "cluster_id": cid,
                "size": len(nodes),
                "internal_weight": internal,
                "outgoing_weight": outgoing,
                "incoming_weight": incoming,
                "cohesion": f"{cohesion:.6f}",
                "top_modules": top_modules,
                "kinds": kinds,
                "members_preview": ";".join(sorted(nodes)[:8]),
            }
        )

    edge_rows: List[dict] = []
    for (a, b), w in sorted(cluster_flow.items(), key=lambda x: (-x[1], x[0][0], x[0][1])):
        edge_rows.append({"src_cluster": a, "dst_cluster": b, "weight": w})

    summary_rows.sort(key=lambda r: (-int(r["size"]), r["cluster_id"]))
    return summary_rows, edge_rows


def write_outputs(
    outdir: Path,
    cluster_of: Dict[str, str],
    summary_rows: List[dict],
    edge_rows: List[dict],
    algorithm: str,
    node_rows: Dict[str, dict],
    degree_map: Dict[str, dict],
) -> None:
    ensure_dir(outdir)

    size_map: Dict[str, int] = defaultdict(int)
    for _, c in cluster_of.items():
        size_map[c] += 1

    assignment_fields = [
        "cluster_id",
        "cluster_size",
        "node",
        "module",
        "qualname",
        "kind",
        "class_name",
        "in_degree",
        "out_degree",
        "total_degree",
        "file",
        "lineno",
    ]
    assignment_rows = []
    for node in sorted(cluster_of, key=lambda n: (cluster_of[n], node_rows.get(n, {}).get("module", ""), node_rows.get(n, {}).get("qualname", ""), n)):
        cid = cluster_of[node]
        row = node_rows.get(node, {})
        degrees = degree_map.get(node, {"in_degree": 0, "out_degree": 0, "total_degree": 0})
        assignment_rows.append(
            {
                "cluster_id": cid,
                "cluster_size": size_map[cid],
                "node": node,
                "module": row.get("module", ""),
                "qualname": row.get("qualname", ""),
                "kind": row.get("kind", ""),
                "class_name": row.get("class_name", ""),
                "in_degree": degrees["in_degree"],
                "out_degree": degrees["out_degree"],
                "total_degree": degrees["total_degree"],
                "file": row.get("file", ""),
                "lineno": row.get("lineno", ""),
            }
        )
    write_csv_rows(outdir / "cluster_assignments.csv", assignment_fields, assignment_rows)

    member_fields = [
        "cluster_id",
        "cluster_size",
        "module",
        "qualname",
        "kind",
        "class_name",
        "node",
        "in_degree",
        "out_degree",
        "total_degree",
    ]
    member_rows = []
    for node in sorted(
        cluster_of,
        key=lambda n: (
            -size_map[cluster_of[n]],
            cluster_of[n],
            node_rows.get(n, {}).get("module", ""),
            node_rows.get(n, {}).get("qualname", ""),
            n,
        ),
    ):
        cid = cluster_of[node]
        row = node_rows.get(node, {})
        degrees = degree_map.get(node, {"in_degree": 0, "out_degree": 0, "total_degree": 0})
        member_rows.append(
            {
                "cluster_id": cid,
                "cluster_size": size_map[cid],
                "module": row.get("module", ""),
                "qualname": row.get("qualname", ""),
                "kind": row.get("kind", ""),
                "class_name": row.get("class_name", ""),
                "node": node,
                "in_degree": degrees["in_degree"],
                "out_degree": degrees["out_degree"],
                "total_degree": degrees["total_degree"],
            }
        )
    write_csv_rows(outdir / "cluster_members.csv", member_fields, member_rows)

    write_csv_rows(
        outdir / "cluster_summary.csv",
        list(summary_rows[0].keys())
        if summary_rows
        else ["cluster_id", "size", "internal_weight", "outgoing_weight", "incoming_weight", "cohesion", "top_modules", "kinds", "members_preview"],
        summary_rows,
    )
    write_csv_rows(
        outdir / "cluster_edges.csv",
        list(edge_rows[0].keys()) if edge_rows else ["src_cluster", "dst_cluster", "weight"],
        edge_rows,
    )

    payload = {
        "algorithm": algorithm,
        "num_nodes": len(cluster_of),
        "num_clusters": len({c for c in cluster_of.values()}),
        "summary": summary_rows,
        "cluster_edges": edge_rows,
    }
    write_json(outdir / "clusters.json", payload)

    report_md = outdir / "cluster_report.md"
    lines = [
        "# Call Graph Clusters",
        "",
        f"- Algorithm: `{algorithm}`",
        f"- Nodes clustered: `{len(cluster_of)}`",
        f"- Clusters: `{len({c for c in cluster_of.values()})}`",
        "",
    ]

    assignments_by_cluster: Dict[str, List[str]] = defaultdict(list)
    for node, cid in cluster_of.items():
        assignments_by_cluster[cid].append(node)

    summary_by_cluster = {row["cluster_id"]: row for row in summary_rows}
    for cid in sorted(assignments_by_cluster, key=lambda c: (-size_map[c], c)):
        row = summary_by_cluster.get(cid, {})
        lines.extend(
            [
                f"## {cid}",
                "",
                f"- Size: `{size_map[cid]}`",
                f"- Cohesion: `{row.get('cohesion', '0.000000')}`",
                f"- Internal / outgoing / incoming weight: `{row.get('internal_weight', 0)}` / `{row.get('outgoing_weight', 0)}` / `{row.get('incoming_weight', 0)}`",
                f"- Top modules: {row.get('top_modules', '') or '(none)'}",
                "",
                "| Module | Qualname | Kind | In | Out |",
                "| --- | --- | --- | ---: | ---: |",
            ]
        )
        cluster_nodes = sorted(
            assignments_by_cluster[cid],
            key=lambda n: (
                node_rows.get(n, {}).get("module", ""),
                node_rows.get(n, {}).get("qualname", ""),
                n,
            ),
        )
        for node in cluster_nodes[:20]:
            meta = node_rows.get(node, {})
            deg = degree_map.get(node, {"in_degree": 0, "out_degree": 0})
            lines.append(
                f"| {meta.get('module', '')} | {meta.get('qualname', node)} | {meta.get('kind', '')} | {deg['in_degree']} | {deg['out_degree']} |"
            )
        if len(cluster_nodes) > 20:
            lines.append(f"| ... | showing first 20 of {len(cluster_nodes)} members |  |  |  |")
        lines.append("")

    write_markdown(report_md, lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster call graph for microservice candidates")
    parser.add_argument("--edges", required=True, help="Path to edges.csv (caller/callee columns)")
    parser.add_argument("--nodes", default=None, help="Optional path to nodes.csv (id column) for isolated nodes")
    parser.add_argument(
        "--algorithm",
        default="label_propagation",
        choices=["infomap", "leiden", "label_propagation", "wcc", "scc"],
        help="Clustering algorithm",
    )
    parser.add_argument("--max-iter", type=int, default=100, help="Max iterations for label propagation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for label propagation")
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Resolution parameter for Leiden clustering",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=1,
        help="Post-process: merge clusters smaller than this size into nearby clusters",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=0,
        help="Post-process: cap final number of clusters (0 means no cap)",
    )
    parser.add_argument(
        "--exclude-isolated",
        action="store_true",
        help="Ignore nodes with no edges (helps avoid singleton-only clusters)",
    )
    parser.add_argument("--outdir", default="artifacts/clusters", help="Output directory")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    edges_path = Path(args.edges).resolve()
    nodes_path = Path(args.nodes).resolve() if args.nodes else None
    outdir = Path(args.outdir).resolve()

    edges, nodes_from_edges = load_edges(edges_path)
    nodes = set(nodes_from_edges)
    node_rows = load_node_rows(nodes_path)
    nodes.update(node_rows)
    if args.exclude_isolated:
        nodes = set(nodes_from_edges)
        node_rows = {node: row for node, row in node_rows.items() if node in nodes}

    out_adj, in_adj = build_directed_adjacency(edges)
    undirected_adj = build_undirected_adjacency(edges, nodes)
    degree_map = compute_node_degrees(nodes, edges)

    if args.algorithm == "wcc":
        cluster_of = cluster_wcc(nodes, undirected_adj)
    elif args.algorithm == "scc":
        cluster_of = cluster_scc(nodes, out_adj, in_adj)
    elif args.algorithm == "infomap":
        cluster_of = cluster_infomap(
            nodes=nodes,
            edges=edges,
            seed=args.seed,
        )
    elif args.algorithm == "leiden":
        cluster_of = cluster_leiden(
            nodes=nodes,
            undirected_adj=undirected_adj,
            resolution=args.resolution,
            seed=args.seed,
        )
    else:
        cluster_of = cluster_label_propagation(
            nodes=nodes,
            undirected_adj=undirected_adj,
            max_iter=args.max_iter,
            seed=args.seed,
        )

    cluster_of = reduce_cluster_count(
        cluster_of=cluster_of,
        edges=edges,
        min_cluster_size=args.min_cluster_size,
        max_clusters=args.max_clusters,
    )

    summary_rows, edge_rows = compute_cluster_artifacts(cluster_of, edges, node_rows=node_rows)
    write_outputs(
        outdir=outdir,
        cluster_of=cluster_of,
        summary_rows=summary_rows,
        edge_rows=edge_rows,
        algorithm=args.algorithm,
        node_rows=node_rows,
        degree_map=degree_map,
    )

    print(f"Clustering output written to: {outdir}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Nodes: {len(nodes)}")
    print(f"Clusters: {len({c for c in cluster_of.values()})}")


if __name__ == "__main__":
    main()
