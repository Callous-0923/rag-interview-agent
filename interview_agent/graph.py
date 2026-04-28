from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


class GraphIndex:
    def __init__(self, graph_path: Path | list[Path]) -> None:
        self.graph_paths = graph_path if isinstance(graph_path, list) else [graph_path]
        self.labels: dict[str, str] = {}
        self.source_files: dict[str, str] = {}
        self.neighbors: dict[str, set[str]] = defaultdict(set)
        self.label_to_ids: dict[str, list[str]] = defaultdict(list)

    def load(self) -> "GraphIndex":
        for graph_path in self.graph_paths:
            self._load_one(graph_path)
        return self

    def _load_one(self, graph_path: Path) -> None:
        if not graph_path.exists():
            return
        data = json.loads(graph_path.read_text(encoding="utf-8"))
        for node in data.get("nodes", []):
            node_id = str(node.get("id", ""))
            label = str(node.get("label", ""))
            if not node_id:
                continue
            self.labels[node_id] = label
            self.source_files[node_id] = str(node.get("source_file") or "")
            self.label_to_ids[label.lower()].append(node_id)
        for edge in data.get("edges") or data.get("links") or []:
            src = str(edge.get("source", ""))
            dst = str(edge.get("target", ""))
            if src and dst:
                self.neighbors[src].add(dst)
                self.neighbors[dst].add(src)

    def query_graph_neighbors(self, topic: str, limit: int = 20) -> list[str]:
        ids = self._find_ids(topic)
        seen: set[str] = set()
        labels: list[str] = []
        for node_id in ids:
            for neighbor in self.neighbors.get(node_id, set()):
                label = self.labels.get(neighbor, neighbor)
                if label and label not in seen:
                    seen.add(label)
                    labels.append(label)
                    if len(labels) >= limit:
                        return labels
        return labels

    def expand_query_terms(self, query: str, limit: int = 10) -> list[str]:
        terms: list[str] = []
        lowered = query.lower()
        for label, ids in self.label_to_ids.items():
            display = self.labels.get(ids[0], label)
            if display and (display.lower() in lowered or lowered in display.lower()):
                terms.append(display)
                terms.extend(self.query_graph_neighbors(display, limit=limit))
                break
        return _dedupe(terms)[:limit]

    def _find_ids(self, label: str) -> list[str]:
        lowered = label.lower()
        exact = self.label_to_ids.get(lowered, [])
        if exact:
            return exact
        matches: list[str] = []
        for key, ids in self.label_to_ids.items():
            if lowered in key or key in lowered:
                matches.extend(ids)
        return matches[:5]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out
