"""
graph.py — the queryable graph projection (Layers 1, 6).

This is a deterministic fold over the event log. In production it is the Neo4j
projection you traverse for lineage and critical-path queries; the write path
stays in Postgres (the store). Because the projection is *derived*, replaying the
log from scratch reconstructs it exactly — which is the whole durability claim.

It owns three guarantees:
  * dataflow firing            — ready_set() = Kahn's ready frontier (Layer 1)
  * acyclicity by construction — generation monotonicity + single-assignment (Layer 6)
  * auditable lineage          — ancestors() / critical_path() with full provenance
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional

from .core import Event, EventType, NodeState, NodeType, Provenance


class AcyclicityError(ValueError):
    """Raised when an edge would violate generation monotonicity (i.e. could
    close a cycle). Generation-monotonic edges are acyclic by construction."""


class SingleAssignmentError(ValueError):
    """Raised on a second completion of the same node. SSA: write once."""


@dataclass
class NodeRecord:
    node_id: str
    node_type: NodeType
    generation: int
    provenance: Provenance
    input_ids: tuple[str, ...] = ()
    state: NodeState = NodeState.CREATED
    output: Any = None
    score: Optional[float] = None
    tier: Optional[str] = None            # model tier chosen at dispatch (cost view)
    cost: float = 0.0
    attempts: int = 0
    params: dict = field(default_factory=dict)


class GraphProjection:
    def __init__(self) -> None:
        self.nodes: dict[str, NodeRecord] = {}
        self.out_edges: dict[str, set[str]] = {}
        self.in_edges: dict[str, set[str]] = {}
        self.terminal_node: Optional[str] = None
        self.terminated_reason: Optional[str] = None

    # --------------------------------------------------------------------- #
    # Folding: apply(event) is the ONLY way the projection changes           #
    # --------------------------------------------------------------------- #
    def apply(self, e: Event) -> None:
        t = e.type
        if t == EventType.NODE_CREATED:
            self._apply_node_created(e)
        elif t == EventType.EDGE_CREATED:
            self._apply_edge_created(e)
        elif t == EventType.NODE_STARTED:
            n = self.nodes[e.node_id]
            n.state = NodeState.STARTED
            n.attempts += 1
            n.tier = e.payload.get("tier")
            n.cost += float(e.payload.get("cost", 0.0))
        elif t == EventType.NODE_COMPLETED:
            n = self.nodes[e.node_id]
            if n.state == NodeState.COMPLETED:
                raise SingleAssignmentError(f"{e.node_id} completed twice (SSA violation)")
            n.state = NodeState.COMPLETED
            n.output = e.payload.get("output")
            n.score = e.payload.get("score")
        elif t == EventType.NODE_FAILED:
            self.nodes[e.node_id].state = NodeState.FAILED
        elif t == EventType.NODE_REOPENED:
            # Return a FAILED node to the frontier. This is folded from a LOGGED
            # fact (never an out-of-band mutation), so replay reconstructs the
            # same state and INV-5.2 holds even when a run ends mid-retry.
            self.nodes[e.node_id].state = NodeState.CREATED
        elif t == EventType.GOAL_SATISFIED:
            self.terminal_node = e.node_id
        elif t == EventType.RUN_TERMINATED:
            self.terminated_reason = e.payload.get("reason")
        # SUBGRAPH_EMITTED / NODE_PRUNED / RUN_STARTED are lineage/audit only.

    def _apply_node_created(self, e: Event) -> None:
        p = e.payload
        prov = _prov_from(p["provenance"])
        rec = NodeRecord(
            node_id=e.node_id,
            node_type=NodeType(p["node_type"]),
            generation=prov.generation,
            provenance=prov,
            input_ids=tuple(p.get("input_ids", ())),
            params=p.get("params", {}),
        )
        self.nodes[e.node_id] = rec
        self.out_edges.setdefault(e.node_id, set())
        self.in_edges.setdefault(e.node_id, set())

    def _apply_edge_created(self, e: Event) -> None:
        producer = e.payload["producer"]
        consumer = e.payload["consumer"]
        self.assert_can_add_edge(producer, consumer)  # Layer 6 guard on fold, too
        self.out_edges.setdefault(producer, set()).add(consumer)
        self.in_edges.setdefault(consumer, set()).add(producer)

    # --------------------------------------------------------------------- #
    # Layer 6 — acyclicity                                                    #
    # --------------------------------------------------------------------- #
    def assert_can_add_edge(self, producer: str, consumer: str) -> None:
        gp = self.nodes[producer].generation
        gc = self.nodes[consumer].generation
        if not gc > gp:
            raise AcyclicityError(
                f"edge {producer}(g{gp}) -> {consumer}(g{gc}) violates "
                f"generation monotonicity; would risk a cycle"
            )

    # --------------------------------------------------------------------- #
    # Layer 1 — dataflow firing                                              #
    # --------------------------------------------------------------------- #
    def ready_set(self) -> list[str]:
        """Kahn frontier: CREATED nodes all of whose producers are COMPLETED.
        This set is the parallelizable batch — in production every element is
        dispatched concurrently; the PoC runs them in a defined order but they
        are, by definition, mutually independent."""
        ready = []
        for nid, rec in self.nodes.items():
            if rec.state != NodeState.CREATED:
                continue
            producers = self.in_edges.get(nid, set())
            if all(self.nodes[p].state == NodeState.COMPLETED for p in producers):
                ready.append(nid)
        # deterministic order: by generation, then id
        return sorted(ready, key=lambda n: (self.nodes[n].generation, n))

    def has_open_work(self) -> bool:
        return any(
            r.state in (NodeState.CREATED, NodeState.STARTED)
            for r in self.nodes.values()
        )

    def next_generation_for(self, parent_ids: tuple[str, ...]) -> int:
        """A new node sits one generation above its highest parent. This single
        rule is what makes every emitted edge monotonic — hence acyclic."""
        if not parent_ids:
            return 0
        return max(self.nodes[p].generation for p in parent_ids) + 1

    # --------------------------------------------------------------------- #
    # Lineage / audit (the Neo4j-style queries)                              #
    # --------------------------------------------------------------------- #
    def ancestors(self, node_id: str) -> set[str]:
        seen: set[str] = set()
        stack = list(self.in_edges.get(node_id, set()))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(self.in_edges.get(cur, set()))
        return seen

    def critical_path(self) -> list[str]:
        """Longest dependency chain (by node count) ending at the terminal node,
        else at the deepest node. On a DAG this is a simple DP in topo order."""
        order = sorted(self.nodes, key=lambda n: self.nodes[n].generation)
        best_len: dict[str, int] = {}
        best_prev: dict[str, Optional[str]] = {}
        for nid in order:
            preds = self.in_edges.get(nid, set())
            if not preds:
                best_len[nid], best_prev[nid] = 1, None
            else:
                p = max(preds, key=lambda x: best_len.get(x, 0))
                best_len[nid], best_prev[nid] = best_len.get(p, 0) + 1, p
        end = self.terminal_node or max(best_len, key=best_len.get)
        path, cur = [], end
        while cur is not None:
            path.append(cur)
            cur = best_prev.get(cur)
        return list(reversed(path))

    def structural_hash(self) -> str:
        """A content hash of the graph's shape + terminal states. Used to prove
        that a replay reconstructs an identical projection."""
        h = hashlib.blake2b(digest_size=16)
        for nid in sorted(self.nodes):
            r = self.nodes[nid]
            h.update(f"{nid}|{r.node_type.value}|g{r.generation}|{r.state.value}|"
                     f"{sorted(self.in_edges.get(nid, set()))}".encode())
        h.update(f"terminal={self.terminal_node}|reason={self.terminated_reason}".encode())
        return h.hexdigest()

    @classmethod
    def replay(cls, events) -> "GraphProjection":
        g = cls()
        for e in events:
            g.apply(e)
        return g


def _prov_from(obj) -> Provenance:
    if isinstance(obj, Provenance):
        return obj
    return Provenance(
        run_id=obj["run_id"],
        batch_id=obj["batch_id"],
        generation=obj["generation"],
        created_at=obj["created_at"],
        expander_id=obj.get("expander_id"),
        parent_ids=tuple(obj.get("parent_ids", ())),
    )
