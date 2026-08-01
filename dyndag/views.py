"""
views.py — projections over the graph for coordination, audit, and durability.

  Blackboard   — the Layer-4 coordination surface. The DAG *is* the board: cards
                 are nodes, columns are node states, dependencies are edges, the
                 WIP limit caps concurrent In-Progress cards, and a card's 'done'
                 is the typed NODE_COMPLETED event (never a free-text token).

  verify_replay — the durability claim, made falsifiable. Fold the event log from
                 scratch into a fresh projection and assert it is structurally
                 identical to the live one. If they match, the run is
                 reconstructible from the log alone.

  lineage_report — the audit claim. Walk provenance and show that every node
                 carries a complete tuple (run_id, batch_id, generation,
                 started_at, expander_id, parents) — i.e. no NULL provenance.
"""

from __future__ import annotations

from collections import defaultdict

from .core import NodeState
from .graph import GraphProjection
from .store import EventStore


# --------------------------------------------------------------------------- #
# Blackboard / Kanban (Layer 4)                                               #
# --------------------------------------------------------------------------- #
_COLUMNS = [
    ("Backlog",     {NodeState.CREATED}),
    ("In-Progress", {NodeState.STARTED}),
    ("Done",        {NodeState.COMPLETED}),
    ("Failed",      {NodeState.FAILED}),
]


def render_blackboard(graph: GraphProjection, wip_limit: int = 4) -> str:
    cols: dict[str, list[str]] = defaultdict(list)
    for nid, rec in sorted(graph.nodes.items(), key=lambda kv: (kv[1].generation, kv[0])):
        for name, states in _COLUMNS:
            if rec.state in states:
                tier = f" [{rec.tier}]" if rec.tier else ""
                cols[name].append(f"{nid} ({rec.node_type.value}){tier}")
    lines = ["BLACKBOARD  (WIP limit = %d)" % wip_limit]
    for name, _ in _COLUMNS:
        cards = cols.get(name, [])
        flag = "  <-- WIP breach" if name == "In-Progress" and len(cards) > wip_limit else ""
        lines.append(f"  {name} ({len(cards)}){flag}")
        for c in cards:
            lines.append(f"      - {c}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Durability: event-sourced replay (Layer 5)                                  #
# --------------------------------------------------------------------------- #
def verify_replay(store: EventStore, live: GraphProjection, run_id: str) -> tuple[bool, str, str]:
    replayed = GraphProjection.replay(store.read(run_id))
    live_h, rep_h = live.structural_hash(), replayed.structural_hash()
    return live_h == rep_h, live_h, rep_h


# --------------------------------------------------------------------------- #
# Audit: lineage + provenance completeness (Layers 1, 5)                       #
# --------------------------------------------------------------------------- #
def provenance_gaps(graph: GraphProjection) -> list[tuple[str, str]]:
    """Return (node_id, missing_field) for any incomplete provenance. Should be
    empty by construction — this function exists to PROVE that, not to fix it."""
    gaps = []
    for nid, rec in graph.nodes.items():
        p = rec.provenance
        for field_name in ("run_id", "batch_id", "created_at"):
            if not getattr(p, field_name):
                gaps.append((nid, field_name))
        if p.generation is None:
            gaps.append((nid, "generation"))
    return gaps


def cost_report(graph: GraphProjection) -> str:
    """Real token + dollar accounting, folded from NODE_COMPLETED. Nodes whose
    executor reported no usage (mocks) contribute only their nominal cost units."""
    tok_in = tok_out = 0
    usd = 0.0
    per_tier: dict[str, float] = defaultdict(float)      # nominal units, by tier
    for rec in graph.nodes.values():
        if rec.tier:
            per_tier[rec.tier] += rec.cost
        if rec.usage:
            tok_in += rec.usage.get("input_tokens", 0)
            tok_out += rec.usage.get("output_tokens", 0)
        if rec.cost_usd:
            usd += rec.cost_usd
    lines = ["COST + USAGE",
             f"  nominal units by tier : {dict(per_tier)}",
             f"  real tokens           : {tok_in} in / {tok_out} out",
             f"  real cost             : ${usd:.6f}"]
    return "\n".join(lines)


def lineage_report(graph: GraphProjection) -> str:
    path = graph.critical_path()
    lines = ["CRITICAL PATH (longest dependency chain):",
             "  " + "  ->  ".join(path)]
    lines.append("")
    lines.append("PROVENANCE (every node stamped at creation):")
    for nid in path:
        rec = graph.nodes[nid]
        p = rec.provenance
        exp = p.expander_id or "(seed)"
        lines.append(
            f"  {nid:<20} gen={p.generation}  batch={p.batch_id}  "
            f"emitted_by={exp}  parents={list(p.parent_ids)}"
        )
    gaps = provenance_gaps(graph)
    lines.append("")
    lines.append(f"PROVENANCE GAPS (NULL fields): {gaps if gaps else 'none'}")
    return "\n".join(lines)
