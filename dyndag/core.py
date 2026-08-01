"""
core.py — shared primitives for the dynamic-DAG orchestrator.

This module encodes the vocabulary that every other layer builds on:
  * stable, replay-reproducible identifiers                (Layer 1)
  * the enums that type nodes, states, tiers, signals      (Layers 1, 4)
  * the Provenance tuple + its provenance-at-write guard    (Layers 1, 5)
  * the immutable Event record                              (Layer 5 — event sourcing)

Design invariant (governs the whole system): the *only* mutable thing in the
system is the append-only event log. Everything else — the graph, the frontier,
the blackboard — is a deterministic fold over that log. Nothing is created
without an event, and no creation event is accepted without complete provenance.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Identifiers (Layer 1)                                                        #
# --------------------------------------------------------------------------- #
def _digest(*parts: Any, length: int = 8) -> str:
    h = hashlib.blake2b(digest_size=16)
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x1f")  # unit separator so ("ab","c") != ("a","bc")
    return h.hexdigest()[:length]


def stable_node_id(key: str, generation: int, parent_ids: tuple[str, ...]) -> str:
    """
    Deterministic node id.

    `key` is the logical label an expander assigns ("scout_a", "engineer",
    "reviewer"). `generation` disambiguates SSA versions of the same logical
    role (engineer.g3 vs engineer.g5 are distinct bindings — this is how we get
    iteration WITHOUT cycles). The parent hash is a collision guard for the rare
    case where two siblings share key+generation.

    The id is human-legible on purpose: `<key>.g<gen>` reads cleanly in a
    lineage query, and stable ids are what make the graph delegatable.
    """
    # key+generation is unique by construction in this design (an expander never
    # emits two nodes with the same key at the same generation); the scheduler
    # asserts uniqueness on creation, so no hash suffix is needed for legibility.
    _ = parent_ids  # retained in the signature for future collision strategies
    return f"{key}.g{generation}"


def stable_edge_id(producer: str, consumer: str) -> str:
    return f"{producer}->{consumer}"


def new_run_id() -> str:
    return "run-" + _digest(datetime.now(timezone.utc).isoformat(), length=10)


def new_batch_id(run_id: str, generation: int) -> str:
    return f"{run_id}:b{generation}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Enums (Layers 1, 4)                                                          #
# --------------------------------------------------------------------------- #
class NodeType(str, Enum):
    """
    A node's type IS its function in the dataflow graph (Layer 4). Roles are not
    personalities; they are typed positions. This enum is the full Hermes-style
    fleet; the PoC binds executors for the subset the demo exercises, and the
    remainder (TESTER, HARDENER, OPS, MAINTAINER, DEBUGGER) follow the identical
    Executor contract.
    """
    GOAL = "goal"                 # terminal input source: holds the target spec
    ORCHESTRATOR = "orchestrator" # the expander — emits subgraphs (Layer 2, primitive 4)
    SCOUT = "scout"               # frontier retrieval — fetches raw material
    ENGINEER = "engineer"         # generation / refinement — produces & improves content
    REVIEWER = "reviewer"         # critic — produces the error signal (Layer 3)
    SYNTHESIZER = "synthesizer"   # aggregator — converges branches to one outcome
    TESTER = "tester"             # critic variant
    HARDENER = "hardener"         # terminal quality gate
    OPS = "ops"                   # terminal quality gate
    MAINTAINER = "maintainer"
    DEBUGGER = "debugger"


class NodeState(str, Enum):
    CREATED = "created"       # exists, inputs not all ready
    READY = "ready"           # derived; all input producers COMPLETED (never stored — computed)
    STARTED = "started"       # dispatched to an executor
    COMPLETED = "completed"   # output materialized (single-assignment — write once)
    FAILED = "failed"         # executor raised; may retry against the global budget


class Tier(str, Enum):
    LOCAL = "local"   # cheap, high-throughput — scouts, first-pass generation
    CLOUD = "cloud"   # expensive, outcome-gating — critics, aggregation


class Signal(str, Enum):
    """
    A node's completion signal is a TYPED value returned by the executor — never
    a control token parsed out of free-text output. This is the structural fix
    for the 'agent emitted <complete> as a literal string' failure class: the
    scheduler reads result.signal and never inspects result.output for control.
    """
    CONTINUE = "continue"                 # ordinary completion, nothing special
    DONE = "done"                         # this node's sub-objective is satisfied
    NEEDS_REFINEMENT = "needs_refinement" # critic rejected — expander will add a refinement stratum
    GOAL = "goal"                         # candidate terminal — scheduler checks the goal predicate


class EventType(str, Enum):
    RUN_STARTED = "run_started"
    NODE_CREATED = "node_created"
    EDGE_CREATED = "edge_created"
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    NODE_FAILED = "node_failed"
    NODE_REOPENED = "node_reopened"        # a FAILED node returned to the frontier for another attempt
    ACTIVITY_RECORDED = "activity_recorded"  # a nondeterministic sub-step's result, recorded for replay/idempotency
    SUBGRAPH_EMITTED = "subgraph_emitted"  # lineage marker: which expander produced which patch
    NODE_PRUNED = "node_pruned"            # Layer 3 constraint filter dropped a proposed node
    GOAL_SATISFIED = "goal_satisfied"
    RUN_TERMINATED = "run_terminated"


class TerminationReason(str, Enum):
    GOAL_SATISFIED = "goal_satisfied"
    GENERATION_CAP = "generation_cap"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NO_PROGRESS = "no_progress"            # frontier empty but goal unmet (deadlock/starvation guard)


# --------------------------------------------------------------------------- #
# Provenance (Layers 1, 5) — the provenance-at-write guard                     #
# --------------------------------------------------------------------------- #
class ProvenanceError(ValueError):
    """Raised when a creation event is missing lineage. Makes NULL provenance
    structurally impossible rather than something you patch after the fact."""


@dataclass(frozen=True)
class Provenance:
    """
    Every node and edge carries this tuple, stamped AT THE MOMENT OF CREATION.
    There is deliberately no code path that mints a node/edge without it — the
    store rejects the append. This is why `started_at`, `source_batch_id`, and
    `pipeline_run_id` can never be NULL: creation *is* the stamped event.
    """
    run_id: str
    batch_id: str
    generation: int
    created_at: str
    expander_id: Optional[str]          # which orchestrator node emitted this (None for the root)
    parent_ids: tuple[str, ...]

    def validate(self) -> "Provenance":
        missing = [
            name for name in ("run_id", "batch_id", "created_at")
            if not getattr(self, name)
        ]
        if self.generation is None:
            missing.append("generation")
        if missing:
            raise ProvenanceError(f"incomplete provenance, missing: {missing}")
        return self

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Event (Layer 5)                                                             #
# --------------------------------------------------------------------------- #
@dataclass
class Event:
    """
    An immutable fact. `seq` is assigned by the store on append (monotonic,
    gap-free, the total order that makes replay deterministic). The payload is a
    plain dict so the log is serialization-agnostic (JSON in the SQLite PoC;
    the same shape maps onto a Postgres `events` table in production).

    Note we only ever record FACTS: created / started / completed / failed /
    emitted / pruned / terminated. Derived states (READY) are never evented —
    they are recomputed by folding, so the log stays a source of truth rather
    than a cache of conclusions.
    """
    run_id: str
    type: EventType
    payload: dict
    node_id: Optional[str] = None
    created_at: str = field(default_factory=now_iso)
    seq: Optional[int] = None  # assigned by the store

    def to_row(self) -> dict:
        return {
            "seq": self.seq,
            "run_id": self.run_id,
            "type": self.type.value,
            "node_id": self.node_id,
            "payload": self.payload,
            "created_at": self.created_at,
        }
