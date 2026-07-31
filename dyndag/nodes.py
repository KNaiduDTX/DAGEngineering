"""
nodes.py — the typed node model and the subgraph-emission patch (Layers 1, 2).

Two objects live here:

  NodeSpec       — a *request* to create a node. An expander emits these; the
                   scheduler turns each into a NODE_CREATED event with full
                   provenance. A NodeSpec is pure data: role, inputs, params.

  SubgraphPatch  — the payload of the chosen construction primitive (Layer 2,
                   highest order). An expander's output is not a value but a
                   patch: V+ (added_nodes) and E+ (added_edges). This is
                   "propagation of nodes and edges on the fly, computed as
                   downstream variables of prior nodes" made literal.

Placeholder resolution implements primitive (1) operating *underneath* (4): a
node's params may reference upstream outputs by producer id ("$scout_a"), and
those references are substituted with real values at dispatch time — exactly the
LLMCompiler $1/$2 mechanism, but scoped to a single node's parameter dict.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .core import NodeType


_PLACEHOLDER = re.compile(r"\$([A-Za-z0-9_.#>-]+)")


@dataclass
class NodeSpec:
    """A request to instantiate one node. `key` is the logical label used to
    derive the stable id; `input_keys` are the *logical* producer keys this node
    depends on (resolved to concrete node ids when the patch is applied)."""
    key: str
    node_type: NodeType
    input_keys: tuple[str, ...] = ()      # logical dependencies (become in-edges)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubgraphPatch:
    """The unit of on-the-fly construction. `added_nodes` is V+, `added_edges`
    is E+ (as logical (producer_key -> consumer_key) pairs). Edges are usually
    implied by input_keys; explicit edges are allowed for cross-links.

    Invariant enforced downstream: every added node's generation is
    max(parent.generation)+1, which makes the graph generation-monotonic and
    therefore acyclic by construction (Layer 6)."""
    added_nodes: list[NodeSpec] = field(default_factory=list)
    added_edges: list[tuple[str, str]] = field(default_factory=list)

    def edges(self) -> list[tuple[str, str]]:
        """All edges implied by the patch: explicit ones plus one per input_key."""
        implied = [(ik, n.key) for n in self.added_nodes for ik in n.input_keys]
        return list({*self.added_edges, *implied})


def resolve_placeholders(params: dict[str, Any], outputs: dict[str, Any]) -> dict[str, Any]:
    """
    Substitute `$producer_key` references in a params dict with the completed
    output of that producer. `outputs` maps producer node_id -> output value.
    Strings are template-substituted; a bare "$key" string is replaced by the
    raw object (so structured outputs survive, not just their str()).
    """
    def sub_one(value: Any) -> Any:
        if isinstance(value, str):
            m = _PLACEHOLDER.fullmatch(value.strip())
            if m and m.group(1) in outputs:            # whole-string ref -> raw object
                return outputs[m.group(1)]
            return _PLACEHOLDER.sub(                    # inline refs -> stringified
                lambda mm: str(outputs.get(mm.group(1), mm.group(0))), value
            )
        if isinstance(value, dict):
            return {k: sub_one(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(sub_one(v) for v in value)
        return value

    return {k: sub_one(v) for k, v in params.items()}
