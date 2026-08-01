"""
executors.py — agents as node-type executors (Layer 4) + cost routing (Layer 5).

An Executor is the code bound to a NodeType. "Agent = executor bound to a node
type; complex agent = nested subgraph" is the approved Layer-4 decision, so the
role executors here are deliberately small and pure; a heavier agent would
return a SubgraphPatch that expands into its own sub-DAG.

Executors are DETERMINISTIC on purpose. The PoC proves the *orchestration
substrate*, so the "model call" is a fixed transform. A real model plugs in at
exactly one seam (the executor body); everything above — scheduling, provenance,
steering, durability — is unchanged. Determinism also lets the replay test
compare graph structure without wrestling nondeterministic LLM output.

Two Layer-3 mechanisms live here as well:
  * the critic (ReviewerExecutor) emits the error signal that steers construction
  * refinement is expressed as a NEW SSA node (engineer.g_{k+1}), never a cycle
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from .core import NodeType, Signal, Tier
from .nodes import NodeSpec, SubgraphPatch


# --------------------------------------------------------------------------- #
# Execution result + context                                                  #
# --------------------------------------------------------------------------- #
class MalformedResultError(ValueError):
    """An executor returned something that is not a valid ExecutionResult — the
    wrong type, an untyped signal, a non-numeric score, or a malformed patch. The
    scheduler treats this as a FAILED attempt and never folds it into the log or
    graph (fail-closed). The mock fleet always returns valid results, but a real
    model will not: this is the boundary where structured-output validation and a
    JSON-mode/repair loop attach (SPEC §9, item 2), so the messiness of a model
    can never corrupt the substrate below it."""


@dataclass
class ExecutionResult:
    output: Any
    signal: Signal = Signal.CONTINUE           # TYPED completion — never parsed from text
    score: Optional[float] = None              # critic value (Layer 3)
    patch: Optional[SubgraphPatch] = None      # on-the-fly construction (Layer 2)


@dataclass
class Context:
    run_id: str
    goal: dict
    self_id: str          # the running node's concrete id (to reference itself in patches)
    generation: int


class Executor(ABC):
    @abstractmethod
    def run(self, inputs: dict[str, Any], params: dict, ctx: Context) -> ExecutionResult: ...


# --------------------------------------------------------------------------- #
# Model router — the cost / local-vs-cloud decision (Layer 5)                  #
# --------------------------------------------------------------------------- #
@dataclass
class ModelBinding:
    name: str
    tier: Tier
    cost: float


class ModelRouter:
    """
    Routing is a property of the NODE TYPE on the frontier, not a global setting.
    High-fan-out / low-stakes roles route to a cheap local model; the roles that
    GATE THE OUTCOME (critic, aggregator, planner) route to an expensive cloud
    model. This is how a capped local-model budget is preserved: capability is
    spent only where it decides whether the target is met.
    """
    _POLICY: dict[NodeType, ModelBinding] = {
        NodeType.GOAL:         ModelBinding("none",        Tier.LOCAL, 0.0),
        NodeType.SCOUT:        ModelBinding("local-8b",    Tier.LOCAL, 1.0),
        NodeType.ENGINEER:     ModelBinding("local-8b",    Tier.LOCAL, 1.0),
        NodeType.ORCHESTRATOR: ModelBinding("cloud-large", Tier.CLOUD, 5.0),
        NodeType.REVIEWER:     ModelBinding("cloud-large", Tier.CLOUD, 5.0),
        NodeType.SYNTHESIZER:  ModelBinding("cloud-large", Tier.CLOUD, 5.0),
    }
    _DEFAULT = ModelBinding("local-8b", Tier.LOCAL, 1.0)

    def route(self, node_type: NodeType) -> ModelBinding:
        return self._POLICY.get(node_type, self._DEFAULT)


# --------------------------------------------------------------------------- #
# Role executors                                                              #
# --------------------------------------------------------------------------- #
class GoalSourceExecutor(Executor):
    """The root input source. Holds the goal spec so downstream nodes can read
    it as an ordinary upstream value."""
    def run(self, inputs, params, ctx) -> ExecutionResult:
        return ExecutionResult(output=params["goal"], signal=Signal.CONTINUE)


class OrchestratorExecutor(Executor):
    """
    The expander (Layer 2, primitive 4). Reads the goal and EMITS the initial
    subgraph: one scout per subtopic (fan-out) -> an engineer that collects them
    (fan-in) -> a reviewer that will gate the result. It produces no content
    value itself — its 'output' is structure.
    """
    def run(self, inputs, params, ctx) -> ExecutionResult:
        goal = ctx.goal
        subtopics: list[str] = goal["subtopics"]
        target = goal["target_score"]

        scouts = [
            NodeSpec(
                key=f"scout_{st}",
                node_type=NodeType.SCOUT,
                input_keys=(ctx.self_id,),          # depend on this orchestrator node
                params={"subtopic": st, "topic": goal["topic"]},
            )
            for st in subtopics
        ]
        engineer = NodeSpec(
            key="engineer",
            node_type=NodeType.ENGINEER,
            input_keys=tuple(s.key for s in scouts),  # fan-in over the scouts
            params={"topic": goal["topic"], "subtopics": subtopics, "revision": 0},
        )
        reviewer = NodeSpec(
            key="reviewer",
            node_type=NodeType.REVIEWER,
            input_keys=("engineer",),
            # constraint propagation: the target flows down to where it is checked
            params={"target_score": target, "subtopics": subtopics},
        )
        patch = SubgraphPatch(added_nodes=[*scouts, engineer, reviewer])
        return ExecutionResult(
            output={"plan": [n.key for n in patch.added_nodes]},
            signal=Signal.CONTINUE,
            patch=patch,
        )


class ScoutExecutor(Executor):
    """Frontier retrieval. Deterministic 'fetch' of material for one subtopic."""
    def run(self, inputs, params, ctx) -> ExecutionResult:
        st = params["subtopic"]
        material = {
            "subtopic": st,
            "facts": [f"{st}:fact-{i}" for i in range(1, 4)],
            "source": f"src://{params['topic']}/{st}",
        }
        return ExecutionResult(output=material, signal=Signal.CONTINUE)


class EngineerExecutor(Executor):
    """
    Generation / refinement. Drafts a brief from the scouts' material. Quality is
    a deterministic function of the revision number: a first pass is shallow;
    incorporating the critic's feedback (revision >= 1) lifts it over the bar.
    This is what makes the critic-controller loop converge in a bounded number of
    generations rather than wandering.
    """
    def run(self, inputs, params, ctx) -> ExecutionResult:
        revision = params.get("revision", 0)
        subtopics = params["subtopics"]
        # scout material arrives as upstream outputs; a real engineer would fuse it.
        covered = sorted(
            v["subtopic"] for v in inputs.values()
            if isinstance(v, dict) and "subtopic" in v
        )
        # If this is a refinement, we also received the prior draft + critique.
        prior = [v for v in inputs.values() if isinstance(v, dict) and v.get("kind") == "draft"]
        base_quality = 0.60
        quality = round(base_quality + 0.30 * min(revision, 1), 3)  # 0.60 -> 0.90
        draft = {
            "kind": "draft",
            "topic": params["topic"],
            "revision": revision,
            "covered": covered or subtopics,
            "quality": quality,
            "note": "revised per critique" if revision else "initial pass",
            "supersedes": prior[0].get("revision") if prior else None,
        }
        return ExecutionResult(output=draft, signal=Signal.CONTINUE)


class ReviewerExecutor(Executor):
    """
    The critic — the controller in the setpoint/error loop (Layer 3). It scores
    the draft against the target (the setpoint). If the error is positive
    (score < target) it does NOT loop; it EMITS a refinement stratum: a new
    engineer version (SSA) fed the failed draft + this critique, plus a fresh
    reviewer to re-measure. If the draft clears the bar it approves and emits the
    aggregation node that will converge the graph to a single outcome.
    """
    def run(self, inputs, params, ctx) -> ExecutionResult:
        target = params["target_score"]
        draft = next(v for v in inputs.values() if isinstance(v, dict) and v.get("kind") == "draft")
        draft_id = next(k for k, v in inputs.items() if v is draft)
        score = draft["quality"]
        error = target - score

        if error > 1e-9:
            # --- steer: emit a refinement subgraph (new SSA nodes, forward edges) ---
            next_rev = draft["revision"] + 1
            refined = NodeSpec(
                key="engineer",  # same role, new generation => engineer.g{k} distinct id
                node_type=NodeType.ENGINEER,
                input_keys=(draft_id, ctx.self_id),      # prior draft + this critique
                params={
                    "topic": draft["topic"],
                    "subtopics": params["subtopics"],
                    "revision": next_rev,
                    "critique": f"raise depth to clear {target:.2f}",
                },
            )
            re_review = NodeSpec(
                key="reviewer",
                node_type=NodeType.REVIEWER,
                input_keys=("engineer",),                 # the refined draft above
                params={"target_score": target, "subtopics": params["subtopics"]},
            )
            patch = SubgraphPatch(added_nodes=[refined, re_review])
            return ExecutionResult(
                output={"verdict": "reject", "score": score, "error": round(error, 3)},
                signal=Signal.NEEDS_REFINEMENT,
                score=score,
                patch=patch,
            )

        # --- approve: emit the aggregator that converges to the terminal output ---
        synth = NodeSpec(
            key="synthesizer",
            node_type=NodeType.SYNTHESIZER,
            input_keys=(draft_id, ctx.self_id),           # approved draft + approval
            params={"subtopics": params["subtopics"], "target_score": target},
        )
        return ExecutionResult(
            output={"verdict": "approve", "score": score},
            signal=Signal.DONE,
            score=score,
            patch=SubgraphPatch(added_nodes=[synth]),
        )


class SynthesizerExecutor(Executor):
    """Aggregator. Merges the approved draft into the terminal brief and raises
    the GOAL signal; the scheduler then checks the goal predicate before
    accepting termination."""
    def run(self, inputs, params, ctx) -> ExecutionResult:
        draft = next(v for v in inputs.values() if isinstance(v, dict) and v.get("kind") == "draft")
        brief = {
            "kind": "brief",
            "topic": draft["topic"],
            "covered": draft["covered"],
            "final_score": draft["quality"],
            "sections": {st: f"section on {st}" for st in draft["covered"]},
        }
        return ExecutionResult(output=brief, signal=Signal.GOAL, score=draft["quality"])


def default_registry() -> dict[NodeType, Executor]:
    return {
        NodeType.GOAL: GoalSourceExecutor(),
        NodeType.ORCHESTRATOR: OrchestratorExecutor(),
        NodeType.SCOUT: ScoutExecutor(),
        NodeType.ENGINEER: EngineerExecutor(),
        NodeType.REVIEWER: ReviewerExecutor(),
        NodeType.SYNTHESIZER: SynthesizerExecutor(),
    }
