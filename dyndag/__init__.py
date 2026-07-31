"""dyndag — a dynamic-DAG multi-agent orchestrator (proof of concept).

Public surface:
    from dyndag import Scheduler, SQLiteEventStore, default_registry, Budget
"""
from .core import (NodeType, NodeState, Signal, Tier, TerminationReason,
                   Provenance, Event, EventType)
from .store import SQLiteEventStore, EventStore
from .graph import GraphProjection
from .nodes import NodeSpec, SubgraphPatch
from .executors import (Executor, ExecutionResult, Context, ModelRouter,
                        default_registry)
from .steering import CoverageGoal, ConstraintFilter
from .governor import Budget, CostGovernor
from .scheduler import Scheduler, RunResult
from .views import render_blackboard, verify_replay, lineage_report, provenance_gaps

__all__ = [
    "Scheduler", "RunResult", "SQLiteEventStore", "EventStore", "GraphProjection",
    "NodeSpec", "SubgraphPatch", "Executor", "ExecutionResult", "Context",
    "ModelRouter", "default_registry", "CoverageGoal", "ConstraintFilter",
    "Budget", "CostGovernor", "NodeType", "NodeState", "Signal", "Tier",
    "TerminationReason", "Provenance", "Event", "EventType",
    "render_blackboard", "verify_replay", "lineage_report", "provenance_gaps",
]
