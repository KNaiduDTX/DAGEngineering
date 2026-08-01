"""dyndag — a dynamic-DAG multi-agent orchestrator (proof of concept).

Public surface:
    from dyndag import Scheduler, SQLiteEventStore, default_registry, Budget
"""
from .core import (NodeType, NodeState, Signal, Tier, TerminationReason,
                   Provenance, Event, EventType)
from .store import SQLiteEventStore, EventStore
from .graph import GraphProjection
from .nodes import NodeSpec, SubgraphPatch
from .activities import ActivityRecorder
from .executors import (Executor, ExecutionResult, Context, ModelRouter,
                        MalformedResultError, TerminalExecutionError, default_registry)
from .executors_ollama import (OllamaEngineerExecutor, OllamaReviewerExecutor,
                               RetrievalScoutExecutor, ollama_registry)
from .executors_openrouter import (OpenRouterEngineerExecutor, OpenRouterReviewerExecutor,
                                   OpenRouterOrchestratorExecutor, openrouter_registry,
                                   http_transport)
from .steering import CoverageGoal, ConstraintFilter
from .governor import Budget, CostGovernor
from .scheduler import Scheduler, RunResult
from .views import (render_blackboard, verify_replay, lineage_report,
                    provenance_gaps, cost_report)
from .evaluation import (EvalTask, TaskResult, DEFAULT_TASKS, run_task, run_eval,
                         cost_routing_report, savings_pct)

__all__ = [
    "Scheduler", "RunResult", "SQLiteEventStore", "EventStore", "GraphProjection",
    "NodeSpec", "SubgraphPatch", "Executor", "ExecutionResult", "Context",
    "ActivityRecorder", "OllamaEngineerExecutor", "OllamaReviewerExecutor",
    "RetrievalScoutExecutor", "ollama_registry",
    "OpenRouterEngineerExecutor", "OpenRouterReviewerExecutor",
    "OpenRouterOrchestratorExecutor", "openrouter_registry", "http_transport",
    "ModelRouter", "MalformedResultError", "TerminalExecutionError",
    "default_registry", "CoverageGoal", "ConstraintFilter",
    "Budget", "CostGovernor", "NodeType", "NodeState", "Signal", "Tier",
    "TerminationReason", "Provenance", "Event", "EventType",
    "render_blackboard", "verify_replay", "lineage_report", "provenance_gaps",
    "cost_report",
    "EvalTask", "TaskResult", "DEFAULT_TASKS", "run_task", "run_eval",
    "cost_routing_report", "savings_pct",
]
