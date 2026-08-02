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
from .phi import PhiExecutor, argmax_score, majority_vote, union_covered
from .roles import (EngineerRole, ReviewerRole, OrchestratorRole,
                    RetrievalScoutExecutor, chat_json_with_repair)
from .executors_ollama import ollama_chat, ollama_registry
from .executors_openrouter import openrouter_chat, openrouter_registry, http_transport
from .steering import CoverageGoal, ConstraintFilter, AnswerGoal, GoalPredicate
from .governor import Budget, CostGovernor
from .scheduler import Scheduler, RunResult
from .views import (render_blackboard, verify_replay, lineage_report,
                    provenance_gaps, cost_report)
from .evaluation import (EvalTask, TaskResult, DEFAULT_TASKS, run_task, run_eval,
                         cost_routing_report, savings_pct, accuracy, cost_per_correct,
                         gsm8k_report, mcnemar, paired_report, per_problem_rows)
from .tasks_gsm8k import (gsm8k_registry, load_gsm8k, sample_tasks, SolverRole,
                          CheckerRole, CritiqueRole, AnswerExecutor, GsmOrchestrator)

__all__ = [
    "Scheduler", "RunResult", "SQLiteEventStore", "EventStore", "GraphProjection",
    "NodeSpec", "SubgraphPatch", "Executor", "ExecutionResult", "Context",
    "ActivityRecorder", "EngineerRole", "ReviewerRole", "OrchestratorRole",
    "RetrievalScoutExecutor", "chat_json_with_repair",
    "ollama_chat", "ollama_registry",
    "openrouter_chat", "openrouter_registry", "http_transport",
    "ModelRouter", "MalformedResultError", "TerminalExecutionError",
    "default_registry", "CoverageGoal", "ConstraintFilter", "AnswerGoal", "GoalPredicate",
    "Budget", "CostGovernor", "NodeType", "NodeState", "Signal", "Tier",
    "TerminationReason", "Provenance", "Event", "EventType",
    "render_blackboard", "verify_replay", "lineage_report", "provenance_gaps",
    "cost_report",
    "EvalTask", "TaskResult", "DEFAULT_TASKS", "run_task", "run_eval",
    "cost_routing_report", "savings_pct", "accuracy", "cost_per_correct", "gsm8k_report",
    "mcnemar", "paired_report", "per_problem_rows",
    "gsm8k_registry", "load_gsm8k", "sample_tasks", "SolverRole", "CheckerRole",
    "CritiqueRole", "AnswerExecutor", "GsmOrchestrator",
    "PhiExecutor", "argmax_score", "majority_vote", "union_covered",
]
