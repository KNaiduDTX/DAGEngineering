"""
schemas.py — the structured-output contracts real executors constrain models to.

Single source of truth so every provider (Ollama, OpenRouter, …) enforces the same
shapes. All three are deliberately FLAT / bounded-depth: structured-output backends
(Anthropic, OpenAI/OpenRouter) reject recursive schemas, so a role never emits raw
graph structure. The ORCHESTRATOR is the sharp case — instead of emitting a recursive
`SubgraphPatch`, it emits a flat PLAN (which subtopics to scout, how to brief the
engineer, the review rubric) and the executor assembles the patch deterministically.
That is what makes a real planner safe: the model chooses content within a fixed
template; the code builds the DAG.
"""

# ENGINEER output: the draft's covered subtopics + one section of text each.
DRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "covered": {"type": "array", "items": {"type": "string"}},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"subtopic": {"type": "string"}, "text": {"type": "string"}},
                "required": ["subtopic", "text"],
            },
        },
    },
    "required": ["covered", "sections"],
}

# REVIEWER output: a numeric score + a one-line critique.
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {"score": {"type": "number"}, "critique": {"type": "string"}},
    "required": ["score", "critique"],
}

# ORCHESTRATOR output: a flat plan. The model picks the scouts (subtopic + focus),
# the engineer brief, and the review rubric; the executor turns this into the
# scouts -> engineer -> reviewer subgraph. No recursion, no raw NodeSpecs.
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "scouts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"subtopic": {"type": "string"}, "focus": {"type": "string"}},
                "required": ["subtopic", "focus"],
            },
        },
        "engineer_brief": {"type": "string"},
        "review_rubric": {"type": "string"},
    },
    "required": ["scouts", "engineer_brief", "review_rubric"],
}

# SOLVER output (question-answering, e.g. GSM8K): step-by-step reasoning + the final answer.
# `final_answer` is a STRING so "18", "18.0", "$18" all validate; the eval judge normalizes
# before the exact-match comparison — the model is never asked to guess a numeric type.
SOLUTION_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "final_answer": {"type": "string"},
    },
    "required": ["reasoning", "final_answer"],
}

# CHECKER output: did the reasoning check out, how confident (0.0-1.0), and a one-line
# critique to feed the next SSA revision. The checker never sees the gold answer — it judges
# self-consistency of the reasoning, and `confidence` drives the approve/refine gate.
VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "verified": {"type": "boolean"},
        "confidence": {"type": "number"},
        "critique": {"type": "string"},
    },
    "required": ["verified", "confidence", "critique"],
}
