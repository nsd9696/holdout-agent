# agent_runner.py
# Agent wrapping: exposes tools to the OpenAI Agents SDK and runs the pipeline.

import tempfile
from typing import List, Optional
from openai import OpenAI
from agents import Agent, function_tool, Runner
from es_index import get_es_client
from meta_infer import embed_text
from split_engine import SplitSpec, plan_split, to_jsonl
from config import OPENAI_TAG_MODEL
from pydantic import BaseModel, Field

es_client = get_es_client()
openai_client = OpenAI()

# -------- New: strict schema-friendly models --------
class DifficultyWeight(BaseModel):
    """One difficulty bucket with its target ratio."""
    level: int = Field(..., ge=1, le=5, description="Difficulty level (1~5)")
    ratio: float = Field(..., ge=0.0, le=1.0, description="Proportion for this level (0..1)")

    class Config:
        extra = "forbid"  # disallow additional properties


@function_tool
def es_search_by_goal(
    goal: str,
    topics: Optional[List[str]] = None,
    benchmarks: Optional[List[str]] = None,
    k: int = 2000,
):
    """
    Tool used by the agent: embed the goal and return top-k candidate docs as sources.
    """
    qv = embed_text(goal)
    # call ES directly using the split engine helper knn_search pattern via plan_split expectations
    spec = SplitSpec(
        goal=goal,
        total=0,
        difficulty_mix={1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2},
        topics=topics or [],
        benchmarks=benchmarks or [],
    )
    hits = plan_split.__globals__['knn_search'](
        es_client,
        qv,
        k=k,
        filters={"topic": spec.topics, "benchmark": spec.benchmarks},
    )
    # return smaller representation
    simplified = [
        {
            "benchmark": h["_source"]["benchmark"],
            "id": h["_source"]["id_in_benchmark"],
            "text": h["_source"]["text"][:400],
            "_score": h["_score"],
        }
        for h in hits[:200]
    ]
    return simplified


@function_tool
def construct_and_export(
    goal: str,
    total: int,
    difficulty_mix: List[DifficultyWeight],
    topics: Optional[List[str]] = None,
    benchmarks: Optional[List[str]] = None,
    heldin_ratio: float = 0.5,
):
    """
    Agent-facing tool: orchestrates plan_split and writes temporary JSONL files for both sets.
    Returns paths to the JSONL files.
    """
    # Convert list of DifficultyWeight -> {level: ratio}
    mix = {int(item.level): float(item.ratio) for item in difficulty_mix}

    # build spec
    spec = SplitSpec(
        goal=goal,
        total=total,
        difficulty_mix=mix,
        topics=topics,
        benchmarks=benchmarks,
        heldin_ratio=heldin_ratio,
    )

    # embed goal
    qv = embed_text(goal)
    held_in, held_out = plan_split(es_client, qv, spec)

    # write out temporary JSONL files
    t1 = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl")
    t1.write(to_jsonl(held_in).encode()); t1.flush()
    t2 = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl")
    t2.write(to_jsonl(held_out).encode()); t2.flush()

    return {
        "held_in_path": t1.name,
        "held_out_path": t2.name,
        "held_in_count": len(held_in),
        "held_out_count": len(held_out),
    }


def build_agent():
    """
    Create an Agent configured with the above tools. The agent acts as a coordinator,
    not to make the split decisions itself beyond invoking the tools.
    """
    return Agent(
        name="Holdout Agent",
        instructions=(
            "You are a dataset curator assistant. "
            "When given a goal, total size, difficulty_mix, optional topics/benchmarks and heldin_ratio, "
            "use the provided tools to search the corpus and construct held-in and held-out JSONL files. "
            "Do NOT invent data; only call the tools."
        ),
        model=OPENAI_TAG_MODEL,
        tools=[es_search_by_goal, construct_and_export],
    )


def run_agent_sync(payload: dict):
    """
    payload should include keys:
      - goal (str)
      - total (int)
      - difficulty_mix (list[{'level': int 1..5, 'ratio': float 0..1}])
      - topics (list[str]) optional
      - benchmarks (list[str]) optional
      - heldin_ratio (float) optional
    """
    agent = build_agent()
    runner = Runner()
    prompt = f"Please create held-in/out using parameters:\n{payload}"
    result = runner.run_sync(agent=agent, input=prompt)
    # Runner returns agent activity; the tools return the JSON with file paths -> find final tool output in result
    return result
