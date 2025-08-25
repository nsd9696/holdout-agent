#!/usr/bin/env python3
# All comments in English.

import argparse, json
from agent_runner import run_agent_sync
from split_engine import SplitSpec
from meta_infer import embed_text
from es_index import get_es_client
from split_engine import plan_split, to_jsonl
from config import TASK_REPO_MAPPING

def parse_difficulty_mix(mix_str: str):
    """
    Parse mix string like: "1:0.05,2:0.15,3:0.5,4:0.2,5:0.1" -> {1:0.05,...}
    """
    out = {}
    for part in mix_str.split(","):
        k, v = part.split(":")
        out[int(k.strip())] = float(v.strip())
    return out

def get_task_info(task_name: str) -> dict:
    """Get the complete task information including repo, split, and subset."""
    for mapping in TASK_REPO_MAPPING:
        if mapping["task_name"] == task_name:
            return mapping
    return None

def main():
    parser = argparse.ArgumentParser(description="Construct held-in/held-out splits via the Agent or directly.")
    parser.add_argument("--goal", required=True, help="High-level evaluation goal.")
    parser.add_argument("--benchmarks", nargs="*", default=None, help="Optional benchmark filters (e.g., hellaswag arc_challenge kmmlu)")
    parser.add_argument("--topics", nargs="*", default=None, help="Optional topic filters (e.g., commonsense science)")
    parser.add_argument("--total", type=int, default=300, help="Total number of items across both splits.")
    parser.add_argument("--heldin-ratio", type=float, default=0.5, help="Proportion for held-in (0~1).")
    parser.add_argument("--difficulty-mix", default="1:0.05,2:0.15,3:0.5,4:0.2,5:0.1",
                        help="Difficulty mix as '1:0.05,2:0.15,3:0.5,4:0.2,5:0.1'")
    parser.add_argument("--mode", choices=["agent", "direct"], default="agent",
                        help="agent: use OpenAI Agent tools; direct: call splitter directly (no orchestration).")
    args = parser.parse_args()

    topics = args.topics or []
    benches = args.benchmarks or []
    difficulty_mix = parse_difficulty_mix(args.difficulty_mix)

    # Display benchmark information if specified
    if benches:
        print("=== Benchmark Information ===")
        for bench in benches:
            task_info = get_task_info(bench)
            if task_info:
                subset_info = f" (subset: {task_info['subset']})" if task_info['subset'] else ""
                print(f"  {bench}: {task_info['huggingface_repo']}{subset_info}")
            else:
                print(f"  {bench}: No mapping found")
        print()

    if args.mode == "agent":
        # Use the Agent pipeline (OpenAI Agent SDK)
        payload = {
            "goal": args.goal,
            "total": args.total,
            "difficulty_mix": {str(k): v for k, v in difficulty_mix.items()},
            "topics": topics,
            "benchmarks": benches,
            "heldin_ratio": args.heldin_ratio
        }
        result = run_agent_sync(payload)
        print("=== Agent result (tool outputs are included in the trace) ===")
        print(result)
        print("\nTip: look for 'held_in_path' and 'held_out_path' in the final tool output.")
    else:
        # Direct mode: same split logic without Agent overhead.
        from config import ES_INDEX
        es = get_es_client()
        qv = embed_text(args.goal)
        spec = SplitSpec(
            goal=args.goal, total=args.total, difficulty_mix=difficulty_mix,
            topics=topics, benchmarks=benches, heldin_ratio=args.heldin_ratio
        )
        held_in, held_out = plan_split(es, qv, spec)
        # Write two local files
        with open("held_in.jsonl", "w") as f:
            f.write(to_jsonl(held_in))
        with open("held_out.jsonl", "w") as f:
            f.write(to_jsonl(held_out))
        print("Wrote held_in.jsonl and held_out.jsonl")

if __name__ == "__main__":
    main()
