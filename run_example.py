# run_example.py
# Example usage: ensure Elasticsearch index exists, ingest tasks including kmmlu, then run an agent request.

from es_index import get_es_client, ensure_index
from ingest import ingest_all
from agent_runner import run_agent_sync
from config import TASK_REPO_MAPPING

def get_task_info(task_name: str) -> dict:
    """Get the complete task information including repo, split, and subset."""
    for mapping in TASK_REPO_MAPPING:
        if mapping["task_name"] == task_name:
            return mapping
    return None

def main():
    es = get_es_client()
    ensure_index(es)

    # Ingest datasets (this step may take time)
    # You can comment this out if you've already ingested previously.
    ingest_all(es)  # uses DEFAULT_TASKS which includes kmmlu

    # Build a sample agent request
    payload = {
        "goal": "Evaluate commonsense reasoning and multi-choice science questions to test generalization.",
        "total": 300,
        "difficulty_mix": {"1": 0.05, "2": 0.15, "3": 0.5, "4": 0.2, "5": 0.1},
        "topics": ["commonsense", "science"],    # optional
        "benchmarks": ["hellaswag", "arc_challenge", "kmmlu"],  # optional
        "heldin_ratio": 0.5
    }
    
    # Display benchmark information
    print("=== Benchmark Information ===")
    for bench in payload["benchmarks"]:
        task_info = get_task_info(bench)
        if task_info:
            subset_info = f" (subset: {task_info['subset']})" if task_info['subset'] else ""
            print(f"  {bench}: {task_info['huggingface_repo']}{subset_info}")
        else:
            print(f"  {bench}: No mapping found")
    print()
    
    res = run_agent_sync(payload)
    print("Agent run result:", res)

if __name__ == "__main__":
    main()
