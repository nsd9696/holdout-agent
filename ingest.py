# ingest.py
# Robust ingestion pipeline: load datasets, infer metadata, embed, and bulk index to Elasticsearch.

import hashlib
from tqdm.auto import tqdm
from elasticsearch import helpers
from datasets import load_dataset
from es_index import get_es_client, ensure_index
from meta_infer import infer_topic_and_difficulty, embed_text
from config import ES_INDEX, BULK_BATCH, DEFAULT_TASKS, TASK_REPO_MAPPING, DEFAULT_SPLIT

def get_task_info(task_name: str) -> dict:
    """Get the complete task information including repo, split, and subset."""
    for mapping in TASK_REPO_MAPPING:
        if mapping["task_name"] == task_name:
            return mapping
    # If not found in mapping, return default values
    print(f"Warning: No task mapping found for task '{task_name}', using defaults")
    return {
        "task_name": task_name,
        "huggingface_repo": task_name,
        "split": DEFAULT_SPLIT,
        "subset": None
    }

def get_repo_for_task(task_name: str) -> str:
    """Get the huggingface repository name for a given task name."""
    task_info = get_task_info(task_name)
    return task_info["huggingface_repo"]

def row_id(benchmark: str, subset: str, local_id: str) -> str:
    """Deterministic id for each indexed row to avoid duplicates."""
    base = f"{benchmark}::{subset or ''}::{local_id}"
    return hashlib.sha1(base.encode()).hexdigest()

def normalize_example(ex: dict) -> dict:
    """
    Normalize a dataset example to a common dict with 'text' and 'answer' fields.
    This covers many lm-eval style datasets (prompt/question/choices...).
    """
    # try standard keys
    for text_key in ("question", "prompt", "input", "text", "query"):
        if ex.get(text_key):
            text = ex[text_key]
            break
    else:
        # fallback: join all string fields
        text_items = []
        for k, v in ex.items():
            if isinstance(v, str) and len(v) < 2000:
                text_items.append(f"{k}: {v}")
        text = "\n".join(text_items) if text_items else str(ex)

    answer = ex.get("answer") or ex.get("label") or ex.get("target") or ""
    return {"text": text, "answer": str(answer), "raw_example": ex}

def ingest_task(es_client, task_name: str, split: str = None, subset: str = None):
    """
    Ingest one HF dataset split. For lm-eval tasks that have other names,
    the caller should supply the correct dataset id.
    """
    # Get the complete task information
    task_info = get_task_info(task_name)
    
    # Use provided parameters or fall back to config defaults
    repo_name = task_info["huggingface_repo"]
    split_to_use = split or task_info["split"]
    subset_to_use = subset or task_info["subset"]
    
    print(f"Ingesting task={task_name} (repo={repo_name}) split={split_to_use} subset={subset_to_use}")
    
    try:
        if subset_to_use:
            # Load dataset with subset (e.g., super_glue with rte subset)
            ds = load_dataset(repo_name, subset_to_use, split=split_to_use)
        else:
            # Load dataset without subset
            ds = load_dataset(repo_name, split=split_to_use)
    except Exception as e:
        print(f"Error loading dataset {repo_name} (subset={subset_to_use}, split={split_to_use}): {e}")
        print(f"Trying fallback with task name {task_name}")
        try:
            ds = load_dataset(task_name, split=split_to_use)
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            return
    
    actions = []
    for i, ex in enumerate(tqdm(ds, desc=f"{task_name}:{split_to_use}")):
        norm = normalize_example(ex)
        meta = infer_topic_and_difficulty(norm["text"])
        emb = embed_text(norm["text"])
        
        # Generate topic embedding for topic-based filtering
        topic_emb = embed_text(meta["topic"])
        doc = {
            "benchmark": task_name,
            "subset": ex.get("subject") or ex.get("category") or subset_to_use,
            "split_hint": split_to_use,
            "id_in_benchmark": str(i),
            "text": norm["text"],
            "answer": norm["answer"],
            "topic": meta["topic"],
            "topic_embedding": topic_emb,  # Add topic embedding for relevance filtering
            "difficulty": meta["difficulty"],
            "raw_example": norm["raw_example"],
            "embedding": emb
        }
        doc_id = row_id(task_name, doc["subset"], doc["id_in_benchmark"])
        actions.append({"_index": ES_INDEX, "_id": doc_id, "_source": doc})

        if len(actions) >= BULK_BATCH:
            helpers.bulk(es_client, actions)
            actions = []
    if actions:
        helpers.bulk(es_client, actions)

def ingest_all(es_client, tasks = None):
    """Ingest all tasks in DEFAULT_TASKS or provided list."""
    tasks = tasks or DEFAULT_TASKS
    for t in tasks:
        ingest_task(es_client, t)  # Use config defaults for split and subset
