# config.py
# Central configuration & environment helpers.

import os

# Elasticsearch connection settings - read from env
ES_ENDPOINT = os.getenv("ES_ENDPOINT", "http://localhost:9201")
ES_USER = os.getenv("ES_USER", "elastic")
ES_PASS = os.getenv("ES_PASS", "changeme")
ES_INDEX = os.getenv("ES_INDEX", "llm_bench_rows")

# OpenAI models used
OPENAI_TAG_MODEL = os.getenv("OPENAI_TAG_MODEL", "gpt-4o-mini")         # model to infer topic/difficulty
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")  # embedding model

# Embedding dimensionality for the chosen model
EMBED_DIMS = int(os.getenv("EMBED_DIMS", 3072))

# ingestion batching
BULK_BATCH = int(os.getenv("BULK_BATCH", 256))

# Default split to use for ingestion
DEFAULT_SPLIT = os.getenv("DEFAULT_SPLIT", "validation")

# Topic relevance threshold for embedding-based filtering (0.0 to 1.0)
TOPIC_RELEVANCE_THRESHOLD = float(os.getenv("TOPIC_RELEVANCE_THRESHOLD", "0.01"))

# Search configuration for better candidate retrieval
SEARCH_MIN_SCORE_THRESHOLD = float(os.getenv("SEARCH_MIN_SCORE_THRESHOLD", "0.01"))
SEARCH_FALLBACK_MULTIPLIER = float(os.getenv("SEARCH_FALLBACK_MULTIPLIER", "1.5"))
SEARCH_MAX_ATTEMPTS = int(os.getenv("SEARCH_MAX_ATTEMPTS", "5"))

# Task name to HuggingFace repository mapping with split and subset info
TASK_REPO_MAPPING = [
    {
        "task_name": "hellaswag", 
        "huggingface_repo": "hellaswag",
        "split": "validation",
        "subset": None
    },
    {
        "task_name": "arc_challenge", 
        "huggingface_repo": "ai2_arc",
        "split": "validation",
        "subset": None
    },
    {
        "task_name": "mmlu", 
        "huggingface_repo": "cais/mmlu",
        "split": "validation",
        "subset": None
    },
    {
        "task_name": "kmmlu", 
        "huggingface_repo": "HAERAE-HUB/KMMLU",
        "split": "train",
        "subset": None
    },
    {
        "task_name": "gsm8k",
        "huggingface_repo": "openai/gsm8k",
        "split": "test",
        "subset": None
    },
    {
        "task_name": "ai2_arc",
        "huggingface_repo": "allenai/ai2_arc",
        "split": "test",
        "subset": None
    }
]

# default LM-EVAL tasks to ingest (including kmmlu)
DEFAULT_TASKS = ["hellaswag", "arc_challenge", "mmlu", "kmmlu", "gsm8k"]


