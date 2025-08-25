#!/usr/bin/env python3
# All comments in English.

import argparse
from es_index import get_es_client, ensure_index
from ingest import ingest_all, ingest_task

def main():
    parser = argparse.ArgumentParser(description="Ingest lm-eval style datasets into Elasticsearch.")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="List of HF datasets to ingest (e.g., hellaswag arc_challenge mmlu kmmlu). "
                             "If omitted, DEFAULT_TASKS from config.py is used.")
    parser.add_argument("--split", default=None,
                        help="HF split to ingest (default: use config DEFAULT_SPLIT)")
    parser.add_argument("--subset", default=None,
                        help="HF subset to ingest (e.g., rte for super_glue, cola for glue)")
    args = parser.parse_args()

    es = get_es_client()
    if not ensure_index(es):
        print("Failed to ensure index exists. Exiting.")
        return

    if args.tasks:
        for t in args.tasks:
            ingest_task(es, t, split=args.split, subset=args.subset)
    else:
        # Use DEFAULT_TASKS defined in config.py
        ingest_all(es)

if __name__ == "__main__":
    main()
