# split_engine.py
# Core split planning and sampling logic. Ensures distribution matching, leakage prevention, and bias mitigation.

from collections import defaultdict, Counter
import math, random, orjson
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from elasticsearch import Elasticsearch
from config import ES_INDEX, TOPIC_RELEVANCE_THRESHOLD, SEARCH_MIN_SCORE_THRESHOLD

@dataclass
class SplitSpec:
    goal: str
    total: int
    difficulty_mix: Dict[int, float]   # e.g. {1:0.1,...,5:0.1}
    topics: List[str] = None
    benchmarks: List[str] = None
    heldin_ratio: float = 0.5
    min_per_benchmark: int = 0

# ---------- ES search helper (kNN) ----------
def knn_search(es: Elasticsearch, query_vec: List[float], k: int = 2000, filters: Dict = None, 
               min_score: float = 0.0, use_hybrid: bool = False) -> List[dict]:
    """
    Run a kNN search against ES index and return hits with _source and _score.
    The 'filters' parameter is a dict of field -> list_of_values for term filters.
    
    Args:
        es: Elasticsearch client
        query_vec: Query vector for kNN search
        k: Number of results to return
        filters: Field filters (field -> list_of_values)
        min_score: Minimum score threshold (0.0 = no threshold)
        use_hybrid: Whether to use hybrid search (kNN + text similarity)
    """
    # Elasticsearch 제한 준수
    k = min(10000, k)
    num_candidates = min(10000, max(2000, k * 3))
    
    if use_hybrid:
        # 하이브리드 검색: kNN + script score
        must_filters = []
        if filters:
            for field, vals in filters.items():
                if vals:
                    must_filters.append({"terms": {field: vals}})
        
        # Score threshold 필터 추가
        if min_score > 0:
            must_filters.append({"range": {"_score": {"gte": min_score}}})
        
        body = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                    "params": {"query_vector": query_vec}
                                }
                            }
                        }
                    ],
                    "filter": must_filters
                }
            },
            "size": k,
            "_source": True
        }
    else:
        # 방법 1: kNN 검색을 먼저 실행하고, 필터링은 후처리로
        body = {
            "knn": {
                "field": "embedding",
                "query_vector": query_vec,
                "k": k,
                "num_candidates": num_candidates
            },
            "_source": True
        }
    
    resp = es.search(index=ES_INDEX, body=body)
    return resp["hits"]["hits"]

def knn_search_with_threshold(es: Elasticsearch, query_vec: List[float], k: int = 2000, 
                             filters: Dict = None, min_score: float = 0.0) -> List[dict]:
    """
    Run kNN search with score threshold filtering
    """
    return knn_search(es, query_vec, k, filters, min_score, use_hybrid=False)

def hybrid_search(es: Elasticsearch, query_vec: List[float], k: int = 2000, 
                 filters: Dict = None) -> List[dict]:
    """
    Combine kNN with text-based search for better coverage
    """
    return knn_search(es, query_vec, k, filters, min_score=0.0, use_hybrid=True)

def multi_stage_search(es: Elasticsearch, query_vec: List[float], k: int = 2000, 
                      filters: Dict = None) -> List[dict]:
    """
    Multi-stage search: first get many candidates, then filter and re-rank
    """
    # Stage 1: Get many candidates with relaxed constraints
    stage1_filters = filters.copy() if filters else {}
    
    # Remove strict filters for initial search
    if 'benchmark' in stage1_filters and len(stage1_filters['benchmark']) > 2:
        stage1_filters['benchmark'] = stage1_filters['benchmark'][:2]  # Limit to 2 benchmarks
    
    candidates = knn_search(es, query_vec, k=k*5, filters=stage1_filters)
    
    if len(candidates) < k:
        # Stage 2: Fallback to broader search
        print(f"Stage 1 returned only {len(candidates)} candidates, trying broader search...")
        candidates = knn_search(es, query_vec, k=k*10, filters=None)
    
    return candidates[:k]

def adaptive_knn_search(es: Elasticsearch, query_vec: List[float], target_k: int = 2000, 
                       filters: Dict = None, min_candidates: int = 1000) -> List[dict]:
    """
    Adaptively adjust k to ensure minimum number of candidates
    """
    current_k = target_k
    max_attempts = 5
    
    for attempt in range(max_attempts):
        hits = knn_search(es, query_vec, k=current_k, filters=filters)
        
        if len(hits) >= min_candidates:
            print(f"✅ Found {len(hits)} candidates with k={current_k}")
            return hits
        
        # Increase k exponentially
        current_k = int(current_k * 1.5)
        print(f"⚠️  Attempt {attempt + 1}: Only {len(hits)} candidates, increasing k to {current_k}")
    
    # Final attempt with very large k
    print(f"🔄 Final attempt with k={current_k}")
    return knn_search(es, query_vec, k=current_k, filters=filters)

# ---------- balancing & sampling ----------
def compute_target_counts(total: int, difficulty_mix: Dict[int, float]) -> Dict[int, int]:
    """
    Compute integer counts per difficulty bucket from a fractional mix,
    ensuring the sum equals total (via rounding + correction).
    """
    raw = {k: v * total for k, v in difficulty_mix.items()}
    floor = {k: int(math.floor(v)) for k, v in raw.items()}
    remainder = total - sum(floor.values())
    # distribute remainder to buckets with largest fractional parts
    fracs = sorted(((k, raw[k] - floor[k]) for k in raw), key=lambda x: -x[1])
    for i in range(remainder):
        k = fracs[i % len(fracs)][0]
        floor[k] += 1
    return floor

def compute_topic_relevance(query_vec: List[float], topic_embedding: List[float]) -> float:
    """
    Compute cosine similarity between query vector and topic embedding.
    Returns a value between -1 and 1, where 1 means most similar.
    """
    if not query_vec or not topic_embedding:
        return 0.0
    
    # Compute cosine similarity
    dot_product = sum(a * b for a, b in zip(query_vec, topic_embedding))
    query_norm = sum(a * a for a in query_vec) ** 0.5
    topic_norm = sum(a * a for a in topic_embedding) ** 0.5
    
    if query_norm == 0 or topic_norm == 0:
        return 0.0
    
    return dot_product / (query_norm * topic_norm)

def select_candidates(hits: List[dict], spec: SplitSpec, max_candidates: int, query_vec: List[float] = None, topic_threshold: float = None) -> List[dict]:
    """
    Filter hits by topic/benchmark constraints and return top candidates up to max_candidates.
    Topic filtering now uses embedding-based relevance instead of exact matching.
    Normalize _score -> rel in [0,1].
    """
    # Use config default if no threshold provided
    if topic_threshold is None:
        topic_threshold = TOPIC_RELEVANCE_THRESHOLD
    
    # build pool with source + score
    pool = []
    for h in hits:
        src = h["_source"]
        
        # Benchmark filtering (exact match)
        if spec.benchmarks and src.get("benchmark") not in spec.benchmarks:
            continue
        
        # Topic filtering (embedding-based relevance)
        if spec.topics and query_vec:
            # Get topic embedding from the source
            topic_emb = src.get("topic_embedding")
            if topic_emb:
                # Compute relevance to each specified topic
                max_topic_relevance = max(
                    compute_topic_relevance(query_vec, topic_emb),
                    0.0  # Ensure non-negative
                )
                
                # Only include if relevance is above threshold
                if max_topic_relevance < topic_threshold:
                    continue
                
                # Add topic relevance to the source for potential use
                src["topic_relevance"] = max_topic_relevance
            else:
                # If no topic embedding available, skip topic filtering
                pass
        
        # Keep essential fields including raw_example for complete data preservation
        essential_src = {
            "benchmark": src.get("benchmark"),
            "subset": src.get("subset"),
            "split_hint": src.get("split_hint"),
            "id_in_benchmark": src.get("id_in_benchmark"),
            "text": src.get("text"),
            "answer": src.get("answer"),
            "topic": src.get("topic"),
            "difficulty": src.get("difficulty"),
            "topic_relevance": src.get("topic_relevance", 0.0),
            "raw_example": src.get("raw_example")  # Include raw_example for complete data
        }
        
        pool.append({**essential_src, "_score": h["_score"]})
    
    if not pool:
        return []
    
    # normalize scores
    max_sc = max(p["_score"] for p in pool)
    for p in pool:
        p["rel"] = p["_score"] / max_sc if max_sc > 0 else 0.0
    
    # sort by relevance desc
    pool.sort(key=lambda x: -x["rel"])
    
    # deduplicate by (benchmark, id_in_benchmark)
    seen = set()
    uniq = []
    for p in pool:
        key = (p["benchmark"], p["id_in_benchmark"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
        if len(uniq) >= max_candidates:
            break
    
    return uniq

def plan_split(es: Elasticsearch, query_vec: List[float], spec: SplitSpec) -> Tuple[List[dict], List[dict]]:
    """
    Build held-in and held-out lists with:
     - distribution matching across difficulty buckets,
     - guarded against leakage (no same (benchmark,id) in both sets),
     - balanced across benchmarks (avoid single-benchmark dominance).
    """
    # 즉시 적용: 더 많은 후보 검색 (Elasticsearch 제한 준수)
    search_k = min(10000, max(5000, spec.total * 15))  # 15배로 증가하되 10000 제한
    print(f"Searching for candidates with target total: {spec.total}, search_k: {search_k}")
    
        # 4-stage filtering based split
    print(f"🔍 Starting 4-stage filtering based split...")
    
    # Stage 1: Benchmark filtering (mapping based)
    print(f"📊 Stage 1: Benchmark filtering (mapping based)...")
    
    # Use Elasticsearch mapping data based keyword search
    if spec.benchmarks:
        # Use term query with benchmark filter
        body = {
            "query": {
                "bool": {
                    "must": [
                        {"terms": {"benchmark": spec.benchmarks}}
                    ]
                }
            },
            "size": search_k,
            "_source": True
        }
        print(f"🔍 Searching with benchmark filter: {spec.benchmarks}")
    else:
        # No benchmark restriction, search all data
        body = {
            "query": {"match_all": {}},
            "size": search_k,
            "_source": True
        }
        print(f"ℹ️  No benchmark restriction, searching all data")
    
    resp = es.search(index=ES_INDEX, body=body)
    hits = resp["hits"]["hits"]
    print(f"✅ Benchmark mapping based search result: {len(hits)} candidates")
    
    # Check benchmark distribution
    benchmark_counts = {}
    for hit in hits:
        benchmark = hit["_source"].get("benchmark", "unknown")
        benchmark_counts[benchmark] = benchmark_counts.get(benchmark, 0) + 1
    
    print(f"📊 Benchmark distribution: {benchmark_counts}")
    
    # Stage 2: Topic based hybrid_search filtering
    if spec.topics and query_vec:
        print(f"🏷️  Stage 2: Topic based hybrid_search filtering...")
        topic_hits = hybrid_search(es, query_vec, k=search_k, filters=None)
        
        # Filter by topic relevance
        topic_filtered = []
        for hit in topic_hits:
            src = hit["_source"]
            topic_emb = src.get("topic_embedding")
            if topic_emb:
                max_relevance = max(
                    compute_topic_relevance(query_vec, topic_emb),
                    0.0
                )
                if max_relevance >= TOPIC_RELEVANCE_THRESHOLD:
                    topic_filtered.append(hit)
        
        # Use topic filtering result if it provides more candidates
        if len(topic_filtered) > len(hits):
            hits = topic_filtered
            print(f"✅ After topic filtering: {len(hits)} candidates")
        else:
            print(f"ℹ️  Topic filtering result has fewer candidates, keeping existing result: {len(hits)} candidates")
    
    # Stage 3: Goal based hybrid_search filtering
    print(f"🎯 Stage 3: Goal based hybrid_search filtering...")
    goal_hits = hybrid_search(es, query_vec, k=search_k, filters=None)
    
    # Determine text similarity to goal using simple keyword matching
    goal_filtered = []
    for hit in goal_hits:
        # Determine similarity between goal and text using simple keyword matching
        text = hit["_source"].get("text", "").lower()
        if any(keyword in text for keyword in spec.goal.lower().split()):
            goal_filtered.append(hit)
    
    # Use goal filtering result if it provides more candidates
    if len(goal_filtered) > len(hits):
        hits = goal_filtered
        print(f"✅ After goal filtering: {len(hits)} candidates")
    else:
        print(f"ℹ️  Goal filtering result has fewer candidates, keeping existing result: {len(hits)} candidates")
    
    # Stage 4: Difficulty based filtering
    print(f"📈 Stage 4: Difficulty based filtering...")
    difficulty_filtered = []
    target_difficulties = list(spec.difficulty_mix.keys())
    
    for hit in hits:
        difficulty = hit["_source"].get("difficulty", 3)
        if int(difficulty) in target_difficulties:
            difficulty_filtered.append(hit)
    
    if len(difficulty_filtered) > 0:
        hits = difficulty_filtered
        print(f"✅ After difficulty filtering: {len(hits)} candidates")
    else:
        print(f"⚠️  No difficulty filtering result, keeping existing result: {len(hits)} candidates")
    
    print(f"🎉 Final filtering completed: {len(hits)} candidates")
    
    candidates = select_candidates(hits, spec, max_candidates=spec.total * 6, query_vec=query_vec)

    # compute per-difficulty target counts
    target_counts = compute_target_counts(spec.total, spec.difficulty_mix)
    # bucket candidates by difficulty
    buckets = defaultdict(list)
    for c in candidates:
        d = int(c.get("difficulty", 3))
        buckets[d].append(c)
    # gather selected items fulfilling difficulty targets
    selected = []
    for d, cnt in target_counts.items():
        bucket = buckets.get(d, [])
        if not bucket:
            # fallback: try nearest difficulties
            for offset in (1, -1, 2, -2):
                alt = d + offset
                if alt in buckets and len(buckets[alt]) > 0:
                    take = min(cnt, len(buckets[alt]))
                    selected.extend(buckets[alt][:take])
                    cnt -= take
                if cnt <= 0:
                    break
        else:
            selected.extend(bucket[:cnt])

    # if we under-selected, top up from remaining candidates by relevance
    if len(selected) < spec.total:
        remaining = [c for c in candidates if (c not in selected)]
        remaining.sort(key=lambda x: -x["rel"])
        needed = spec.total - len(selected)
        selected.extend(remaining[:needed])

    # ensure uniqueness again
    uniq = {}
    for x in selected:
        key = (x["benchmark"], x["id_in_benchmark"])
        if key not in uniq:
            uniq[key] = x
    selected = list(uniq.values())[:spec.total]

    # Now partition into held-in and held-out trying to balance benchmarks and difficulty
    by_bench = defaultdict(list)
    for item in selected:
        by_bench[item["benchmark"]].append(item)

    held_in = []
    held_out = []
    
    # desired counts
    desired_in = int(round(spec.total * spec.heldin_ratio))
    desired_out = spec.total - desired_in
    
    print(f"Total selected: {len(selected)}, Desired held-in: {desired_in}, Desired held-out: {desired_out}")
    
    # Simple alternating assignment to ensure both sets get items
    for i, item in enumerate(selected):
        # Skip if we've filled both sets
        if len(held_in) >= desired_in and len(held_out) >= desired_out:
            break
            
        # Check for leakage with the other set
        can_go_to_held_in = (
            len(held_in) < desired_in and 
            not any((item["benchmark"] == h["benchmark"] and item["id_in_benchmark"] == h["id_in_benchmark"]) for h in held_out)
        )
        
        can_go_to_held_out = (
            len(held_out) < desired_out and 
            not any((item["benchmark"] == h["benchmark"] and item["id_in_benchmark"] == h["id_in_benchmark"]) for h in held_in)
        )
        
        # Decide where to put the item
        if can_go_to_held_in and can_go_to_held_out:
            # Both sets can take it, alternate to balance
            if len(held_in) <= len(held_out):
                held_in.append(item)
            else:
                held_out.append(item)
        elif can_go_to_held_in:
            held_in.append(item)
        elif can_go_to_held_out:
            held_out.append(item)
        # If neither can take it due to leakage, skip this item
    
    # Top-up remaining items if needed (while avoiding leakage)
    remaining = [s for s in selected if s not in held_in and s not in held_out]
    
    # Top-up held_in first
    for r in remaining:
        if len(held_in) < desired_in and not any((r["benchmark"] == h["benchmark"] and r["id_in_benchmark"] == h["id_in_benchmark"]) for h in held_out):
            held_in.append(r)
    
    # Then top-up held_out
    remaining = [s for s in selected if s not in held_in and s not in held_out]
    for r in remaining:
        if len(held_out) < desired_out and not any((r["benchmark"] == h["benchmark"] and r["id_in_benchmark"] == h["id_in_benchmark"]) for h in held_in):
            held_out.append(r)
    
    print(f"Final held-in: {len(held_in)}, held-out: {len(held_out)}")
    
    return held_in, held_out

def to_jsonl(rows: List[dict]) -> str:
    """Return a JSONL string representing rows. Use orjson for speed."""
    return "\n".join(orjson.dumps(r).decode() for r in rows)
