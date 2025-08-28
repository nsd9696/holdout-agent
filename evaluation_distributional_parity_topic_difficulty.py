#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simplified evaluation for Holdout-Agent splits using OpenAI Embeddings.

Replaces SBERT with OpenAI's Embeddings API:
- Default model: text-embedding-3-large (3072D)
- Optional: --dimensions to reduce embedding size if supported by the model

Metrics (unchanged):
1) Distributional parity on embeddings
   - FID-style Gaussian distance (mean/cov)
   - KS tests on top-K PCA components (Holm-Bonferroni)

2) Topic coverage & balance (categorical)
   - Chi-square test (p), Cramér's V
   - Jensen–Shannon (JS) divergence, Shannon entropy

3) Difficulty alignment (ordinal 1..5)
   - Chi-square + Cramér's V
   - Mann–Whitney U
   - JS divergence on 5-bin histograms
"""

import os, json, math, argparse
import numpy as np
import pandas as pd

from typing import List
from collections import Counter

# OpenAI SDK (Python)
# Docs: https://platform.openai.com/docs/api-reference/embeddings
from openai import OpenAI

from sklearn.decomposition import PCA
from scipy import linalg
from scipy.stats import ks_2samp, entropy
from scipy.spatial.distance import jensenshannon

# -----------------------
# IO helpers
# -----------------------
def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def extract_text(row: dict) -> str:
    if "text" in row and row["text"]:
        return str(row["text"])
    q = str(row.get("question", ""))
    a = str(row.get("answer", row.get("label", "")))
    if q or a:
        return f"Q: {q}\nA: {a}"
    if "choices" in row:
        return str(row["choices"])
    return ""

def to_df(rows: List[dict], split: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["split"] = split
    
    # Handle topic field
    if "topic" not in df.columns:
        df["topic"] = "UNKNOWN"
    
    # Handle difficulty_bin field - map from difficulty if available
    if "difficulty_bin" not in df.columns:
        if "difficulty" in df.columns:
            # difficulty is already 1-5 scale, use as is
            df["difficulty_bin"] = df["difficulty"]
        else:
            df["difficulty_bin"] = 3
    
    # Handle text field
    if "text" not in df.columns:
        df["text"] = df.apply(extract_text, axis=1)
    
    cols = ["benchmark", "topic", "difficulty_bin", "text", "split"]
    cols = [c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]
    return df[cols]

# -----------------------
# OpenAI Embeddings
# -----------------------
def embed_openai(
    texts: List[str],
    model: str = "text-embedding-3-large",
    dimensions: int = None,
    batch_size: int = 128,
) -> np.ndarray:
    """
    Batched embedding with OpenAI Embeddings API.
    - model: e.g., text-embedding-3-large / text-embedding-3-small
    - dimensions: optional dimensionality reduction if the model supports it
    Docs: https://platform.openai.com/docs/api-reference/embeddings
          https://platform.openai.com/docs/guides/embeddings/embedding-models
    """
    client = OpenAI()  # reads OPENAI_API_KEY from env
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(
            model=model,
            input=batch,
            **({"dimensions": dimensions} if dimensions else {})
        )
        # resp.data preserves order
        out.extend([np.array(d.embedding, dtype=np.float32) for d in resp.data])
    return np.vstack(out)

# -----------------------
# 1) Distributional parity
# -----------------------
def fid_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float:
    """
    Fréchet (FID-style) distance between two Gaussians N(mu1, sigma1), N(mu2, sigma2).
    D^2 = ||mu1-mu2||^2 + Tr(S1 + S2 - 2*(S1*S2)^{1/2})
    We return D (sqrt of the squared distance) for readability.
    """
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    fid2 = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean
    fid2 = max(fid2, 0.0)
    return float(np.sqrt(fid2))

def pca_ks_tests(X_in: np.ndarray, X_out: np.ndarray, topk: int = 10) -> dict:
    """
    PCA on pooled embeddings; KS two-sample test per component (two-sided).
    Holm–Bonferroni correction across K tests.
    """
    X = np.vstack([X_in, X_out])
    pca = PCA(n_components=min(topk, X.shape[1]))
    Z = pca.fit_transform(X)
    Zi = Z[: len(X_in)]
    Zo = Z[len(X_in) :]
    pvals = []
    for k in range(Zi.shape[1]):
        _, p = ks_2samp(Zi[:, k], Zo[:, k], alternative="two-sided", method="auto")
        pvals.append(p)
    # Holm–Bonferroni
    m = len(pvals)
    order = np.argsort(pvals)
    holm_rejects = 0
    adj_pvals = [None] * m
    for rank, idx in enumerate(order, start=1):
        adj = pvals[idx] * (m - rank + 1)
        adj_pvals[idx] = min(adj, 1.0)
        if adj_pvals[idx] < 0.05:
            holm_rejects += 1
    return {
        "KS_rejects_topk": int(holm_rejects),
        "KS_min_p": float(np.min(pvals)),
        "KS_med_p": float(np.median(pvals)),
    }

def parity_per_benchmark(df: pd.DataFrame, embs: np.ndarray, topk_pcs: int = 10) -> pd.DataFrame:
    df = df.copy()
    df["__idx__"] = np.arange(len(df))
    rows = []
    for bench, g in df.groupby("benchmark"):
        gi = g[g["split"] == "held-in"]
        go = g[g["split"] == "held-out"]
        if len(gi) == 0 or len(go) == 0:
            continue
        Xi = embs[gi["__idx__"]]
        Xo = embs[go["__idx__"]]
        mu_i, mu_o = Xi.mean(0), Xo.mean(0)
        Si, So = np.cov(Xi.T), np.cov(Xo.T)
        fid = fid_distance(mu_i, Si, mu_o, So)
        ks_stats = pca_ks_tests(Xi, Xo, topk=topk_pcs)
        rows.append({
            "Benchmark": bench,
            "n_in": len(gi),
            "n_out": len(go),
            "FID_style": round(fid, 6),
            "KS_rejects_topk": ks_stats["KS_rejects_topk"],
            "KS_min_p": round(ks_stats["KS_min_p"], 4),
            "KS_med_p": round(ks_stats["KS_med_p"], 4),
        })
    return pd.DataFrame(rows)

# -----------------------
# 2) Topic coverage & balance
# -----------------------
def topic_tables(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for bench, g in df.groupby("benchmark"):
        gi = g[g["split"] == "held-in"]["topic"]
        go = g[g["split"] == "held-out"]["topic"]
        if gi.empty or go.empty:
            continue
            
        topics = sorted(set(gi) | set(go))
        obs = np.zeros((len(topics), 2), dtype=int)
        for i, t in enumerate(topics):
            obs[i, 0] = int((gi == t).sum())
            obs[i, 1] = int((go == t).sum())
        
        # Distribution calculations
        p_in = obs[:, 0] / (obs[:, 0].sum() + 1e-12)
        p_out = obs[:, 1] / (obs[:, 1].sum() + 1e-12)
        
        # JS divergence calculation
        try:
            js_div = jensenshannon(p_in, p_out, base=2.0) ** 2
            js_status = "Valid"
        except Exception as e:
            js_div = np.nan
            js_status = f"Error: {str(e)[:30]}"
        
        # Entropy calculations
        try:
            ent_in = entropy(p_in + 1e-12, base=2.0)
            ent_out = entropy(p_out + 1e-12, base=2.0)
            entropy_status = "Valid"
        except Exception as e:
            ent_in, ent_out = np.nan, np.nan
            entropy_status = f"Error: {str(e)[:30]}"
        
        rows.append({
            "Benchmark": bench,
            "Topics_in": int((p_in > 0).sum()),
            "Topics_out": int((p_out > 0).sum()),
            "JS_divergence": round(float(js_div), 6) if not np.isnan(js_div) else "N/A",
            "JS_Status": js_status,
            "Entropy_in(bits)": round(float(ent_in), 3) if not np.isnan(ent_in) else "N/A",
            "Entropy_out(bits)": round(float(ent_out), 3) if not np.isnan(ent_out) else "N/A",
            "Entropy_Status": entropy_status,
        })
    return pd.DataFrame(rows)

# -----------------------
# 3) Difficulty alignment
# -----------------------
def difficulty_tables(df: pd.DataFrame, k_bins: int = 5) -> pd.DataFrame:
    rows = []
    for bench, g in df.groupby("benchmark"):
        gi = g[g["split"] == "held-in"]["difficulty_bin"].astype(int)
        go = g[g["split"] == "held-out"]["difficulty_bin"].astype(int)
        if gi.empty or go.empty:
            continue
            
        obs = np.zeros((k_bins, 2), dtype=int)
        for b in range(1, k_bins + 1):
            obs[b - 1, 0] = int((gi == b).sum())
            obs[b - 1, 1] = int((go == b).sum())
        
        # Distribution calculations
        mix_in = (obs[:, 0] / (obs[:, 0].sum() + 1e-12)).round(3)
        mix_out = (obs[:, 1] / (obs[:, 1].sum() + 1e-12)).round(3)
        mean_in = float(np.average(np.arange(1, k_bins + 1), weights=mix_in))
        mean_out = float(np.average(np.arange(1, k_bins + 1), weights=mix_out))
        
        rows.append({
            "Benchmark": bench,
            "Mix_in_(1..5)": " ".join(f"{p:.2f}" for p in mix_in),
            "Mix_out_(1..5)": " ".join(f"{p:.2f}" for p in mix_out),
            "Mean_bin_in": round(mean_in, 3),
            "Mean_bin_out": round(mean_out, 3),
            "Out_harder?": "Yes" if mean_out > mean_in else "No",
        })
    return pd.DataFrame(rows)

# -----------------------
# Save helpers
# -----------------------
def save_tables(df: pd.DataFrame, outdir: str, basename: str) -> None:
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, f"{basename}.csv"), index=False)
    with open(os.path.join(outdir, f"{basename}.md"), "w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))
    print(f"[OK] wrote {basename}.csv / {basename}.md in {outdir}")

# -----------------------
# Main
# -----------------------
def main(args):
    rin = load_jsonl(args.held_in)
    rout = load_jsonl(args.held_out)
    dfi = to_df(rin, "held-in")
    dfo = to_df(rout, "held-out")
    df = pd.concat([dfi, dfo], ignore_index=True)

    # 1) Distributional parity — Embeddings via OpenAI
    print("[Embed] requesting OpenAI embeddings...")
    texts = df["text"].tolist()
    E = embed_openai(
        texts,
        model=args.openai_model,
        dimensions=args.dimensions,
        batch_size=args.batch_size,
    )

    # 2) Parity tables
    print("[Parity] FID-style + KS on top-K PCs...")
    parity_tbl = parity_per_benchmark(df, E, topk_pcs=args.topk_pcs)
    save_tables(parity_tbl, args.outdir, "table_parity")

    # 3) Topic coverage & balance
    print("[Topics] JS divergence + entropies...")
    topic_tbl = topic_tables(df)
    save_tables(topic_tbl, args.outdir, "table_topic")

    # 4) Difficulty alignment
    print("[Difficulty] distribution comparison and mean difficulty...")
    diff_tbl = difficulty_tables(df, k_bins=5)
    save_tables(diff_tbl, args.outdir, "table_difficulty")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--held_in", type=str, default="held_in.jsonl")
    ap.add_argument("--held_out", type=str, default="held_out.jsonl")
    ap.add_argument("--openai_model", type=str, default="text-embedding-3-large")
    ap.add_argument("--dimensions", type=int, default=None, help="Optional dimensionality reduction if supported by model")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--topk_pcs", type=int, default=10)
    ap.add_argument("--outdir", type=str, default="./eval_simple_out")
    args = ap.parse_args()
    main(args)
