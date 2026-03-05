#!/usr/bin/env python3
"""
Step 4: Reranking search pipeline over the PEX/SAQ corpus.

Implements a two-stage retrieval system:
  Stage 1 — Embedding similarity (cosine) for fast candidate retrieval
  Stage 2 — Isaacus Universal Classifier as reranker for precision

This combines the outputs of Step 1 (embeddings) with the Isaacus
reranking API to provide high-quality semantic search over the corpus.

Also pre-computes a similarity matrix across all documents for
cluster analysis and study-path recommendations.

Order of operations: Run AFTER embed.py (Step 1). Independent of others.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
from isaacus import Isaacus

API_KEY = os.environ.get(
    "ISAACUS_API_KEY",
    "iuak_v1_alcvXayhjV_8O92bz8S8kphmSOBpiNNoDDAQoexh6q1_feeb2b85",
)
OUTPUT_DIR = Path(__file__).parent / "search_index"
ENRICHMENT_DIR = Path(__file__).parent / "enrichment_results"
EMBEDDING_DIR = Path(__file__).parent / "embedding_results"
ENRICHMENT_MANIFEST = ENRICHMENT_DIR / "manifest.json"
EMBEDDING_MANIFEST = EMBEDDING_DIR / "manifest.json"


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def load_all_document_embeddings() -> tuple[list[str], list[list[float]]]:
    """Load all document-level embeddings. Returns (paths, vectors)."""
    with open(EMBEDDING_MANIFEST) as f:
        embed_manifest = json.load(f)

    paths = []
    vectors = []
    for path, info in sorted(embed_manifest.items()):
        with open(EMBEDDING_DIR / info["output_file"]) as f:
            data = json.load(f)
        if data["document_embedding"]:
            paths.append(path)
            vectors.append(data["document_embedding"])

    return paths, vectors


def embedding_search(
    query_embedding: list[float],
    doc_paths: list[str],
    doc_vectors: list[list[float]],
    top_n: int = 20,
) -> list[tuple[str, float]]:
    """Stage 1: Fast cosine similarity search over embeddings."""
    scores = [
        (path, cosine_similarity(query_embedding, vec))
        for path, vec in zip(doc_paths, doc_vectors)
    ]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_n]


def rerank_candidates(
    client: Isaacus,
    query: str,
    candidate_paths: list[str],
    candidate_texts: list[str],
    top_n: int = 10,
) -> list[dict]:
    """Stage 2: Rerank candidates using Isaacus Universal Classifier."""
    response = client.rerankings.create(
        model="kanon-universal-classifier",
        query=query,
        texts=candidate_texts,
        top_n=top_n,
        is_iql=False,
    )
    results = []
    for r in response.results:
        results.append({
            "path": candidate_paths[r.index],
            "rerank_score": r.score,
        })
    return results


def build_similarity_matrix(
    doc_paths: list[str], doc_vectors: list[list[float]]
) -> dict:
    """
    Compute pairwise cosine similarity matrix for cluster analysis.

    Returns a sparse representation (only pairs with similarity > 0.5)
    to keep the output manageable for 382 documents.
    """
    n = len(doc_paths)
    matrix = np.zeros((n, n))
    vecs = np.array(doc_vectors)

    # Normalise all vectors
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalised = vecs / norms

    # Compute full similarity matrix via matrix multiplication
    matrix = normalised @ normalised.T

    # Extract significant pairs (above threshold, excluding self-similarity)
    threshold = 0.5
    significant_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(matrix[i, j])
            if sim > threshold:
                significant_pairs.append({
                    "doc_a": doc_paths[i],
                    "doc_b": doc_paths[j],
                    "similarity": round(sim, 4),
                })

    significant_pairs.sort(key=lambda x: x["similarity"], reverse=True)

    return {
        "num_documents": n,
        "threshold": threshold,
        "num_significant_pairs": len(significant_pairs),
        "pairs": significant_pairs,
    }


def build_topic_clusters(
    doc_paths: list[str], doc_vectors: list[list[float]]
) -> dict:
    """
    Group documents by subcategory and compute inter/intra-cluster
    similarity statistics for study-path recommendations.
    """
    # Group by subcategory
    clusters = {}
    for i, path in enumerate(doc_paths):
        parts = path.split("/")
        subcat = "/".join(parts[:2]) if len(parts) >= 2 else parts[0]
        if subcat not in clusters:
            clusters[subcat] = []
        clusters[subcat].append(i)

    vecs = np.array(doc_vectors)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalised = vecs / norms

    cluster_stats = {}
    for subcat, indices in sorted(clusters.items()):
        if len(indices) < 2:
            cluster_stats[subcat] = {
                "size": len(indices),
                "mean_intra_similarity": None,
            }
            continue

        # Intra-cluster similarity
        cluster_vecs = normalised[indices]
        sim_matrix = cluster_vecs @ cluster_vecs.T
        # Extract upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
        intra_sims = sim_matrix[mask]

        cluster_stats[subcat] = {
            "size": len(indices),
            "mean_intra_similarity": round(float(np.mean(intra_sims)), 4),
            "min_intra_similarity": round(float(np.min(intra_sims)), 4),
            "max_intra_similarity": round(float(np.max(intra_sims)), 4),
        }

    return cluster_stats


def main():
    client = Isaacus(api_key=API_KEY)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading document embeddings...")
    doc_paths, doc_vectors = load_all_document_embeddings()
    print(f"Loaded {len(doc_paths)} document embeddings")

    # --- Build similarity matrix ---
    print("\nComputing pairwise similarity matrix...")
    sim_data = build_similarity_matrix(doc_paths, doc_vectors)
    with open(OUTPUT_DIR / "similarity_matrix.json", "w") as f:
        json.dump(sim_data, f, indent=2, ensure_ascii=False)
    print(f"Found {sim_data['num_significant_pairs']} significant pairs (>{sim_data['threshold']})")

    # --- Build topic cluster stats ---
    print("\nComputing topic cluster statistics...")
    cluster_stats = build_topic_clusters(doc_paths, doc_vectors)
    with open(OUTPUT_DIR / "cluster_stats.json", "w") as f:
        json.dump(cluster_stats, f, indent=2, ensure_ascii=False)
    print(f"Computed stats for {len(cluster_stats)} subcategories")

    # --- Pre-compute nearest neighbours for each document ---
    print("\nComputing nearest neighbours...")
    with open(ENRICHMENT_MANIFEST) as f:
        enrich_manifest = json.load(f)

    neighbours = {}
    vecs = np.array(doc_vectors)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalised = vecs / norms
    sim_matrix = normalised @ normalised.T

    for i, path in enumerate(doc_paths):
        sims = [(doc_paths[j], float(sim_matrix[i, j])) for j in range(len(doc_paths)) if j != i]
        sims.sort(key=lambda x: x[1], reverse=True)
        neighbours[path] = [
            {"path": s[0], "similarity": round(s[1], 4)} for s in sims[:10]
        ]

    with open(OUTPUT_DIR / "nearest_neighbours.json", "w") as f:
        json.dump(neighbours, f, indent=2, ensure_ascii=False)
    print(f"Computed 10 nearest neighbours for {len(neighbours)} documents")

    # --- Save search index metadata ---
    index_meta = {
        "num_documents": len(doc_paths),
        "embedding_dim": len(doc_vectors[0]) if doc_vectors else 0,
        "document_paths": doc_paths,
        "similarity_threshold": 0.5,
        "num_significant_pairs": sim_data["num_significant_pairs"],
    }
    with open(OUTPUT_DIR / "index_meta.json", "w") as f:
        json.dump(index_meta, f, indent=2, ensure_ascii=False)

    print(f"\nSearch index built! Results: {OUTPUT_DIR}")


def search(query: str, top_n: int = 10) -> list[dict]:
    """
    Two-stage semantic search over the SAQ corpus.

    Stage 1: Embed query -> cosine similarity -> top 20 candidates
    Stage 2: Rerank candidates with Universal Classifier -> top N results
    """
    client = Isaacus(api_key=API_KEY)

    # Load embeddings
    doc_paths, doc_vectors = load_all_document_embeddings()

    # Stage 1: Embed query
    response = client.embeddings.create(
        model="kanon-2-embedder",
        texts=[query],
        task="retrieval/query",
    )
    query_vec = response.embeddings[0].embedding

    # Stage 1: Fast retrieval
    candidates = embedding_search(query_vec, doc_paths, doc_vectors, top_n=20)

    # Load candidate texts
    with open(ENRICHMENT_MANIFEST) as f:
        manifest = json.load(f)

    candidate_paths = [c[0] for c in candidates]
    candidate_texts = []
    for path in candidate_paths:
        info = manifest[path]
        with open(ENRICHMENT_DIR / info["output_file"]) as f:
            doc = json.load(f)
        candidate_texts.append(doc["text"][:5000])  # Truncate for reranker

    # Stage 2: Rerank
    results = rerank_candidates(client, query, candidate_paths, candidate_texts, top_n)

    # Merge scores
    embedding_scores = {c[0]: c[1] for c in candidates}
    for r in results:
        r["embedding_score"] = round(embedding_scores.get(r["path"], 0), 4)
        r["combined_score"] = round(
            0.3 * r["embedding_score"] + 0.7 * r["rerank_score"], 4
        )

    results.sort(key=lambda x: x["combined_score"], reverse=True)
    return results


if __name__ == "__main__":
    main()
