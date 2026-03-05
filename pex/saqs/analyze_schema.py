#!/usr/bin/env python3
"""
Post-Step 2 schema analysis: multi-stage graph-aware analytics.

Leverages the full enrichment knowledge graph (segment hierarchy,
cross-references, term networks), embedding vector space, and
classification ontology in concert to refine the schema.

Stages:
   1. Dimension activation analysis
   2. Score distribution calibration
   3. Co-activation matrix (Jaccard)
   4. Threshold sensitivity
   5. Cognitive domain confusion
   6. Category–dimension affinity
   7. Information-theoretic discriminability
   8. Dimension profile clustering
   9. Knowledge graph topology analysis (enrichment hierarchy + crossrefs)
  10. Embedding-space coherence (dimension clusters in vector space)
  11. Term–dimension alignment (enrichment terms vs classification tags)
  12. Actionable schema refinement recommendations (synthesised from all stages)

Usage:
    python analyze_schema.py              # Full analysis
    python analyze_schema.py --preview    # Run on partial data (during Step 2)
"""

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CLASSIFICATION_DIR = Path(__file__).parent / "classification_results"
ENRICHMENT_DIR = Path(__file__).parent / "enrichment_results"
EMBEDDING_DIR = Path(__file__).parent / "embedding_results"
OUTPUT_DIR = Path(__file__).parent / "schema_analysis"


def load_data():
    """Load classification manifest and all individual result files."""
    manifest_path = CLASSIFICATION_DIR / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Load full classification scores from individual files
    docs = {}
    for path, entry in manifest.items():
        result_path = CLASSIFICATION_DIR / entry["output_file"]
        if result_path.exists():
            with open(result_path) as f:
                docs[path] = json.load(f)

    return manifest, docs


def load_enrichment_data(manifest):
    """Load enrichment results for graph topology analysis."""
    enrich_manifest_path = ENRICHMENT_DIR / "manifest.json"
    if not enrich_manifest_path.exists():
        return {}

    with open(enrich_manifest_path) as f:
        enrich_manifest = json.load(f)

    enrichments = {}
    for path in manifest:
        if path in enrich_manifest:
            fpath = ENRICHMENT_DIR / enrich_manifest[path]["output_file"]
            if fpath.exists():
                with open(fpath) as f:
                    enrichments[path] = json.load(f)

    return enrichments


def load_embedding_data(manifest):
    """Load embedding results for vector space analysis."""
    embed_manifest_path = EMBEDDING_DIR / "manifest.json"
    if not embed_manifest_path.exists():
        return {}, {}

    with open(embed_manifest_path) as f:
        embed_manifest = json.load(f)

    doc_embeddings = {}
    seg_embeddings = {}
    for path in manifest:
        if path in embed_manifest:
            fpath = EMBEDDING_DIR / embed_manifest[path]["output_file"]
            if fpath.exists():
                with open(fpath) as f:
                    data = json.load(f)
                if data.get("document_embedding"):
                    doc_embeddings[path] = data["document_embedding"]
                if data.get("segment_embeddings"):
                    seg_embeddings[path] = data["segment_embeddings"]

    return doc_embeddings, seg_embeddings


# =========================================================================
# STAGE 1: Dimension Activation Analysis
# =========================================================================

def analyze_activation(manifest):
    """Compute activation rates, score distributions, and outlier dimensions."""
    all_dims = set()
    dim_activations = defaultdict(int)
    dim_counts_per_doc = []
    n = len(manifest)

    for entry in manifest.values():
        dims = entry.get("content_dimensions", [])
        dim_counts_per_doc.append(len(dims))
        for d in dims:
            dim_activations[d] += 1
            all_dims.add(d)

    counts = np.array(dim_counts_per_doc)
    activation_rates = {d: dim_activations[d] / n for d in sorted(all_dims)}

    return {
        "total_documents": n,
        "dimensions_per_doc": {
            "mean": float(np.mean(counts)),
            "median": float(np.median(counts)),
            "std": float(np.std(counts)),
            "min": int(np.min(counts)),
            "max": int(np.max(counts)),
            "p25": float(np.percentile(counts, 25)),
            "p75": float(np.percentile(counts, 75)),
        },
        "activation_rates": activation_rates,
        "high_activation": {d: r for d, r in activation_rates.items() if r > 0.7},
        "low_activation": {d: r for d, r in activation_rates.items() if r < 0.1},
    }


# =========================================================================
# STAGE 2: Score Distribution Analysis (per dimension)
# =========================================================================

def analyze_score_distributions(docs):
    """Analyze raw score distributions per dimension for calibration issues."""
    dim_scores = defaultdict(list)

    for doc in docs.values():
        classifications = doc.get("classifications", {})
        content_dims = classifications.get("content_dimensions", {})
        for dim, info in content_dims.items():
            score = info.get("score")
            if score is not None:
                dim_scores[dim].append(score)

    results = {}
    for dim, scores in sorted(dim_scores.items()):
        arr = np.array(scores)
        # Bimodality: compute Hartigan's dip statistic approximation
        # Use coefficient of bimodality: (skew^2 + 1) / kurtosis
        # Values > 0.555 suggest bimodality
        skew = float(np.mean(((arr - arr.mean()) / max(arr.std(), 1e-9)) ** 3))
        kurt = float(np.mean(((arr - arr.mean()) / max(arr.std(), 1e-9)) ** 4))
        bimodality_coeff = (skew ** 2 + 1) / max(kurt, 1e-9)

        # Score buckets for histogram
        buckets = {}
        for low, high, label in [
            (0.0, 0.3, "low_0_to_0.3"),
            (0.3, 0.5, "mid_0.3_to_0.5"),
            (0.5, 0.6, "marginal_0.5_to_0.6"),
            (0.6, 0.8, "moderate_0.6_to_0.8"),
            (0.8, 1.01, "high_0.8_to_1.0"),
        ]:
            buckets[label] = int(np.sum((arr >= low) & (arr < high)))

        results[dim] = {
            "count": len(scores),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
            "skewness": round(skew, 3),
            "bimodality_coefficient": round(bimodality_coeff, 3),
            "is_bimodal": bimodality_coeff > 0.555,
            "score_buckets": buckets,
        }

    return results


# =========================================================================
# STAGE 3: Co-Activation Matrix (Jaccard similarity)
# =========================================================================

def analyze_coactivation(manifest):
    """Compute pairwise co-activation rates using Jaccard similarity."""
    all_dims = set()
    doc_dim_sets = []

    for entry in manifest.values():
        dims = frozenset(entry.get("content_dimensions", []))
        doc_dim_sets.append(dims)
        all_dims |= dims

    dims_sorted = sorted(all_dims)
    n_dims = len(dims_sorted)
    dim_idx = {d: i for i, d in enumerate(dims_sorted)}

    # Compute Jaccard matrix
    jaccard = np.zeros((n_dims, n_dims))
    co_count = np.zeros((n_dims, n_dims), dtype=int)

    for dims in doc_dim_sets:
        indices = [dim_idx[d] for d in dims]
        for i in indices:
            for j in indices:
                co_count[i, j] += 1

    for i in range(n_dims):
        for j in range(n_dims):
            union = co_count[i, i] + co_count[j, j] - co_count[i, j]
            jaccard[i, j] = co_count[i, j] / max(union, 1)

    # Find redundant pairs (Jaccard > 0.6)
    redundant = []
    for i in range(n_dims):
        for j in range(i + 1, n_dims):
            if jaccard[i, j] > 0.6:
                redundant.append({
                    "dim_a": dims_sorted[i],
                    "dim_b": dims_sorted[j],
                    "jaccard": round(float(jaccard[i, j]), 3),
                    "co_occurrences": int(co_count[i, j]),
                    "a_total": int(co_count[i, i]),
                    "b_total": int(co_count[j, j]),
                })

    redundant.sort(key=lambda x: x["jaccard"], reverse=True)

    return {
        "dimensions": dims_sorted,
        "jaccard_matrix": {
            dims_sorted[i]: {
                dims_sorted[j]: round(float(jaccard[i, j]), 3)
                for j in range(n_dims)
            }
            for i in range(n_dims)
        },
        "redundant_pairs": redundant,
    }


# =========================================================================
# STAGE 4: Threshold Sensitivity Analysis
# =========================================================================

def analyze_threshold_sensitivity(docs):
    """Test multiple thresholds to find optimal activation point per dimension."""
    dim_scores = defaultdict(list)

    for doc in docs.values():
        classifications = doc.get("classifications", {})
        content_dims = classifications.get("content_dimensions", {})
        for dim, info in content_dims.items():
            score = info.get("score")
            if score is not None:
                dim_scores[dim].append(score)

    thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
    results = {}

    for dim, scores in sorted(dim_scores.items()):
        arr = np.array(scores)
        n = len(arr)

        # For each threshold, compute activation rate
        threshold_data = {}
        for t in thresholds:
            active = int(np.sum(arr > t))
            rate = active / n
            threshold_data[str(t)] = {
                "active_count": active,
                "rate": round(rate, 3),
            }

        # Find the threshold that maximizes separation (largest gap in density)
        # Use the threshold where the activation rate drop is steepest
        rates = [threshold_data[str(t)]["rate"] for t in thresholds]
        drops = [rates[i] - rates[i + 1] for i in range(len(rates) - 1)]
        max_drop_idx = int(np.argmax(drops))
        optimal_threshold = (thresholds[max_drop_idx] + thresholds[max_drop_idx + 1]) / 2

        results[dim] = {
            "thresholds": threshold_data,
            "optimal_threshold": round(optimal_threshold, 3),
            "max_drop_between": [thresholds[max_drop_idx], thresholds[max_drop_idx + 1]],
        }

    return results


# =========================================================================
# STAGE 5: Cognitive Domain Confusion Analysis
# =========================================================================

def analyze_cognitive_confusion(docs):
    """Identify documents with ambiguous cognitive domain classification."""
    confused = []

    for path, doc in docs.items():
        classifications = doc.get("classifications", {})
        cog = classifications.get("cognitive_domain", {})

        scores = {
            label: info.get("score", 0) or 0
            for label, info in cog.items()
        }
        if not scores:
            continue

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_score = ranked[0][1]
        top_label = ranked[0][0]

        if len(ranked) > 1:
            second_score = ranked[1][1]
            second_label = ranked[1][0]
            gap = top_score - second_score

            if gap < 0.1 and top_score > 0.3:
                confused.append({
                    "source": path,
                    "primary": top_label,
                    "primary_score": round(top_score, 3),
                    "secondary": second_label,
                    "secondary_score": round(second_score, 3),
                    "gap": round(gap, 3),
                })

    confused.sort(key=lambda x: x["gap"])

    # Domain distribution
    domain_counts = Counter()
    for doc in docs.values():
        summary = doc.get("summary", {})
        cog = summary.get("cognitive_domain", {})
        primary = cog.get("primary")
        if primary:
            domain_counts[primary] += 1

    return {
        "domain_distribution": dict(domain_counts),
        "ambiguous_documents": confused[:30],
        "total_ambiguous": len(confused),
        "ambiguity_rate": round(len(confused) / max(len(docs), 1), 3),
    }


# =========================================================================
# STAGE 6: Category–Dimension Affinity
# =========================================================================

def analyze_category_affinity(manifest):
    """Map source categories to their dimension profiles — expected vs actual."""
    # Extract top-level category from path (e.g., "pharmacology/analgesics" → "pharmacology")
    # and subcategory
    category_dims = defaultdict(lambda: defaultdict(int))
    category_counts = Counter()
    subcat_dims = defaultdict(lambda: defaultdict(int))
    subcat_counts = Counter()

    for path, entry in manifest.items():
        parts = path.split("/")
        category = parts[0] if parts else "unknown"
        subcategory = "/".join(parts[:2]) if len(parts) >= 2 else category

        dims = entry.get("content_dimensions", [])
        category_counts[category] += 1
        subcat_counts[subcategory] += 1

        for d in dims:
            category_dims[category][d] += 1
            subcat_dims[subcategory][d] += 1

    # Compute affinity (dimension rate per category)
    affinity = {}
    for cat in sorted(category_dims):
        n = category_counts[cat]
        affinity[cat] = {
            "count": n,
            "dimension_rates": {
                d: round(c / n, 3)
                for d, c in sorted(
                    category_dims[cat].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            },
        }

    # Subcategory-level (more granular)
    subcat_affinity = {}
    for subcat in sorted(subcat_dims):
        n = subcat_counts[subcat]
        if n < 3:
            continue
        subcat_affinity[subcat] = {
            "count": n,
            "top_dimensions": {
                d: round(c / n, 3)
                for d, c in sorted(
                    subcat_dims[subcat].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:5]
            },
        }

    # Surprising affinities: dimensions that activate in unexpected categories
    # (e.g., haemostasis in equipment docs)
    surprises = []
    expected_map = {
        "other": {"equipment_and_physics"},
        "pharmacology": {
            "drug_receptor_disposition", "pharmacokinetic_process",
            "organ_system_process", "nociception_process",
            "neuromuscular_process",
        },
        "physiology": {
            "organ_system_process", "gas_exchange_process",
            "acid_base_quality", "autonomic_disposition",
        },
    }

    for cat, dim_counts in category_dims.items():
        n = category_counts[cat]
        expected = expected_map.get(cat, set())
        for dim, count in dim_counts.items():
            rate = count / n
            if dim not in expected and rate > 0.4:
                surprises.append({
                    "category": cat,
                    "dimension": dim,
                    "rate": round(rate, 3),
                    "note": f"{dim} activates in {rate:.0%} of {cat} docs but is not expected",
                })

    surprises.sort(key=lambda x: x["rate"], reverse=True)

    return {
        "category_affinity": affinity,
        "subcategory_affinity": subcat_affinity,
        "surprising_affinities": surprises,
    }


# =========================================================================
# STAGE 7: Information-Theoretic Analysis
# =========================================================================

def analyze_information_theory(manifest):
    """Compute entropy and mutual information for each dimension."""
    all_dims = set()
    for entry in manifest.values():
        all_dims |= set(entry.get("content_dimensions", []))

    n = len(manifest)
    dim_presence = {}
    for d in all_dims:
        count = sum(
            1 for e in manifest.values()
            if d in e.get("content_dimensions", [])
        )
        dim_presence[d] = count

    # Shannon entropy per dimension (binary: active or not)
    # H = -p*log2(p) - (1-p)*log2(1-p)
    entropies = {}
    for d in sorted(all_dims):
        p = dim_presence[d] / n
        if p == 0 or p == 1:
            h = 0.0
        else:
            h = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
        entropies[d] = round(h, 4)

    # Joint entropy and mutual information between dimension pairs
    # MI(A,B) = H(A) + H(B) - H(A,B)
    mi_pairs = []
    dims_list = sorted(all_dims)
    doc_dim_sets = [
        frozenset(e.get("content_dimensions", []))
        for e in manifest.values()
    ]

    for i, da in enumerate(dims_list):
        for db in dims_list[i + 1:]:
            both = sum(1 for s in doc_dim_sets if da in s and db in s)
            a_only = dim_presence[da] - both
            b_only = dim_presence[db] - both
            neither = n - dim_presence[da] - dim_presence[db] + both

            # Joint entropy
            probs = [both / n, a_only / n, b_only / n, neither / n]
            h_joint = sum(-p * math.log2(p) for p in probs if p > 0)

            mi = entropies[da] + entropies[db] - h_joint
            # Normalized MI (0-1)
            nmi = mi / max(min(entropies[da], entropies[db]), 1e-9)

            if nmi > 0.15:
                mi_pairs.append({
                    "dim_a": da,
                    "dim_b": db,
                    "mutual_information": round(mi, 4),
                    "normalized_mi": round(nmi, 4),
                })

    mi_pairs.sort(key=lambda x: x["normalized_mi"], reverse=True)

    # Discriminability ranking: dimensions with entropy closest to 1.0
    # are the most discriminative (split the corpus most evenly)
    ranked = sorted(entropies.items(), key=lambda x: x[1], reverse=True)

    return {
        "dimension_entropy": entropies,
        "discriminability_ranking": [
            {"dimension": d, "entropy": h, "activation_rate": round(dim_presence[d] / n, 3)}
            for d, h in ranked
        ],
        "high_mutual_information_pairs": mi_pairs[:20],
    }


# =========================================================================
# STAGE 8: Dimension Profile Clustering
# =========================================================================

def analyze_clusters(manifest):
    """Cluster documents by their dimension activation profiles."""
    all_dims = set()
    for entry in manifest.values():
        all_dims |= set(entry.get("content_dimensions", []))

    dims_sorted = sorted(all_dims)
    dim_idx = {d: i for i, d in enumerate(dims_sorted)}
    n_dims = len(dims_sorted)

    # Build binary feature matrix
    paths = list(manifest.keys())
    X = np.zeros((len(paths), n_dims))
    for i, path in enumerate(paths):
        for d in manifest[path].get("content_dimensions", []):
            if d in dim_idx:
                X[i, dim_idx[d]] = 1

    # K-means clustering (simple, no sklearn dependency)
    # Use 5 clusters as a reasonable default for this corpus size
    k = 5
    np.random.seed(42)
    centroids = X[np.random.choice(len(X), k, replace=False)]

    for _ in range(50):
        # Assign clusters
        dists = np.array([
            np.sum((X - c) ** 2, axis=1)
            for c in centroids
        ]).T
        labels = np.argmin(dists, axis=1)

        # Update centroids
        new_centroids = np.array([
            X[labels == c].mean(axis=0) if np.sum(labels == c) > 0 else centroids[c]
            for c in range(k)
        ])

        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    # Characterize clusters
    clusters = {}
    for c in range(k):
        mask = labels == c
        cluster_docs = [paths[i] for i in range(len(paths)) if mask[i]]
        if not cluster_docs:
            continue

        cluster_profile = X[mask].mean(axis=0)
        top_dims = sorted(
            [(dims_sorted[j], float(cluster_profile[j])) for j in range(n_dims)],
            key=lambda x: x[1],
            reverse=True,
        )

        # Most common source categories in this cluster
        cat_counter = Counter()
        for p in cluster_docs:
            parts = p.split("/")
            cat_counter["/".join(parts[:2])] += 1

        clusters[f"cluster_{c}"] = {
            "size": int(mask.sum()),
            "top_dimensions": [
                {"dimension": d, "rate": round(r, 3)}
                for d, r in top_dims[:5] if r > 0.2
            ],
            "top_categories": [
                {"category": cat, "count": cnt}
                for cat, cnt in cat_counter.most_common(5)
            ],
            "sample_docs": cluster_docs[:5],
        }

    return {
        "num_clusters": k,
        "clusters": clusters,
    }


# =========================================================================
# STAGE 9: Knowledge Graph Topology Analysis
# =========================================================================

def analyze_graph_topology(enrichments, manifest):
    """Analyze the enrichment knowledge graph structure in relation to classifications.

    Uses segment hierarchy, cross-references, and term networks to assess
    whether classification dimensions align with the structural topology.
    """
    # Per-document graph metrics
    doc_metrics = {}
    total_crossrefs = 0
    total_terms = 0
    total_segments = 0
    depth_by_dim = defaultdict(list)  # dimension -> list of avg tree depths

    for path, enrich in enrichments.items():
        segments = enrich.get("segments", [])
        crossrefs = enrich.get("crossreferences", [])
        terms = enrich.get("terms", [])

        total_crossrefs += len(crossrefs)
        total_terms += len(terms)
        total_segments += len(segments)

        # Compute tree depth and branching factor
        levels = [s["level"] for s in segments if s.get("level") is not None]
        max_depth = max(levels) if levels else 0
        avg_depth = sum(levels) / len(levels) if levels else 0

        # Branching factor: avg children per container node
        containers = [s for s in segments if s["kind"] == "container"]
        avg_branching = (
            sum(len(s.get("children", [])) for s in containers) / len(containers)
            if containers else 0
        )

        # Term density: terms per 1000 chars
        text_len = len(enrich.get("text", ""))
        term_density = len(terms) / max(text_len / 1000, 1)

        doc_metrics[path] = {
            "num_segments": len(segments),
            "max_depth": max_depth,
            "avg_depth": round(avg_depth, 2),
            "avg_branching": round(avg_branching, 2),
            "num_crossrefs": len(crossrefs),
            "num_terms": len(terms),
            "term_density": round(term_density, 2),
        }

        # Map document dimensions to structural metrics
        entry = manifest.get(path, {})
        dims = entry.get("content_dimensions", [])
        for dim in dims:
            depth_by_dim[dim].append(avg_depth)

    # Dimension–complexity correlation: do some dimensions appear in
    # structurally more complex documents?
    dim_complexity = {}
    for dim, depths in depth_by_dim.items():
        arr = np.array(depths)
        dim_complexity[dim] = {
            "mean_tree_depth": round(float(arr.mean()), 2),
            "doc_count": len(depths),
        }

    # Crossref network density
    docs_with_crossrefs = sum(1 for m in doc_metrics.values() if m["num_crossrefs"] > 0)
    docs_with_terms = sum(1 for m in doc_metrics.values() if m["num_terms"] > 0)

    # Identify documents with rich graph structure but few dimensions (under-tagged)
    under_tagged = []
    for path, metrics in doc_metrics.items():
        entry = manifest.get(path, {})
        n_dims = entry.get("num_active_dimensions", 0)
        complexity = metrics["num_segments"] + metrics["num_crossrefs"] * 3 + metrics["num_terms"] * 2
        if complexity > 50 and n_dims <= 2:
            under_tagged.append({
                "source": path,
                "complexity_score": complexity,
                "num_dimensions": n_dims,
                "num_segments": metrics["num_segments"],
                "num_crossrefs": metrics["num_crossrefs"],
                "num_terms": metrics["num_terms"],
            })

    under_tagged.sort(key=lambda x: x["complexity_score"], reverse=True)

    return {
        "corpus_totals": {
            "total_segments": total_segments,
            "total_crossrefs": total_crossrefs,
            "total_terms": total_terms,
            "docs_with_crossrefs": docs_with_crossrefs,
            "docs_with_terms": docs_with_terms,
        },
        "dimension_complexity_correlation": dim_complexity,
        "under_tagged_complex_documents": under_tagged[:15],
    }


# =========================================================================
# STAGE 10: Embedding-Space Coherence
# =========================================================================

def analyze_embedding_coherence(doc_embeddings, manifest):
    """Assess whether classification dimensions form coherent clusters in
    the embedding vector space.

    For each dimension, compute the mean intra-cluster cosine similarity of
    documents tagged with that dimension vs. the global mean similarity.
    High coherence = the dimension captures a real semantic grouping.
    Low coherence = the dimension cuts across unrelated content (noisy).
    """
    if len(doc_embeddings) < 10:
        return {"error": "Not enough embeddings for meaningful analysis"}

    # Build embedding matrix for classified documents
    paths_with_both = [
        p for p in manifest
        if p in doc_embeddings and doc_embeddings[p]
    ]

    if len(paths_with_both) < 10:
        return {"error": "Not enough overlapping documents"}

    # Use a random subsample if corpus is large (cosine matrix is O(n^2))
    np.random.seed(42)
    if len(paths_with_both) > 200:
        sample_idx = np.random.choice(len(paths_with_both), 200, replace=False)
        paths_with_both = [paths_with_both[i] for i in sample_idx]

    vecs = np.array([doc_embeddings[p] for p in paths_with_both])
    # Normalize for cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    vecs_normed = vecs / norms

    # Full cosine similarity matrix
    sim_matrix = vecs_normed @ vecs_normed.T

    # Global mean similarity (upper triangle only)
    n = len(paths_with_both)
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    global_mean_sim = float(sim_matrix[upper_mask].mean())

    # Per-dimension intra-cluster coherence
    all_dims = set()
    for p in paths_with_both:
        all_dims |= set(manifest.get(p, {}).get("content_dimensions", []))

    dim_coherence = {}
    for dim in sorted(all_dims):
        # Indices of documents with this dimension active
        indices = [
            i for i, p in enumerate(paths_with_both)
            if dim in manifest.get(p, {}).get("content_dimensions", [])
        ]

        if len(indices) < 3:
            continue

        # Intra-cluster similarity
        idx = np.array(indices)
        sub_matrix = sim_matrix[np.ix_(idx, idx)]
        sub_mask = np.triu(np.ones((len(idx), len(idx)), dtype=bool), k=1)
        if sub_mask.sum() == 0:
            continue

        intra_sim = float(sub_matrix[sub_mask].mean())

        # Inter-cluster: similarity between tagged and non-tagged docs
        non_indices = [i for i in range(n) if i not in set(indices)]
        if non_indices:
            inter_matrix = sim_matrix[np.ix_(idx, np.array(non_indices))]
            inter_sim = float(inter_matrix.mean())
        else:
            inter_sim = global_mean_sim

        # Coherence ratio: how much tighter is the cluster than random?
        coherence_ratio = intra_sim / max(global_mean_sim, 1e-9)
        # Separation: intra vs inter
        separation = intra_sim - inter_sim

        dim_coherence[dim] = {
            "num_docs": len(indices),
            "intra_similarity": round(intra_sim, 4),
            "inter_similarity": round(inter_sim, 4),
            "global_similarity": round(global_mean_sim, 4),
            "coherence_ratio": round(coherence_ratio, 3),
            "separation": round(separation, 4),
        }

    # Rank by coherence ratio
    coherence_ranking = sorted(
        dim_coherence.items(),
        key=lambda x: x[1]["coherence_ratio"],
        reverse=True,
    )

    # Low-coherence dimensions: not forming real semantic clusters
    low_coherence = [
        {"dimension": d, **v}
        for d, v in coherence_ranking
        if v["coherence_ratio"] < 1.05 or v["separation"] < 0.01
    ]

    return {
        "global_mean_similarity": round(global_mean_sim, 4),
        "dimension_coherence": dict(coherence_ranking),
        "low_coherence_dimensions": low_coherence,
        "coherence_ranking": [
            {"dimension": d, "coherence_ratio": v["coherence_ratio"], "separation": v["separation"]}
            for d, v in coherence_ranking
        ],
    }


# =========================================================================
# STAGE 11: Term–Dimension Alignment
# =========================================================================

def analyze_term_alignment(enrichments, manifest):
    """Check whether enrichment-extracted terms align with classification dimensions.

    Documents with many domain-specific terms but few active dimensions may
    indicate the ontology has gaps. Documents with many dimensions but no
    extracted terms may indicate over-tagging.
    """
    alignment_data = []

    for path, enrich in enrichments.items():
        terms = enrich.get("terms", [])
        text = enrich.get("text", "")

        # Extract actual term strings
        term_names = []
        for t in terms:
            name_span = t.get("name", {})
            start, end = name_span.get("start", 0), name_span.get("end", 0)
            if start < end <= len(text):
                term_names.append(text[start:end].strip().lower())

        entry = manifest.get(path, {})
        dims = entry.get("content_dimensions", [])

        alignment_data.append({
            "source": path,
            "num_terms": len(terms),
            "num_dimensions": len(dims),
            "dimensions": dims,
            "terms": term_names[:10],  # Sample
            "term_mention_count": sum(len(t.get("mentions", [])) for t in terms),
        })

    # Find misalignments
    term_rich_dim_poor = [
        d for d in alignment_data
        if d["num_terms"] >= 3 and d["num_dimensions"] <= 1
    ]
    term_poor_dim_rich = [
        d for d in alignment_data
        if d["num_terms"] == 0 and d["num_dimensions"] >= 6
    ]

    # Term frequency across dimensions
    dim_term_counts = defaultdict(list)
    for d in alignment_data:
        for dim in d["dimensions"]:
            dim_term_counts[dim].append(d["num_terms"])

    dim_term_summary = {}
    for dim, counts in sorted(dim_term_counts.items()):
        arr = np.array(counts)
        dim_term_summary[dim] = {
            "mean_terms": round(float(arr.mean()), 1),
            "median_terms": float(np.median(arr)),
            "docs_with_terms": int(np.sum(arr > 0)),
            "total_docs": len(counts),
        }

    return {
        "total_docs_with_terms": sum(1 for d in alignment_data if d["num_terms"] > 0),
        "term_rich_dim_poor": sorted(
            term_rich_dim_poor,
            key=lambda x: x["num_terms"],
            reverse=True,
        )[:10],
        "term_poor_dim_rich": sorted(
            term_poor_dim_rich,
            key=lambda x: x["num_dimensions"],
            reverse=True,
        )[:10],
        "dimension_term_density": dim_term_summary,
    }


# =========================================================================
# STAGE 12: Synthesize Recommendations
# =========================================================================

def synthesize_recommendations(
    activation, score_dist, coactivation, threshold,
    confusion, affinity, info_theory, clusters,
    graph_topology=None, embedding_coherence=None, term_alignment=None,
):
    """Generate actionable schema refinement recommendations."""
    recommendations = []

    # R1: Drop low-activation dimensions
    for dim, rate in activation["low_activation"].items():
        recommendations.append({
            "type": "DROP_DIMENSION",
            "priority": "high",
            "dimension": dim,
            "reason": f"Activation rate {rate:.1%} is below 10% — too sparse to be useful",
            "action": f"Remove '{dim}' from the ontology",
        })

    # R2: Merge redundant pairs
    for pair in coactivation["redundant_pairs"]:
        if pair["jaccard"] > 0.7:
            recommendations.append({
                "type": "MERGE_DIMENSIONS",
                "priority": "high",
                "dimensions": [pair["dim_a"], pair["dim_b"]],
                "reason": f"Jaccard similarity {pair['jaccard']:.2f} — these dimensions "
                         f"co-activate in most documents and are measuring the same signal",
                "action": f"Merge '{pair['dim_a']}' and '{pair['dim_b']}' into a single dimension",
            })
        elif pair["jaccard"] > 0.6:
            recommendations.append({
                "type": "INVESTIGATE_OVERLAP",
                "priority": "medium",
                "dimensions": [pair["dim_a"], pair["dim_b"]],
                "reason": f"Jaccard similarity {pair['jaccard']:.2f} — significant overlap",
                "action": "Review queries for semantic overlap; consider merging or sharpening",
            })

    # R3: Recalibrate dimensions with poor score separation
    for dim, dist in score_dist.items():
        if dist["std"] < 0.1:
            recommendations.append({
                "type": "RECALIBRATE_QUERY",
                "priority": "medium",
                "dimension": dim,
                "reason": f"Score std={dist['std']:.3f} — all documents score similarly, "
                         f"no discriminative power (mean={dist['mean']:.3f})",
                "action": f"Rewrite the IQL query for '{dim}' to be more specific",
            })

    # R4: Adjust per-dimension thresholds
    for dim, th in threshold.items():
        optimal = th["optimal_threshold"]
        if abs(optimal - 0.6) > 0.1:
            recommendations.append({
                "type": "ADJUST_THRESHOLD",
                "priority": "low",
                "dimension": dim,
                "reason": f"Optimal threshold {optimal:.2f} differs from global 0.6",
                "action": f"Consider per-dimension threshold of {optimal:.2f} for '{dim}'",
            })

    # R5: Flag excessive activation
    for dim, rate in activation["high_activation"].items():
        recommendations.append({
            "type": "SHARPEN_QUERY",
            "priority": "medium",
            "dimension": dim,
            "reason": f"Activation rate {rate:.1%} — too broad, nearly every document matches",
            "action": f"Make '{dim}' query more specific to reduce false positives",
        })

    # R6: Surprising cross-category affinities
    for surprise in affinity["surprising_affinities"]:
        recommendations.append({
            "type": "INVESTIGATE_AFFINITY",
            "priority": "low",
            "dimension": surprise["dimension"],
            "category": surprise["category"],
            "reason": surprise["note"],
            "action": "Check if this is genuine signal or classifier noise",
        })

    # R7: Cognitive domain rebalancing
    dist = confusion["domain_distribution"]
    total = sum(dist.values())
    for domain, count in dist.items():
        rate = count / total
        if rate > 0.6:
            recommendations.append({
                "type": "REBALANCE_COGNITIVE",
                "priority": "high",
                "domain": domain,
                "reason": f"{domain} accounts for {rate:.0%} of documents — "
                         f"classifier may be poorly calibrated or query too broad",
                "action": f"Sharpen '{domain}' query or add discriminative features "
                         f"to competing domains",
            })

    # R8: Low embedding coherence — dimension doesn't form a real semantic cluster
    if embedding_coherence and "low_coherence_dimensions" in embedding_coherence:
        for item in embedding_coherence["low_coherence_dimensions"]:
            recommendations.append({
                "type": "LOW_EMBEDDING_COHERENCE",
                "priority": "high",
                "dimension": item["dimension"],
                "reason": f"Coherence ratio {item['coherence_ratio']:.2f}, "
                         f"separation {item['separation']:.3f} — documents tagged with "
                         f"this dimension are not semantically closer to each other than "
                         f"to the general corpus in embedding space",
                "action": f"'{item['dimension']}' is not capturing a real semantic concept. "
                         f"Rewrite the query to be more specific or consider dropping it.",
            })

    # R9: Under-tagged complex documents (rich graph structure, few dimensions)
    if graph_topology and graph_topology.get("under_tagged_complex_documents"):
        n_under = len(graph_topology["under_tagged_complex_documents"])
        if n_under >= 3:
            recommendations.append({
                "type": "UNDER_TAGGED_DOCUMENTS",
                "priority": "medium",
                "reason": f"{n_under} documents have rich structural complexity "
                         f"(many segments, crossrefs, terms) but ≤2 active dimensions — "
                         f"the ontology may be missing content types present in these docs",
                "action": "Review these documents to identify missing ontology dimensions",
                "examples": [
                    d["source"]
                    for d in graph_topology["under_tagged_complex_documents"][:5]
                ],
            })

    # R10: Term-rich but dimension-poor documents
    if term_alignment and term_alignment.get("term_rich_dim_poor"):
        n_misaligned = len(term_alignment["term_rich_dim_poor"])
        if n_misaligned >= 2:
            recommendations.append({
                "type": "TERM_DIMENSION_MISALIGNMENT",
                "priority": "medium",
                "reason": f"{n_misaligned} documents have ≥3 extracted domain terms but ≤1 "
                         f"active dimension — terminology density suggests content the "
                         f"ontology fails to classify",
                "action": "Inspect extracted terms for these documents and consider "
                         "adding new dimensions that cover this content",
                "examples": [
                    {"source": d["source"], "terms": d["terms"][:5]}
                    for d in term_alignment["term_rich_dim_poor"][:5]
                ],
            })

    # Sort by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    recommendations.sort(key=lambda r: priority_order.get(r["priority"], 3))

    return recommendations


# =========================================================================
# MAIN
# =========================================================================

def main():
    preview = "--preview" in sys.argv
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading classification data...")
    manifest, docs = load_data()
    n = len(manifest)
    print(f"  {n} documents loaded" + (" (preview mode)" if preview else ""))

    print("\n[Stage 1/12] Dimension activation analysis...")
    activation = analyze_activation(manifest)
    print(f"  Mean dims/doc: {activation['dimensions_per_doc']['mean']:.1f} "
          f"(median: {activation['dimensions_per_doc']['median']:.0f})")

    print("[Stage 2/12] Score distribution analysis...")
    score_dist = analyze_score_distributions(docs)
    bimodal = [d for d, v in score_dist.items() if v["is_bimodal"]]
    print(f"  Bimodal dimensions: {bimodal or 'none'}")

    print("[Stage 3/12] Co-activation matrix (Jaccard)...")
    coactivation = analyze_coactivation(manifest)
    print(f"  Redundant pairs (Jaccard>0.6): {len(coactivation['redundant_pairs'])}")

    print("[Stage 4/12] Threshold sensitivity...")
    threshold = analyze_threshold_sensitivity(docs)
    optimal_range = [v["optimal_threshold"] for v in threshold.values()]
    if optimal_range:
        print(f"  Optimal threshold range: {min(optimal_range):.2f}–{max(optimal_range):.2f}")

    print("[Stage 5/12] Cognitive domain confusion...")
    confusion = analyze_cognitive_confusion(docs)
    print(f"  Ambiguous documents: {confusion['total_ambiguous']} "
          f"({confusion['ambiguity_rate']:.0%})")
    print(f"  Domain distribution: {confusion['domain_distribution']}")

    print("[Stage 6/12] Category–dimension affinity...")
    affinity = analyze_category_affinity(manifest)
    print(f"  Surprising affinities: {len(affinity['surprising_affinities'])}")

    print("[Stage 7/12] Information-theoretic analysis...")
    info_theory = analyze_information_theory(manifest)
    top3 = info_theory["discriminability_ranking"][:3]
    print(f"  Most discriminative: {[d['dimension'] for d in top3]}")

    print("[Stage 8/12] Dimension profile clustering...")
    clusters = analyze_clusters(manifest)
    sizes = [c["size"] for c in clusters["clusters"].values()]
    print(f"  Cluster sizes: {sizes}")

    print("\n[Stage 9/12] Knowledge graph topology analysis...")
    enrichments = load_enrichment_data(manifest)
    graph_topology = analyze_graph_topology(enrichments, manifest)
    print(f"  Corpus: {graph_topology['corpus_totals']['total_segments']} segments, "
          f"{graph_topology['corpus_totals']['total_crossrefs']} crossrefs, "
          f"{graph_topology['corpus_totals']['total_terms']} terms")
    print(f"  Under-tagged complex docs: "
          f"{len(graph_topology['under_tagged_complex_documents'])}")

    print("[Stage 10/12] Embedding-space coherence analysis...")
    doc_embeddings, seg_embeddings = load_embedding_data(manifest)
    embedding_coherence = analyze_embedding_coherence(doc_embeddings, manifest)
    if "error" not in embedding_coherence:
        low_coh = len(embedding_coherence.get("low_coherence_dimensions", []))
        print(f"  Global mean similarity: {embedding_coherence['global_mean_similarity']}")
        print(f"  Low-coherence dimensions: {low_coh}")
        if embedding_coherence.get("coherence_ranking"):
            top = embedding_coherence["coherence_ranking"][0]
            print(f"  Most coherent: {top['dimension']} "
                  f"(ratio={top['coherence_ratio']:.2f})")
    else:
        print(f"  Skipped: {embedding_coherence['error']}")

    print("[Stage 11/12] Term–dimension alignment...")
    term_alignment = analyze_term_alignment(enrichments, manifest)
    print(f"  Docs with terms: {term_alignment['total_docs_with_terms']}")
    print(f"  Term-rich/dim-poor: {len(term_alignment['term_rich_dim_poor'])}")
    print(f"  Term-poor/dim-rich: {len(term_alignment['term_poor_dim_rich'])}")

    # Synthesize recommendations
    print("\n[Stage 12/12] Synthesizing recommendations...")
    recommendations = synthesize_recommendations(
        activation, score_dist, coactivation, threshold,
        confusion, affinity, info_theory, clusters,
        graph_topology, embedding_coherence, term_alignment,
    )

    high = sum(1 for r in recommendations if r["priority"] == "high")
    med = sum(1 for r in recommendations if r["priority"] == "medium")
    low = sum(1 for r in recommendations if r["priority"] == "low")
    print(f"  {len(recommendations)} recommendations: {high} high, {med} medium, {low} low")

    # Print high-priority recommendations
    print("\n" + "=" * 70)
    print("HIGH-PRIORITY RECOMMENDATIONS")
    print("=" * 70)
    for r in recommendations:
        if r["priority"] != "high":
            continue
        print(f"\n  [{r['type']}] {r.get('dimension', r.get('domain', r.get('dimensions', '')))}")
        print(f"  Reason: {r['reason']}")
        print(f"  Action: {r['action']}")

    # Save full results
    full_report = {
        "metadata": {
            "documents_analyzed": n,
            "preview_mode": preview,
        },
        "stage_1_activation": activation,
        "stage_2_score_distributions": score_dist,
        "stage_3_coactivation": coactivation,
        "stage_4_threshold_sensitivity": threshold,
        "stage_5_cognitive_confusion": confusion,
        "stage_6_category_affinity": affinity,
        "stage_7_information_theory": info_theory,
        "stage_8_clusters": clusters,
        "stage_9_graph_topology": graph_topology,
        "stage_10_embedding_coherence": embedding_coherence,
        "stage_11_term_alignment": term_alignment,
        "recommendations": recommendations,
    }

    report_path = OUTPUT_DIR / "schema_analysis_report.json"
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    print(f"\nFull report saved: {report_path}")
    print(f"Results directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
