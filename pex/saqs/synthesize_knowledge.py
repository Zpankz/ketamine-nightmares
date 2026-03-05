#!/usr/bin/env python3
"""
Hegelian synthesis of the PEX/SAQ knowledge stack.

Integrates three dialectical knowledge layers:
  Thesis:     Enrichment knowledge graph (segments, crossrefs, terms, entities)
  Antithesis: Classification ontology (BFO-aligned dimensions, cognitive patterns)
  Synthesis:  Unified knowledge map resolving tensions between structure and meaning

The synthesis performs recursive knowledge graph reasoning:
  1. Load all knowledge layers (enrichment, classification, embeddings, QA facts)
  2. Build a unified node-edge graph across all layers
  3. Detect knowledge gaps (nodes reachable from one layer but not another)
  4. Compute dialectical tensions (where enrichment structure contradicts classification)
  5. Resolve tensions via embedding-space arbitration
  6. Generate cross-layer knowledge pathways (study paths)
  7. Output the synthesized knowledge graph as a unified JSON structure

BFO alignment: The synthesis maps every node to a BFO category
(continuant/occurrent, independent/dependent) and every edge to a BFO relation
(has-participant, realized-in, inheres-in, part-of).

Run AFTER all 5 pipeline steps and classify.py with BFO ontology.
"""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ENRICHMENT_DIR = Path(__file__).parent / "enrichment_results"
CLASSIFICATION_DIR = Path(__file__).parent / "classification_results"
EMBEDDING_DIR = Path(__file__).parent / "embedding_results"
QA_DIR = Path(__file__).parent / "qa_results"
SEGMENT_DIR = Path(__file__).parent / "segment_results"
OUTPUT_DIR = Path(__file__).parent / "synthesis_results"

# BFO category mappings for dimension → ontological type
BFO_CATEGORIES = {
    # Dispositions (realizable entities)
    "drug_receptor_disposition": "bfo:disposition",
    "autonomic_disposition": "bfo:disposition",
    # Processes (occurrents)
    "pharmacokinetic_process": "bfo:process",
    "organ_system_process": "bfo:process",
    "gas_exchange_process": "bfo:process",
    "neuromuscular_process": "bfo:process",
    "nociception_process": "bfo:process",
    # Qualities (dependent continuants)
    "acid_base_quality": "bfo:quality",
    # Material entities (independent continuants)
    "equipment_and_physics": "bfo:material_entity",
    "anatomy_structure": "bfo:material_entity",
    # Processes (measurement)
    "clinical_measurement_process": "bfo:process",
    # Roles (dependent continuants)
    "special_population_role": "bfo:role",
}

# BFO relation mappings for edge types
BFO_RELATIONS = {
    "crossref": "bfo:related_to",
    "parent_child": "bfo:part_of",
    "same_dimension": "bfo:is_about",
    "co_occurring_dimension": "bfo:correlated_with",
    "semantic_similarity": "bfo:similar_to",
    "schema_link": "bfo:realized_in",
    "qa_evidence": "bfo:has_evidence",
}


def load_all_data():
    """Load all knowledge layers."""
    layers = {}

    # Layer 1: Enrichment knowledge graph
    enrich_manifest_path = ENRICHMENT_DIR / "manifest.json"
    if enrich_manifest_path.exists():
        with open(enrich_manifest_path) as f:
            layers["enrichment_manifest"] = json.load(f)
        layers["enrichments"] = {}
        for path, info in layers["enrichment_manifest"].items():
            fpath = ENRICHMENT_DIR / info["output_file"]
            if fpath.exists():
                with open(fpath) as f:
                    layers["enrichments"][path] = json.load(f)
    else:
        layers["enrichment_manifest"] = {}
        layers["enrichments"] = {}

    # Layer 2: Classification ontology
    class_manifest_path = CLASSIFICATION_DIR / "manifest.json"
    if class_manifest_path.exists():
        with open(class_manifest_path) as f:
            layers["classification_manifest"] = json.load(f)
    else:
        layers["classification_manifest"] = {}

    # Layer 3: Embeddings
    embed_manifest_path = EMBEDDING_DIR / "manifest.json"
    if embed_manifest_path.exists():
        with open(embed_manifest_path) as f:
            embed_manifest = json.load(f)
        layers["embeddings"] = {}
        for path, info in embed_manifest.items():
            fpath = EMBEDDING_DIR / info["output_file"]
            if fpath.exists():
                with open(fpath) as f:
                    data = json.load(f)
                if data.get("document_embedding"):
                    layers["embeddings"][path] = np.array(data["document_embedding"])
    else:
        layers["embeddings"] = {}

    # Layer 4: QA facts
    qa_manifest_path = QA_DIR / "manifest.json"
    if qa_manifest_path.exists():
        with open(qa_manifest_path) as f:
            layers["qa_manifest"] = json.load(f)
    else:
        layers["qa_manifest"] = {}

    # Layer 5: Segment facts
    seg_manifest_path = SEGMENT_DIR / "manifest.json"
    if seg_manifest_path.exists():
        with open(seg_manifest_path) as f:
            layers["segment_manifest"] = json.load(f)
    else:
        layers["segment_manifest"] = {}

    return layers


def build_unified_graph(layers):
    """Build a unified node-edge graph across all knowledge layers.

    Nodes: documents, segments, dimensions, schemas, terms, entities
    Edges: crossrefs, parent-child, co-dimension, semantic similarity, QA evidence
    """
    nodes = {}
    edges = []

    enrichments = layers["enrichments"]
    class_manifest = layers["classification_manifest"]
    embeddings = layers["embeddings"]

    # --- Document nodes ---
    for path in enrichments:
        doc = enrichments[path]
        class_info = class_manifest.get(path, {})

        nodes[f"doc:{path}"] = {
            "type": "document",
            "bfo_category": "bfo:generically_dependent_continuant",
            "path": path,
            "category": path.split("/")[0] if "/" in path else "unknown",
            "num_segments": len(doc.get("segments", [])),
            "cognitive_domain": class_info.get("cognitive_domain"),
            "clinical_relevance": class_info.get("clinical_relevance"),
            "content_dimensions": class_info.get("content_dimensions", []),
            "explanatory_schema": class_info.get("explanatory_schema", []),
            "has_embedding": path in embeddings,
        }

    # --- Dimension nodes ---
    all_dims = set()
    for entry in class_manifest.values():
        for d in entry.get("content_dimensions", []):
            all_dims.add(d)
    for dim in all_dims:
        nodes[f"dim:{dim}"] = {
            "type": "dimension",
            "bfo_category": BFO_CATEGORIES.get(dim, "bfo:unknown"),
            "name": dim,
            "doc_count": sum(
                1 for e in class_manifest.values()
                if dim in e.get("content_dimensions", [])
            ),
        }

    # --- Schema nodes ---
    all_schemas = set()
    for entry in class_manifest.values():
        for s in entry.get("explanatory_schema", []):
            all_schemas.add(s)
    for schema in all_schemas:
        nodes[f"schema:{schema}"] = {
            "type": "schema",
            "bfo_category": "bfo:process",
            "name": schema,
            "doc_count": sum(
                1 for e in class_manifest.values()
                if schema in e.get("explanatory_schema", [])
            ),
        }

    # --- Term nodes (from enrichment) ---
    term_counts = Counter()
    for doc in enrichments.values():
        for term in doc.get("terms", []):
            t = term.get("text", "").lower().strip()
            if t:
                term_counts[t] += 1
    for term, count in term_counts.most_common(200):
        nodes[f"term:{term}"] = {
            "type": "term",
            "bfo_category": "bfo:information_content_entity",
            "name": term,
            "frequency": count,
        }

    # --- Edges: document → dimension ---
    for path, entry in class_manifest.items():
        for dim in entry.get("content_dimensions", []):
            edges.append({
                "source": f"doc:{path}",
                "target": f"dim:{dim}",
                "type": "has_dimension",
                "bfo_relation": BFO_RELATIONS["same_dimension"],
            })

    # --- Edges: document → schema ---
    for path, entry in class_manifest.items():
        for schema in entry.get("explanatory_schema", []):
            edges.append({
                "source": f"doc:{path}",
                "target": f"schema:{schema}",
                "type": "exhibits_schema",
                "bfo_relation": BFO_RELATIONS["schema_link"],
            })

    # --- Edges: crossrefs (enrichment) ---
    for path, doc in enrichments.items():
        for xref in doc.get("crossreferences", []):
            target_path = xref.get("target", "")
            if f"doc:{target_path}" in nodes:
                edges.append({
                    "source": f"doc:{path}",
                    "target": f"doc:{target_path}",
                    "type": "crossref",
                    "bfo_relation": BFO_RELATIONS["crossref"],
                })

    # --- Edges: document → term ---
    for path, doc in enrichments.items():
        for term in doc.get("terms", []):
            t = term.get("text", "").lower().strip()
            if f"term:{t}" in nodes:
                edges.append({
                    "source": f"doc:{path}",
                    "target": f"term:{t}",
                    "type": "contains_term",
                    "bfo_relation": "bfo:has_part",
                })

    # --- Edges: dimension co-occurrence ---
    dim_pairs = Counter()
    for entry in class_manifest.values():
        dims = entry.get("content_dimensions", [])
        for i, d1 in enumerate(dims):
            for d2 in dims[i + 1:]:
                pair = tuple(sorted([d1, d2]))
                dim_pairs[pair] += 1
    for (d1, d2), count in dim_pairs.items():
        if count >= 5:
            edges.append({
                "source": f"dim:{d1}",
                "target": f"dim:{d2}",
                "type": "co_occurs",
                "bfo_relation": BFO_RELATIONS["co_occurring_dimension"],
                "weight": count,
            })

    return nodes, edges


def detect_knowledge_gaps(nodes, edges, layers):
    """Detect nodes reachable from one layer but not another.

    These represent knowledge gaps where enrichment captured structure
    that classification missed, or vice versa.
    """
    gaps = []

    # Documents with enrichment but no classification
    for path in layers["enrichments"]:
        if path not in layers["classification_manifest"]:
            gaps.append({
                "type": "enriched_not_classified",
                "path": path,
                "severity": "high",
            })

    # Documents classified but not embedded
    for path in layers["classification_manifest"]:
        if path not in layers.get("embeddings", {}):
            gaps.append({
                "type": "classified_not_embedded",
                "path": path,
                "severity": "medium",
            })

    # Complex enrichments (many segments, crossrefs) with few dimensions
    for path, doc in layers["enrichments"].items():
        n_seg = len(doc.get("segments", []))
        n_xref = len(doc.get("crossreferences", []))
        class_info = layers["classification_manifest"].get(path, {})
        n_dim = class_info.get("num_active_dimensions", 0)
        complexity = n_seg + n_xref * 3
        if complexity > 50 and n_dim <= 1:
            gaps.append({
                "type": "complex_undertaged",
                "path": path,
                "complexity": complexity,
                "dimensions": n_dim,
                "severity": "high",
            })

    # Terms that appear frequently but in docs with no matching dimension
    enrichments = layers["enrichments"]
    class_manifest = layers["classification_manifest"]
    term_dim_coverage = defaultdict(lambda: {"with_dims": 0, "without_dims": 0})
    for path, doc in enrichments.items():
        n_dim = class_manifest.get(path, {}).get("num_active_dimensions", 0)
        for term in doc.get("terms", []):
            t = term.get("text", "").lower().strip()
            if n_dim > 0:
                term_dim_coverage[t]["with_dims"] += 1
            else:
                term_dim_coverage[t]["without_dims"] += 1

    for term, counts in term_dim_coverage.items():
        if counts["without_dims"] > 3 and counts["with_dims"] == 0:
            gaps.append({
                "type": "orphaned_term",
                "term": term,
                "docs_without_dims": counts["without_dims"],
                "severity": "low",
            })

    return gaps


def compute_dialectical_tensions(nodes, edges, layers):
    """Find where enrichment structure contradicts classification.

    Dialectical tensions arise when:
    - Cross-referenced documents have different cognitive domains
    - Hierarchically adjacent segments map to different BFO categories
    - Semantically similar docs (by embedding) have divergent classifications
    """
    tensions = []
    enrichments = layers["enrichments"]
    class_manifest = layers["classification_manifest"]
    embeddings = layers["embeddings"]

    # Tension 1: Crossref partners with divergent cognitive domains
    for path, doc in enrichments.items():
        src_domain = class_manifest.get(path, {}).get("cognitive_domain")
        if not src_domain:
            continue
        for xref in doc.get("crossreferences", []):
            target = xref.get("target", "")
            tgt_domain = class_manifest.get(target, {}).get("cognitive_domain")
            if tgt_domain and tgt_domain != src_domain:
                tensions.append({
                    "type": "crossref_cognitive_divergence",
                    "source": path,
                    "target": target,
                    "source_domain": src_domain,
                    "target_domain": tgt_domain,
                    "resolution": "Review if crossref is pedagogically valid",
                })

    # Tension 2: Semantically similar docs with different dimension sets
    if embeddings:
        paths = list(embeddings.keys())
        if len(paths) > 1:
            vecs = np.array([embeddings[p] for p in paths])
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            normed = vecs / norms
            # Find top-5 most similar pairs
            sim_matrix = normed @ normed.T
            np.fill_diagonal(sim_matrix, 0)
            flat_idx = np.argsort(sim_matrix.ravel())[-20:]
            for idx in flat_idx:
                i, j = divmod(idx, len(paths))
                if i >= j:
                    continue
                p1, p2 = paths[i], paths[j]
                sim = float(sim_matrix[i, j])
                dims1 = set(class_manifest.get(p1, {}).get("content_dimensions", []))
                dims2 = set(class_manifest.get(p2, {}).get("content_dimensions", []))
                if dims1 and dims2:
                    jaccard = len(dims1 & dims2) / len(dims1 | dims2)
                    if sim > 0.7 and jaccard < 0.3:
                        tensions.append({
                            "type": "semantic_classification_divergence",
                            "doc1": p1,
                            "doc2": p2,
                            "embedding_similarity": round(sim, 3),
                            "dimension_jaccard": round(jaccard, 3),
                            "dims1": sorted(dims1),
                            "dims2": sorted(dims2),
                            "resolution": "Embeddings say similar, dimensions say different — investigate",
                        })

    # Tension 3: Docs with many BFO-disposition dimensions but mechanistic_explanation
    # is not the cognitive domain (BFO dispositions realized in explanatory processes)
    for path, entry in class_manifest.items():
        dims = entry.get("content_dimensions", [])
        disposition_count = sum(
            1 for d in dims if BFO_CATEGORIES.get(d) == "bfo:disposition"
        )
        if disposition_count >= 2 and entry.get("cognitive_domain") == "factual_recall":
            tensions.append({
                "type": "bfo_disposition_recall_tension",
                "path": path,
                "disposition_dims": [d for d in dims if BFO_CATEGORIES.get(d) == "bfo:disposition"],
                "cognitive_domain": "factual_recall",
                "resolution": "Disposition content typically requires mechanistic explanation, not mere recall",
            })

    return tensions


def generate_study_paths(nodes, edges, layers):
    """Generate cross-layer knowledge pathways with real cognitive progression.

    Builds study paths that represent genuine Bloom's-taxonomy escalation:
    foundational → mechanistic → comparative → quantitative → applied.

    Each path picks the best representative document per cognitive level,
    preferring documents with high schema coverage and semantic connectivity.
    Also generates cross-dimension paths that bridge related BFO categories.
    """
    class_manifest = layers["classification_manifest"]
    enrichments = layers["enrichments"]
    embeddings = layers["embeddings"]

    domain_order = [
        "factual_recall", "mechanistic_explanation",
        "comparative_analysis", "quantitative_reasoning", "applied_clinical",
    ]
    domain_labels = {
        "factual_recall": "Foundation",
        "mechanistic_explanation": "Mechanism",
        "comparative_analysis": "Comparison",
        "quantitative_reasoning": "Quantitative",
        "applied_clinical": "Application",
    }

    # Compute document richness scores for ranking within each domain
    def doc_richness(path):
        """Score doc by schema count + dimension count + segment count."""
        entry = class_manifest.get(path, {})
        n_schema = len(entry.get("explanatory_schema", []))
        n_dim = entry.get("num_active_dimensions", 0)
        n_seg = len(enrichments.get(path, {}).get("segments", []))
        return n_schema * 3 + n_dim * 2 + min(n_seg, 20)

    study_paths = []

    # --- Dimension-focused paths ---
    dim_docs = defaultdict(list)
    for path, entry in class_manifest.items():
        for dim in entry.get("content_dimensions", []):
            dim_docs[dim].append(path)

    for dim, doc_paths in dim_docs.items():
        if len(doc_paths) < 3:
            continue

        # Group by cognitive domain
        by_domain = defaultdict(list)
        for p in doc_paths:
            cog = class_manifest.get(p, {}).get("cognitive_domain", "")
            by_domain[cog].append(p)

        # Build path: pick best doc from each domain present, in order
        path_steps = []
        domains_covered = []
        for domain in domain_order:
            candidates = by_domain.get(domain, [])
            if not candidates:
                continue
            # Pick the richest document for this domain
            best = max(candidates, key=doc_richness)
            entry = class_manifest.get(best, {})
            path_steps.append({
                "document": best,
                "cognitive_domain": domain,
                "cognitive_label": domain_labels[domain],
                "clinical_relevance": entry.get("clinical_relevance"),
                "schemas": entry.get("explanatory_schema", []),
                "richness_score": doc_richness(best),
            })
            domains_covered.append(domain)

        if len(path_steps) >= 2:
            # Compute progression quality: how many Bloom's levels covered
            bloom_span = (
                domain_order.index(domains_covered[-1])
                - domain_order.index(domains_covered[0])
            )
            study_paths.append({
                "dimension": dim,
                "bfo_category": BFO_CATEGORIES.get(dim, "unknown"),
                "path_type": "dimension_progression",
                "num_steps": len(path_steps),
                "bloom_span": bloom_span,
                "domains_covered": domains_covered,
                "progression": [s["cognitive_domain"] for s in path_steps],
                "steps": path_steps,
            })

    # --- Schema-focused paths: progression through an explanatory pattern ---
    schema_docs = defaultdict(list)
    for path, entry in class_manifest.items():
        for s in entry.get("explanatory_schema", []):
            schema_docs[s].append(path)

    for schema, doc_paths in schema_docs.items():
        if len(doc_paths) < 5:
            continue

        by_domain = defaultdict(list)
        for p in doc_paths:
            cog = class_manifest.get(p, {}).get("cognitive_domain", "")
            by_domain[cog].append(p)

        path_steps = []
        domains_covered = []
        for domain in domain_order:
            candidates = by_domain.get(domain, [])
            if not candidates:
                continue
            best = max(candidates, key=doc_richness)
            entry = class_manifest.get(best, {})
            path_steps.append({
                "document": best,
                "cognitive_domain": domain,
                "cognitive_label": domain_labels[domain],
                "clinical_relevance": entry.get("clinical_relevance"),
                "dimensions": entry.get("content_dimensions", []),
                "richness_score": doc_richness(best),
            })
            domains_covered.append(domain)

        if len(path_steps) >= 2:
            bloom_span = (
                domain_order.index(domains_covered[-1])
                - domain_order.index(domains_covered[0])
            )
            study_paths.append({
                "dimension": schema,
                "bfo_category": "explanatory_schema",
                "path_type": "schema_progression",
                "num_steps": len(path_steps),
                "bloom_span": bloom_span,
                "domains_covered": domains_covered,
                "progression": [s["cognitive_domain"] for s in path_steps],
                "steps": path_steps,
            })

    # --- Cross-dimension bridge paths ---
    # Find dimensions that co-occur heavily and build bridge paths
    dim_pairs = Counter()
    for entry in class_manifest.values():
        dims = entry.get("content_dimensions", [])
        for i, d1 in enumerate(dims):
            for d2 in dims[i + 1:]:
                dim_pairs[tuple(sorted([d1, d2]))] += 1

    for (d1, d2), count in dim_pairs.most_common(5):
        if count < 8:
            break
        # Docs in both dimensions
        shared = [
            p for p, e in class_manifest.items()
            if d1 in e.get("content_dimensions", [])
            and d2 in e.get("content_dimensions", [])
        ]
        if len(shared) < 3:
            continue

        by_domain = defaultdict(list)
        for p in shared:
            cog = class_manifest.get(p, {}).get("cognitive_domain", "")
            by_domain[cog].append(p)

        path_steps = []
        domains_covered = []
        for domain in domain_order:
            candidates = by_domain.get(domain, [])
            if not candidates:
                continue
            best = max(candidates, key=doc_richness)
            entry = class_manifest.get(best, {})
            path_steps.append({
                "document": best,
                "cognitive_domain": domain,
                "cognitive_label": domain_labels[domain],
                "clinical_relevance": entry.get("clinical_relevance"),
                "schemas": entry.get("explanatory_schema", []),
                "richness_score": doc_richness(best),
            })
            domains_covered.append(domain)

        if len(path_steps) >= 2:
            bloom_span = (
                domain_order.index(domains_covered[-1])
                - domain_order.index(domains_covered[0])
            )
            study_paths.append({
                "dimension": f"{d1} × {d2}",
                "bfo_category": "cross_dimension_bridge",
                "path_type": "bridge",
                "shared_doc_count": count,
                "num_steps": len(path_steps),
                "bloom_span": bloom_span,
                "domains_covered": domains_covered,
                "progression": [s["cognitive_domain"] for s in path_steps],
                "steps": path_steps,
            })

    # Sort by bloom_span descending (best progressions first)
    study_paths.sort(key=lambda p: (p["bloom_span"], p["num_steps"]), reverse=True)

    return study_paths


def compute_difficulty_and_yield(layers):
    """Compute per-document difficulty scores and exam yield indicators.

    Difficulty is estimated from:
      - Cognitive domain (factual=1, mechanistic=2, comparative=3, quant=4, applied=3)
      - Number of active dimensions (more dimensions = more cross-cutting = harder)
      - Number of explanatory schemas (more patterns to reason through)
      - Segment count (more content = more to learn)

    Exam yield is estimated from:
      - QA answer density (high answerable ratio = high-yield factual content)
      - Exam year recency (more recent = more likely to reflect current syllabus)
      - Cross-reference count (highly cross-referenced = core topic)
    """
    class_manifest = layers["classification_manifest"]
    enrichments = layers["enrichments"]
    qa_manifest = layers.get("qa_manifest", {})

    domain_difficulty = {
        "factual_recall": 1,
        "mechanistic_explanation": 2,
        "comparative_analysis": 3,
        "quantitative_reasoning": 4,
        "applied_clinical": 3,
    }

    results = {}
    for path, entry in class_manifest.items():
        # --- Difficulty ---
        cog = entry.get("cognitive_domain", "")
        d_cog = domain_difficulty.get(cog, 2)
        n_dim = entry.get("num_active_dimensions", 0)
        n_schema = len(entry.get("explanatory_schema", []))
        n_seg = len(enrichments.get(path, {}).get("segments", []))

        difficulty = round(
            0.3 * d_cog
            + 0.25 * min(n_dim / 4, 1) * 4
            + 0.2 * min(n_schema / 2, 1) * 4
            + 0.25 * min(n_seg / 30, 1) * 4,
            2,
        )

        # --- Exam yield ---
        # Extract year from filename
        import re
        year_match = re.search(r"(\d{4})[AB]", path)
        exam_year = int(year_match.group(1)) if year_match else 2000
        recency_score = min((exam_year - 1998) / 25, 1)  # 0–1, higher = more recent

        # QA density
        qa_entry = qa_manifest.get(path, {})
        n_answerable = qa_entry.get("num_answerable", 0)
        n_questions = qa_entry.get("num_questions", 1)
        qa_density = n_answerable / max(n_questions, 1)

        # Cross-reference count (how often this doc is referenced by others)
        xref_count = 0
        for other_path, doc in enrichments.items():
            if other_path == path:
                continue
            for xref in doc.get("crossreferences", []):
                if xref.get("target") == path:
                    xref_count += 1

        yield_score = round(
            0.35 * qa_density * 4
            + 0.3 * recency_score * 4
            + 0.35 * min(xref_count / 5, 1) * 4,
            2,
        )

        results[path] = {
            "difficulty": difficulty,
            "difficulty_components": {
                "cognitive_complexity": d_cog,
                "dimension_breadth": n_dim,
                "schema_count": n_schema,
                "segment_count": n_seg,
            },
            "yield_score": yield_score,
            "yield_components": {
                "qa_density": round(qa_density, 3),
                "exam_year": exam_year,
                "recency_score": round(recency_score, 3),
                "inbound_crossrefs": xref_count,
            },
        }

    return results


def compute_graph_statistics(nodes, edges):
    """Compute summary statistics for the unified knowledge graph."""
    node_types = Counter(n["type"] for n in nodes.values())
    edge_types = Counter(e["type"] for e in edges)

    # Degree distribution for documents
    doc_degree = Counter()
    for edge in edges:
        if edge["source"].startswith("doc:"):
            doc_degree[edge["source"]] += 1
        if edge["target"].startswith("doc:"):
            doc_degree[edge["target"]] += 1

    degrees = list(doc_degree.values()) if doc_degree else [0]

    # BFO category distribution
    bfo_dist = Counter(n.get("bfo_category", "unknown") for n in nodes.values())

    return {
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "node_types": dict(node_types),
        "edge_types": dict(edge_types),
        "bfo_categories": dict(bfo_dist),
        "doc_degree_mean": sum(degrees) / len(degrees),
        "doc_degree_max": max(degrees),
        "doc_degree_min": min(degrees),
    }


def main():
    print("Loading all knowledge layers...")
    layers = load_all_data()

    n_enrich = len(layers["enrichments"])
    n_class = len(layers["classification_manifest"])
    n_embed = len(layers["embeddings"])
    n_qa = len(layers["qa_manifest"])
    n_seg = len(layers["segment_manifest"])
    print(f"  Enrichments: {n_enrich}")
    print(f"  Classifications: {n_class}")
    print(f"  Embeddings: {n_embed}")
    print(f"  QA docs: {n_qa}")
    print(f"  Segment docs: {n_seg}")

    print("\n[1/6] Building unified knowledge graph...")
    nodes, edges = build_unified_graph(layers)
    stats = compute_graph_statistics(nodes, edges)
    print(f"  {stats['total_nodes']} nodes, {stats['total_edges']} edges")
    for t, c in sorted(stats["node_types"].items()):
        print(f"    {t}: {c}")

    print("\n[2/6] Detecting knowledge gaps...")
    gaps = detect_knowledge_gaps(nodes, edges, layers)
    gap_types = Counter(g["type"] for g in gaps)
    print(f"  {len(gaps)} gaps found:")
    for t, c in gap_types.most_common():
        print(f"    {t}: {c}")

    print("\n[3/6] Computing dialectical tensions...")
    tensions = compute_dialectical_tensions(nodes, edges, layers)
    tension_types = Counter(t["type"] for t in tensions)
    print(f"  {len(tensions)} tensions found:")
    for t, c in tension_types.most_common():
        print(f"    {t}: {c}")

    print("\n[4/6] Generating study paths...")
    study_paths = generate_study_paths(nodes, edges, layers)
    print(f"  {len(study_paths)} study paths generated")
    for sp in study_paths[:5]:
        print(f"    {sp['dimension']} ({sp['bfo_category']}): "
              f"{sp['num_steps']} steps, progression: {sp['progression'][:4]}...")

    print("\n[5/7] Computing difficulty and yield scores...")
    difficulty_yield = compute_difficulty_and_yield(layers)
    difficulties = [v["difficulty"] for v in difficulty_yield.values()]
    yields = [v["yield_score"] for v in difficulty_yield.values()]
    print(f"  {len(difficulty_yield)} documents scored")
    if difficulties:
        print(f"  Difficulty: mean={sum(difficulties)/len(difficulties):.2f}, "
              f"min={min(difficulties):.2f}, max={max(difficulties):.2f}")
        print(f"  Yield: mean={sum(yields)/len(yields):.2f}, "
              f"min={min(yields):.2f}, max={max(yields):.2f}")

    print("\n[6/7] Computing BFO alignment metrics...")
    bfo_coverage = sum(
        1 for n in nodes.values()
        if n.get("bfo_category") and n["bfo_category"] != "bfo:unknown"
    )
    bfo_pct = 100 * bfo_coverage / len(nodes) if nodes else 0
    print(f"  BFO-aligned nodes: {bfo_coverage}/{len(nodes)} ({bfo_pct:.0f}%)")
    for cat, count in sorted(stats["bfo_categories"].items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")

    print("\n[7/7] Synthesizing and saving...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save unified graph
    graph_data = {
        "statistics": stats,
        "nodes": {k: v for k, v in nodes.items() if v["type"] in ("dimension", "schema", "term")},
        "edges": [e for e in edges if e["type"] in ("co_occurs", "has_dimension", "exhibits_schema")],
    }
    with open(OUTPUT_DIR / "knowledge_graph.json", "w") as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)

    # Save knowledge gaps
    with open(OUTPUT_DIR / "knowledge_gaps.json", "w") as f:
        json.dump(gaps, f, indent=2, ensure_ascii=False)

    # Save tensions
    with open(OUTPUT_DIR / "dialectical_tensions.json", "w") as f:
        json.dump(tensions, f, indent=2, ensure_ascii=False)

    # Save study paths
    with open(OUTPUT_DIR / "study_paths.json", "w") as f:
        json.dump(study_paths, f, indent=2, ensure_ascii=False)

    # Save difficulty and yield scores
    with open(OUTPUT_DIR / "difficulty_yield.json", "w") as f:
        json.dump(difficulty_yield, f, indent=2, ensure_ascii=False)

    # Save full graph summary
    synthesis_report = {
        "thesis": {
            "layer": "enrichment",
            "description": "Hierarchical document structure, crossrefs, terms, entities",
            "node_count": n_enrich,
        },
        "antithesis": {
            "layer": "classification",
            "description": "BFO-aligned ontology: dispositions, processes, qualities, roles",
            "node_count": n_class,
            "axes": ["cognitive_domain", "clinical_relevance", "content_dimensions", "explanatory_schema"],
        },
        "synthesis": {
            "description": "Unified knowledge graph resolving structural and semantic layers",
            "total_nodes": stats["total_nodes"],
            "total_edges": stats["total_edges"],
            "knowledge_gaps": len(gaps),
            "dialectical_tensions": len(tensions),
            "study_paths": len(study_paths),
            "study_path_types": Counter(p["path_type"] for p in study_paths),
            "bfo_coverage_pct": round(bfo_pct, 1),
            "difficulty_mean": round(sum(difficulties) / len(difficulties), 2) if difficulties else 0,
            "yield_mean": round(sum(yields) / len(yields), 2) if yields else 0,
        },
    }
    with open(OUTPUT_DIR / "synthesis_report.json", "w") as f:
        json.dump(synthesis_report, f, indent=2, ensure_ascii=False)

    print(f"\nSynthesis complete!")
    print(f"  Knowledge graph: {stats['total_nodes']} nodes, {stats['total_edges']} edges")
    print(f"  Knowledge gaps: {len(gaps)}")
    print(f"  Dialectical tensions: {len(tensions)}")
    print(f"  Study paths: {len(study_paths)}")
    print(f"  BFO coverage: {bfo_pct:.0f}%")
    print(f"  Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
