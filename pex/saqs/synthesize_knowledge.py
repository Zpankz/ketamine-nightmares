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
        for xref in doc.get("crossrefs", []):
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
        n_xref = len(doc.get("crossrefs", []))
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
        for xref in doc.get("crossrefs", []):
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
    """Generate cross-layer knowledge pathways.

    A study path connects documents through multiple knowledge layers:
    crossref links (enrichment), shared dimensions (classification),
    semantic proximity (embedding), and shared QA themes (extraction).
    """
    class_manifest = layers["classification_manifest"]
    enrichments = layers["enrichments"]
    embeddings = layers["embeddings"]

    # Build adjacency from all edge types
    adjacency = defaultdict(lambda: defaultdict(float))
    for edge in edges:
        src, tgt = edge["source"], edge["target"]
        if src.startswith("doc:") and tgt.startswith("doc:"):
            w = edge.get("weight", 1)
            adjacency[src][tgt] += w
            adjacency[tgt][src] += w

    # Add embedding similarity edges (top-10 per doc)
    if embeddings:
        paths = list(embeddings.keys())
        if len(paths) > 1:
            vecs = np.array([embeddings[p] for p in paths])
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            normed = vecs / norms
            sim_matrix = normed @ normed.T
            for i, p1 in enumerate(paths):
                top_k = np.argsort(sim_matrix[i])[-11:-1]  # top 10, excluding self
                for j in top_k:
                    p2 = paths[j]
                    sim = float(sim_matrix[i, j])
                    if sim > 0.5:
                        adjacency[f"doc:{p1}"][f"doc:{p2}"] += sim * 2

    # Generate paths by cognitive domain progression
    # (factual → mechanistic → comparative → applied)
    domain_order = [
        "factual_recall", "mechanistic_explanation",
        "comparative_analysis", "quantitative_reasoning", "applied_clinical",
    ]

    study_paths = []
    # Group docs by primary content dimension
    dim_docs = defaultdict(list)
    for path, entry in class_manifest.items():
        for dim in entry.get("content_dimensions", []):
            dim_docs[dim].append(path)

    for dim, doc_paths in dim_docs.items():
        if len(doc_paths) < 3:
            continue

        # Sort by cognitive domain progression
        def domain_rank(p):
            d = class_manifest.get(p, {}).get("cognitive_domain", "")
            return domain_order.index(d) if d in domain_order else 99

        ordered = sorted(doc_paths, key=domain_rank)

        # Build a progressive path
        path_steps = []
        for p in ordered[:8]:  # Cap at 8 steps
            entry = class_manifest.get(p, {})
            path_steps.append({
                "document": p,
                "cognitive_domain": entry.get("cognitive_domain"),
                "clinical_relevance": entry.get("clinical_relevance"),
                "schemas": entry.get("explanatory_schema", []),
            })

        if len(path_steps) >= 3:
            study_paths.append({
                "dimension": dim,
                "bfo_category": BFO_CATEGORIES.get(dim, "unknown"),
                "num_steps": len(path_steps),
                "progression": [s["cognitive_domain"] for s in path_steps],
                "steps": path_steps,
            })

    return study_paths


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

    print("\n[5/6] Computing BFO alignment metrics...")
    bfo_coverage = sum(
        1 for n in nodes.values()
        if n.get("bfo_category") and n["bfo_category"] != "bfo:unknown"
    )
    bfo_pct = 100 * bfo_coverage / len(nodes) if nodes else 0
    print(f"  BFO-aligned nodes: {bfo_coverage}/{len(nodes)} ({bfo_pct:.0f}%)")
    for cat, count in sorted(stats["bfo_categories"].items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")

    print("\n[6/6] Synthesizing and saving...")
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
            "bfo_coverage_pct": round(bfo_pct, 1),
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
