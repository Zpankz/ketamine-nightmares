#!/usr/bin/env python3
"""
Generate a semantic knowledge index from the PEX/SAQ pipeline outputs.

Produces structured markdown files that gitnexus can index for semantic
interconnections across the SAQ corpus. Each file maps a knowledge domain
(BFO category, dimension, schema) to the documents and facts it contains.

This bridges the gap between gitnexus (code-oriented graph) and our
content-oriented knowledge graph (enrichment + classification + QA).

Run AFTER all pipeline steps complete.
"""

import json
from collections import defaultdict
from pathlib import Path

CLASSIFICATION_DIR = Path(__file__).parent / "classification_results"
ENRICHMENT_DIR = Path(__file__).parent / "enrichment_results"
QA_DIR = Path(__file__).parent / "qa_results"
SEGMENT_DIR = Path(__file__).parent / "segment_results"
OUTPUT_DIR = Path(__file__).parent / "semantic_index"


def load_manifests():
    """Load all pipeline manifests."""
    data = {}
    for name, d in [
        ("classification", CLASSIFICATION_DIR),
        ("enrichment", ENRICHMENT_DIR),
        ("qa", QA_DIR),
        ("segment", SEGMENT_DIR),
    ]:
        mp = d / "manifest.json"
        if mp.exists():
            with open(mp) as f:
                data[name] = json.load(f)
        else:
            data[name] = {}
    return data


def load_qa_facts(manifests):
    """Load all QA extractions."""
    facts_by_doc = {}
    for path, info in manifests["qa"].items():
        fpath = QA_DIR / info["output_file"]
        if fpath.exists():
            with open(fpath) as f:
                qa_data = json.load(f)
            facts_by_doc[path] = qa_data.get("qa_pairs", [])
    return facts_by_doc


def generate_dimension_pages(manifests, facts_by_doc):
    """Generate a markdown page per content dimension."""
    dim_dir = OUTPUT_DIR / "dimensions"
    dim_dir.mkdir(parents=True, exist_ok=True)

    # Group docs by dimension
    dim_docs = defaultdict(list)
    for path, entry in manifests["classification"].items():
        for dim in entry.get("content_dimensions", []):
            dim_docs[dim].append(path)

    for dim, doc_paths in dim_docs.items():
        lines = [f"# {dim.replace('_', ' ').title()}", ""]
        lines.append(f"BFO Category: {_bfo_cat(dim)}")
        lines.append(f"Documents: {len(doc_paths)}")
        lines.append("")

        # List documents with their cognitive domains and schemas
        lines.append("## Documents")
        lines.append("")
        for p in sorted(doc_paths):
            entry = manifests["classification"][p]
            cog = entry.get("cognitive_domain", "?")
            schemas = ", ".join(entry.get("explanatory_schema", [])) or "none"
            lines.append(f"- **{p}** — {cog} | schemas: {schemas}")

            # Add QA facts if available
            if p in facts_by_doc:
                for qa in facts_by_doc[p][:3]:
                    q = qa.get("question", "")[:80]
                    a = qa.get("answer", "")[:120]
                    if a:
                        lines.append(f"  - Q: {q}")
                        lines.append(f"    A: {a}")
        lines.append("")

        # Cross-references to other dimensions
        co_dims = defaultdict(int)
        for p in doc_paths:
            entry = manifests["classification"][p]
            for d in entry.get("content_dimensions", []):
                if d != dim:
                    co_dims[d] += 1
        if co_dims:
            lines.append("## Co-occurring Dimensions")
            lines.append("")
            for d, c in sorted(co_dims.items(), key=lambda x: -x[1]):
                lines.append(f"- [{d}](./{d}.md): {c} shared documents")
            lines.append("")

        with open(dim_dir / f"{dim}.md", "w") as f:
            f.write("\n".join(lines))

    return list(dim_docs.keys())


def generate_schema_pages(manifests, facts_by_doc):
    """Generate a markdown page per explanatory schema."""
    schema_dir = OUTPUT_DIR / "schemas"
    schema_dir.mkdir(parents=True, exist_ok=True)

    schema_docs = defaultdict(list)
    for path, entry in manifests["classification"].items():
        for s in entry.get("explanatory_schema", []):
            schema_docs[s].append(path)

    for schema, doc_paths in schema_docs.items():
        lines = [f"# {schema.replace('_', ' ').title()}", ""]
        lines.append(f"Documents: {len(doc_paths)}")
        lines.append("")

        # Group by cognitive domain for progression
        by_domain = defaultdict(list)
        for p in doc_paths:
            cog = manifests["classification"][p].get("cognitive_domain", "?")
            by_domain[cog].append(p)

        for domain in ["factual_recall", "mechanistic_explanation",
                        "comparative_analysis", "quantitative_reasoning",
                        "applied_clinical"]:
            if domain in by_domain:
                lines.append(f"## {domain.replace('_', ' ').title()} ({len(by_domain[domain])})")
                lines.append("")
                for p in sorted(by_domain[domain])[:10]:
                    dims = ", ".join(
                        manifests["classification"][p].get("content_dimensions", [])
                    ) or "none"
                    lines.append(f"- **{p}** — dims: {dims}")
                lines.append("")

        with open(schema_dir / f"{schema}.md", "w") as f:
            f.write("\n".join(lines))

    return list(schema_docs.keys())


def generate_crossref_graph(manifests):
    """Generate a crossref graph page."""
    lines = ["# Cross-Reference Knowledge Graph", ""]
    lines.append("Semantic links between documents from enrichment cross-references.")
    lines.append("")

    enrich_manifest = manifests["enrichment"]
    class_manifest = manifests["classification"]

    xref_count = 0
    for path, info in enrich_manifest.items():
        fpath = ENRICHMENT_DIR / info["output_file"]
        if not fpath.exists():
            continue
        with open(fpath) as f:
            doc = json.load(f)
        xrefs = doc.get("crossreferences", [])
        if not xrefs:
            continue

        src_dims = class_manifest.get(path, {}).get("content_dimensions", [])
        lines.append(f"## {path}")
        lines.append(f"Dimensions: {', '.join(src_dims) or 'none'}")
        lines.append("")
        for xref in xrefs:
            target = xref.get("target", "")
            tgt_dims = class_manifest.get(target, {}).get("content_dimensions", [])
            lines.append(f"- → **{target}** (dims: {', '.join(tgt_dims) or 'none'})")
            xref_count += 1
        lines.append("")

    lines.insert(2, f"Total cross-references: {xref_count}")

    with open(OUTPUT_DIR / "crossref_graph.md", "w") as f:
        f.write("\n".join(lines))

    return xref_count


def generate_overview(manifests, dimensions, schemas, xref_count):
    """Generate main index page."""
    n_class = len(manifests["classification"])
    n_enrich = len(manifests["enrichment"])
    n_qa = len(manifests["qa"])

    lines = [
        "# PEX/SAQ Semantic Knowledge Index",
        "",
        f"Corpus: {n_enrich} enriched documents, {n_class} classified, {n_qa} QA extracted",
        "",
        "## Content Dimensions (BFO-aligned)",
        "",
    ]
    for dim in sorted(dimensions):
        count = sum(
            1 for e in manifests["classification"].values()
            if dim in e.get("content_dimensions", [])
        )
        lines.append(f"- [{dim}](dimensions/{dim}.md) ({count} docs) — {_bfo_cat(dim)}")
    lines.append("")

    lines.append("## Explanatory Schemas")
    lines.append("")
    for schema in sorted(schemas):
        count = sum(
            1 for e in manifests["classification"].values()
            if schema in e.get("explanatory_schema", [])
        )
        lines.append(f"- [{schema}](schemas/{schema}.md) ({count} docs)")
    lines.append("")

    lines.append(f"## [Cross-Reference Graph](crossref_graph.md)")
    lines.append(f"{xref_count} semantic links between documents")
    lines.append("")

    with open(OUTPUT_DIR / "INDEX.md", "w") as f:
        f.write("\n".join(lines))


BFO_CATEGORIES = {
    "drug_receptor_disposition": "bfo:disposition",
    "autonomic_disposition": "bfo:disposition",
    "pharmacokinetic_process": "bfo:process",
    "organ_system_process": "bfo:process",
    "gas_exchange_process": "bfo:process",
    "neuromuscular_process": "bfo:process",
    "nociception_process": "bfo:process",
    "acid_base_quality": "bfo:quality",
    "equipment_and_physics": "bfo:material_entity",
    "special_population_role": "bfo:role",
}


def _bfo_cat(dim):
    return BFO_CATEGORIES.get(dim, "unknown")


def main():
    print("Loading pipeline manifests...")
    manifests = load_manifests()
    for name, m in manifests.items():
        print(f"  {name}: {len(m)} entries")

    print("\nLoading QA facts...")
    facts_by_doc = load_qa_facts(manifests)
    print(f"  {len(facts_by_doc)} docs with QA facts")

    print("\nGenerating dimension pages...")
    dimensions = generate_dimension_pages(manifests, facts_by_doc)
    print(f"  {len(dimensions)} dimension pages")

    print("Generating schema pages...")
    schemas = generate_schema_pages(manifests, facts_by_doc)
    print(f"  {len(schemas)} schema pages")

    print("Generating cross-reference graph...")
    xref_count = generate_crossref_graph(manifests)
    print(f"  {xref_count} cross-references")

    print("Generating overview index...")
    generate_overview(manifests, dimensions, schemas, xref_count)

    print(f"\nSemantic index generated: {OUTPUT_DIR}")
    print(f"  {len(dimensions)} dimensions, {len(schemas)} schemas, {xref_count} crossrefs")


if __name__ == "__main__":
    main()
