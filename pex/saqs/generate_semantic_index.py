#!/usr/bin/env python3
"""
Generate a semantic knowledge index from the PEX/SAQ pipeline outputs.

Produces structured markdown files that gitnexus can index for semantic
interconnections across the SAQ corpus. Each file maps a knowledge domain
(BFO category, dimension, schema) to the documents and facts it contains.

This bridges the gap between gitnexus (code-oriented graph) and our
content-oriented knowledge graph (enrichment + classification + QA).

Generates:
  - Dimension pages with difficulty/yield annotations and segment facts
  - Schema pages with cognitive domain progression
  - Cross-reference graph with dimension overlap analysis
  - Study path pages showing Bloom's progression per dimension
  - Corpus analytics overview with distribution histograms
  - High-yield topic guides ranked by exam relevance

Run AFTER all pipeline steps and synthesize_knowledge.py complete.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

CLASSIFICATION_DIR = Path(__file__).parent / "classification_results"
ENRICHMENT_DIR = Path(__file__).parent / "enrichment_results"
QA_DIR = Path(__file__).parent / "qa_results"
SEGMENT_DIR = Path(__file__).parent / "segment_results"
SYNTHESIS_DIR = Path(__file__).parent / "synthesis_results"
OUTPUT_DIR = Path(__file__).parent / "semantic_index"

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
    "anatomy_structure": "bfo:material_entity",
    "clinical_measurement_process": "bfo:process",
    "special_population_role": "bfo:role",
}


def _bfo_cat(dim):
    return BFO_CATEGORIES.get(dim, "unknown")


def _difficulty_bar(score):
    """Return a text difficulty indicator."""
    if score < 1.5:
        return "Easy"
    elif score < 2.5:
        return "Medium"
    elif score < 3.2:
        return "Hard"
    return "Very Hard"


def _yield_bar(score):
    """Return a text yield indicator."""
    if score < 1.0:
        return "Low"
    elif score < 2.0:
        return "Medium"
    elif score < 3.0:
        return "High"
    return "Very High"


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
    """Load all QA extractions with answerable facts only."""
    facts_by_doc = {}
    for path, info in manifests["qa"].items():
        fpath = QA_DIR / info["output_file"]
        if fpath.exists():
            with open(fpath) as f:
                qa_data = json.load(f)
            answerable = [
                e for e in qa_data.get("extractions", [])
                if e.get("answerable") and e.get("answers")
            ]
            facts_by_doc[path] = answerable
    return facts_by_doc


def load_difficulty_yield():
    """Load difficulty/yield scores from synthesis."""
    dy_path = SYNTHESIS_DIR / "difficulty_yield.json"
    if dy_path.exists():
        with open(dy_path) as f:
            return json.load(f)
    return {}


def load_study_paths():
    """Load study paths from synthesis."""
    sp_path = SYNTHESIS_DIR / "study_paths.json"
    if sp_path.exists():
        with open(sp_path) as f:
            return json.load(f)
    return []


def load_segment_facts(manifests):
    """Load segment-level key facts."""
    facts_by_doc = {}
    for path, info in manifests["segment"].items():
        if not info.get("output_file"):
            continue
        fpath = SEGMENT_DIR / info["output_file"]
        if fpath.exists():
            with open(fpath) as f:
                seg_data = json.load(f)
            facts_by_doc[path] = seg_data.get("segments", [])
    return facts_by_doc


def generate_dimension_pages(manifests, qa_facts, seg_facts, dy_scores):
    """Generate a rich markdown page per content dimension."""
    dim_dir = OUTPUT_DIR / "dimensions"
    dim_dir.mkdir(parents=True, exist_ok=True)

    dim_docs = defaultdict(list)
    for path, entry in manifests["classification"].items():
        for dim in entry.get("content_dimensions", []):
            dim_docs[dim].append(path)

    for dim, doc_paths in dim_docs.items():
        lines = [f"# {dim.replace('_', ' ').title()}", ""]
        lines.append(f"**BFO Category:** {_bfo_cat(dim)}")
        lines.append(f"**Documents:** {len(doc_paths)}")

        # Compute aggregate difficulty/yield for this dimension
        dim_difficulties = []
        dim_yields = []
        for p in doc_paths:
            if p in dy_scores:
                dim_difficulties.append(dy_scores[p]["difficulty"])
                dim_yields.append(dy_scores[p]["yield_score"])

        if dim_difficulties:
            avg_d = sum(dim_difficulties) / len(dim_difficulties)
            avg_y = sum(dim_yields) / len(dim_yields)
            lines.append(f"**Avg Difficulty:** {avg_d:.1f}/4 ({_difficulty_bar(avg_d)})")
            lines.append(f"**Avg Yield:** {avg_y:.1f}/4 ({_yield_bar(avg_y)})")
        lines.append("")

        # Cognitive domain distribution for this dimension
        cog_dist = Counter()
        for p in doc_paths:
            cog = manifests["classification"][p].get("cognitive_domain", "?")
            cog_dist[cog] += 1
        lines.append("## Cognitive Domain Distribution")
        lines.append("")
        for cog in ["factual_recall", "mechanistic_explanation",
                     "comparative_analysis", "quantitative_reasoning", "applied_clinical"]:
            c = cog_dist.get(cog, 0)
            bar = "#" * min(c, 40)
            lines.append(f"- {cog}: {c} {bar}")
        lines.append("")

        # Documents sorted by yield (highest first)
        lines.append("## Documents (by exam yield)")
        lines.append("")
        ranked = sorted(doc_paths, key=lambda p: dy_scores.get(p, {}).get("yield_score", 0), reverse=True)
        for p in ranked:
            entry = manifests["classification"][p]
            cog = entry.get("cognitive_domain", "?")
            schemas = ", ".join(entry.get("explanatory_schema", [])) or "none"
            dy = dy_scores.get(p, {})
            diff = dy.get("difficulty", 0)
            yld = dy.get("yield_score", 0)
            lines.append(f"- **{p}**")
            lines.append(f"  {cog} | schemas: {schemas} | "
                         f"difficulty: {diff:.1f} ({_difficulty_bar(diff)}) | "
                         f"yield: {yld:.1f} ({_yield_bar(yld)})")

            # Top QA facts
            if p in qa_facts:
                for qa in qa_facts[p][:2]:
                    q = qa.get("question", "")[:80]
                    a = qa["answers"][0]["text"][:120] if qa.get("answers") else ""
                    if a:
                        lines.append(f"  - Q: {q}")
                        lines.append(f"    A: {a}")

            # Top segment key facts
            if p in seg_facts:
                for seg in seg_facts[p][:2]:
                    for fact in seg.get("key_facts", [])[:1]:
                        a = fact.get("answer", "")[:120]
                        if a:
                            lines.append(f"  - Fact: {a}")
        lines.append("")

        # Co-occurring dimensions
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
                pct = 100 * c / len(doc_paths)
                lines.append(f"- [{d}](./{d}.md): {c} shared ({pct:.0f}%)")
            lines.append("")

        with open(dim_dir / f"{dim}.md", "w") as f:
            f.write("\n".join(lines))

    return list(dim_docs.keys())


def generate_schema_pages(manifests, qa_facts, dy_scores):
    """Generate a markdown page per explanatory schema."""
    schema_dir = OUTPUT_DIR / "schemas"
    schema_dir.mkdir(parents=True, exist_ok=True)

    schema_docs = defaultdict(list)
    for path, entry in manifests["classification"].items():
        for s in entry.get("explanatory_schema", []):
            schema_docs[s].append(path)

    for schema, doc_paths in schema_docs.items():
        lines = [f"# {schema.replace('_', ' ').title()}", ""]
        lines.append(f"**Documents:** {len(doc_paths)}")
        lines.append("")

        by_domain = defaultdict(list)
        for p in doc_paths:
            cog = manifests["classification"][p].get("cognitive_domain", "?")
            by_domain[cog].append(p)

        for domain in ["factual_recall", "mechanistic_explanation",
                        "comparative_analysis", "quantitative_reasoning",
                        "applied_clinical"]:
            if domain not in by_domain:
                continue
            # Sort by yield within each domain
            docs = sorted(
                by_domain[domain],
                key=lambda p: dy_scores.get(p, {}).get("yield_score", 0),
                reverse=True,
            )
            lines.append(f"## {domain.replace('_', ' ').title()} ({len(docs)})")
            lines.append("")
            for p in docs[:12]:
                dims = ", ".join(
                    manifests["classification"][p].get("content_dimensions", [])
                ) or "none"
                dy = dy_scores.get(p, {})
                yld = dy.get("yield_score", 0)
                lines.append(f"- **{p}** — dims: {dims} | yield: {yld:.1f}")
            lines.append("")

        with open(schema_dir / f"{schema}.md", "w") as f:
            f.write("\n".join(lines))

    return list(schema_docs.keys())


def generate_study_path_pages(study_paths):
    """Generate individual pages for each study path."""
    sp_dir = OUTPUT_DIR / "study_paths"
    sp_dir.mkdir(parents=True, exist_ok=True)

    for i, sp in enumerate(study_paths):
        dim = sp["dimension"]
        safe_name = re.sub(r"[^a-z0-9_]", "_", dim.lower())
        path_type = sp.get("path_type", "unknown")

        lines = [f"# Study Path: {dim.replace('_', ' ').title()}", ""]
        lines.append(f"**Type:** {path_type}")
        lines.append(f"**BFO Category:** {sp['bfo_category']}")
        lines.append(f"**Steps:** {sp['num_steps']}")
        lines.append(f"**Bloom's Span:** {sp.get('bloom_span', 0)} levels")
        lines.append(f"**Progression:** {' → '.join(sp.get('domains_covered', []))}")
        lines.append("")

        for j, step in enumerate(sp["steps"], 1):
            label = step.get("cognitive_label", step.get("cognitive_domain", "?"))
            lines.append(f"## Step {j}: {label}")
            lines.append(f"**Document:** {step['document']}")
            lines.append(f"**Clinical relevance:** {step.get('clinical_relevance', '?')}")
            schemas = step.get("schemas", step.get("dimensions", []))
            if schemas:
                lines.append(f"**Tags:** {', '.join(schemas)}")
            richness = step.get("richness_score", 0)
            if richness:
                lines.append(f"**Richness score:** {richness}")
            lines.append("")

        with open(sp_dir / f"{safe_name}_{path_type}.md", "w") as f:
            f.write("\n".join(lines))

    return len(study_paths)


def generate_crossref_graph(manifests):
    """Generate a crossref graph page with dimension overlap analysis."""
    lines = ["# Cross-Reference Knowledge Graph", ""]
    lines.append("Semantic links between documents from enrichment cross-references.")
    lines.append("")

    enrich_manifest = manifests["enrichment"]
    class_manifest = manifests["classification"]

    xref_count = 0
    same_dim_count = 0
    cross_dim_count = 0

    for path, info in sorted(enrich_manifest.items()):
        fpath = ENRICHMENT_DIR / info["output_file"]
        if not fpath.exists():
            continue
        with open(fpath) as f:
            doc = json.load(f)
        xrefs = doc.get("crossreferences", [])
        if not xrefs:
            continue

        src_dims = set(class_manifest.get(path, {}).get("content_dimensions", []))
        lines.append(f"## {path}")
        lines.append(f"Dimensions: {', '.join(sorted(src_dims)) or 'none'}")
        lines.append("")
        for xref in xrefs:
            target = xref.get("target", "")
            tgt_dims = set(class_manifest.get(target, {}).get("content_dimensions", []))
            shared = src_dims & tgt_dims
            if shared:
                same_dim_count += 1
                overlap = f"shared: {', '.join(sorted(shared))}"
            else:
                cross_dim_count += 1
                overlap = "no dimension overlap"
            lines.append(f"- → **{target}** ({overlap})")
            xref_count += 1
        lines.append("")

    lines.insert(2, f"Total cross-references: {xref_count}")
    lines.insert(3, f"Same-dimension links: {same_dim_count} | "
                    f"Cross-dimension links: {cross_dim_count}")
    lines.insert(4, "")

    with open(OUTPUT_DIR / "crossref_graph.md", "w") as f:
        f.write("\n".join(lines))

    return xref_count


def generate_high_yield_guide(manifests, qa_facts, dy_scores):
    """Generate a high-yield study guide ranked by exam relevance."""
    hy_dir = OUTPUT_DIR / "high_yield"
    hy_dir.mkdir(parents=True, exist_ok=True)

    class_manifest = manifests["classification"]

    # Rank all docs by yield score
    ranked = sorted(
        class_manifest.items(),
        key=lambda x: dy_scores.get(x[0], {}).get("yield_score", 0),
        reverse=True,
    )

    # Top 50 high-yield documents
    lines = ["# High-Yield Topics for PEX", ""]
    lines.append("Documents ranked by exam yield score (QA density, recency, cross-reference importance).")
    lines.append("")

    for rank, (path, entry) in enumerate(ranked[:50], 1):
        dy = dy_scores.get(path, {})
        diff = dy.get("difficulty", 0)
        yld = dy.get("yield_score", 0)
        cog = entry.get("cognitive_domain", "?")
        clin = entry.get("clinical_relevance", "?")
        dims = ", ".join(entry.get("content_dimensions", [])) or "none"
        schemas = ", ".join(entry.get("explanatory_schema", [])) or "none"

        lines.append(f"### {rank}. {path}")
        lines.append(f"- **Yield:** {yld:.1f}/4 ({_yield_bar(yld)}) | "
                     f"**Difficulty:** {diff:.1f}/4 ({_difficulty_bar(diff)})")
        lines.append(f"- **Cognitive:** {cog} | **Clinical:** {clin}")
        lines.append(f"- **Dimensions:** {dims}")
        lines.append(f"- **Schemas:** {schemas}")

        # Top facts
        if path in qa_facts:
            for qa in qa_facts[path][:3]:
                q = qa.get("question", "")[:80]
                a = qa["answers"][0]["text"][:150] if qa.get("answers") else ""
                if a:
                    lines.append(f"- Q: {q}")
                    lines.append(f"  A: {a}")
        lines.append("")

    with open(hy_dir / "top_50.md", "w") as f:
        f.write("\n".join(lines))

    # Group by subcategory for quick-reference guides
    subcat_docs = defaultdict(list)
    for path, entry in class_manifest.items():
        parts = path.split("/")
        subcat = "/".join(parts[:2]) if len(parts) >= 2 else parts[0]
        subcat_docs[subcat].append((path, entry))

    for subcat, docs in sorted(subcat_docs.items()):
        safe = re.sub(r"[^a-z0-9_]", "_", subcat.lower())
        docs_ranked = sorted(
            docs,
            key=lambda x: dy_scores.get(x[0], {}).get("yield_score", 0),
            reverse=True,
        )

        lines = [f"# {subcat.replace('/', ' / ').title()}", ""]
        lines.append(f"Documents: {len(docs_ranked)}")
        lines.append("")

        for path, entry in docs_ranked:
            dy = dy_scores.get(path, {})
            diff = dy.get("difficulty", 0)
            yld = dy.get("yield_score", 0)
            cog = entry.get("cognitive_domain", "?")
            dims = ", ".join(entry.get("content_dimensions", [])) or "none"
            lines.append(f"- **{path.split('/')[-1]}** — "
                        f"{cog} | yield: {yld:.1f} | diff: {diff:.1f} | dims: {dims}")
        lines.append("")

        with open(hy_dir / f"{safe}.md", "w") as f:
            f.write("\n".join(lines))

    return len(ranked)


def generate_analytics(manifests, dy_scores):
    """Generate a corpus analytics overview page."""
    class_manifest = manifests["classification"]

    lines = ["# PEX/SAQ Corpus Analytics", ""]

    # Cognitive domain distribution
    cog_dist = Counter()
    for entry in class_manifest.values():
        cog_dist[entry.get("cognitive_domain", "?")] += 1

    lines.append("## Cognitive Domain Distribution")
    lines.append("")
    lines.append("| Domain | Count | % |")
    lines.append("|--------|-------|---|")
    for domain in ["factual_recall", "mechanistic_explanation",
                    "comparative_analysis", "quantitative_reasoning", "applied_clinical"]:
        c = cog_dist.get(domain, 0)
        pct = 100 * c / len(class_manifest) if class_manifest else 0
        lines.append(f"| {domain} | {c} | {pct:.0f}% |")
    lines.append("")

    # Clinical relevance distribution
    clin_dist = Counter()
    for entry in class_manifest.values():
        clin_dist[entry.get("clinical_relevance", "?")] += 1

    lines.append("## Clinical Relevance Distribution")
    lines.append("")
    lines.append("| Tier | Count | % |")
    lines.append("|------|-------|---|")
    for tier in ["immediately_applicable", "integrative", "foundational_science"]:
        c = clin_dist.get(tier, 0)
        pct = 100 * c / len(class_manifest) if class_manifest else 0
        lines.append(f"| {tier} | {c} | {pct:.0f}% |")
    lines.append("")

    # Dimension distribution
    dim_dist = Counter()
    for entry in class_manifest.values():
        for d in entry.get("content_dimensions", []):
            dim_dist[d] += 1

    lines.append("## Content Dimension Distribution")
    lines.append("")
    lines.append("| Dimension | Docs | % | BFO |")
    lines.append("|-----------|------|---|-----|")
    for dim, c in dim_dist.most_common():
        pct = 100 * c / len(class_manifest) if class_manifest else 0
        lines.append(f"| {dim} | {c} | {pct:.0f}% | {_bfo_cat(dim)} |")
    lines.append("")

    # Dimension count distribution
    ndim_dist = Counter()
    for entry in class_manifest.values():
        ndim_dist[entry.get("num_active_dimensions", 0)] += 1

    lines.append("## Dimensions per Document")
    lines.append("")
    for k in sorted(ndim_dist.keys()):
        bar = "#" * min(ndim_dist[k], 50)
        lines.append(f"- {k} dims: {ndim_dist[k]} docs {bar}")
    lines.append("")

    # Difficulty/yield summary
    if dy_scores:
        difficulties = [v["difficulty"] for v in dy_scores.values()]
        yields = [v["yield_score"] for v in dy_scores.values()]
        lines.append("## Difficulty & Yield Summary")
        lines.append("")
        lines.append(f"- **Difficulty:** mean={sum(difficulties)/len(difficulties):.2f}, "
                    f"min={min(difficulties):.2f}, max={max(difficulties):.2f}")
        lines.append(f"- **Yield:** mean={sum(yields)/len(yields):.2f}, "
                    f"min={min(yields):.2f}, max={max(yields):.2f}")
        lines.append("")

        # Difficulty histogram
        diff_buckets = Counter()
        for d in difficulties:
            if d < 1.5:
                diff_buckets["Easy (<1.5)"] += 1
            elif d < 2.5:
                diff_buckets["Medium (1.5-2.5)"] += 1
            elif d < 3.2:
                diff_buckets["Hard (2.5-3.2)"] += 1
            else:
                diff_buckets["Very Hard (>3.2)"] += 1

        lines.append("### Difficulty Histogram")
        lines.append("")
        for label in ["Easy (<1.5)", "Medium (1.5-2.5)", "Hard (2.5-3.2)", "Very Hard (>3.2)"]:
            c = diff_buckets.get(label, 0)
            bar = "#" * min(c, 50)
            lines.append(f"- {label}: {c} {bar}")
        lines.append("")

    # Schema distribution
    schema_dist = Counter()
    for entry in class_manifest.values():
        for s in entry.get("explanatory_schema", []):
            schema_dist[s] += 1

    lines.append("## Explanatory Schema Distribution")
    lines.append("")
    for s, c in schema_dist.most_common():
        pct = 100 * c / len(class_manifest) if class_manifest else 0
        lines.append(f"- {s}: {c} ({pct:.0f}%)")
    lines.append("")

    # Exam year distribution
    year_dist = Counter()
    for path in class_manifest:
        m = re.search(r"(\d{4})[AB]", path)
        if m:
            year_dist[int(m.group(1))] += 1

    lines.append("## Exam Year Distribution")
    lines.append("")
    for year in sorted(year_dist.keys()):
        c = year_dist[year]
        bar = "#" * min(c, 40)
        lines.append(f"- {year}: {c} {bar}")
    lines.append("")

    with open(OUTPUT_DIR / "analytics.md", "w") as f:
        f.write("\n".join(lines))


def generate_overview(manifests, dimensions, schemas, xref_count, study_paths, dy_scores):
    """Generate main index page."""
    n_class = len(manifests["classification"])
    n_enrich = len(manifests["enrichment"])
    n_qa = len(manifests["qa"])
    n_seg = len(manifests["segment"])

    lines = [
        "# PEX/SAQ Semantic Knowledge Index",
        "",
        f"**Corpus:** {n_enrich} enriched, {n_class} classified, {n_qa} QA extracted, {n_seg} segment-processed",
        "",
    ]

    # Difficulty/yield summary
    if dy_scores:
        difficulties = [v["difficulty"] for v in dy_scores.values()]
        yields = [v["yield_score"] for v in dy_scores.values()]
        lines.append(f"**Avg Difficulty:** {sum(difficulties)/len(difficulties):.1f}/4 | "
                    f"**Avg Yield:** {sum(yields)/len(yields):.1f}/4")
        lines.append("")

    lines.append("## Content Dimensions (BFO-aligned)")
    lines.append("")
    for dim in sorted(dimensions):
        count = sum(
            1 for e in manifests["classification"].values()
            if dim in e.get("content_dimensions", [])
        )
        pct = 100 * count / n_class if n_class else 0
        lines.append(f"- [{dim}](dimensions/{dim}.md) ({count} docs, {pct:.0f}%) — {_bfo_cat(dim)}")
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

    lines.append(f"## [Study Paths](study_paths/) ({len(study_paths)} paths)")
    lines.append("Bloom's-taxonomy cognitive progressions through each dimension and schema.")
    lines.append("")

    # Summarise top study paths
    for sp in study_paths[:8]:
        dim = sp["dimension"]
        bloom = sp.get("bloom_span", 0)
        ptype = sp.get("path_type", "?")
        domains = " → ".join(sp.get("domains_covered", []))
        lines.append(f"- **{dim}** ({ptype}): {sp['num_steps']} steps, "
                    f"Bloom span {bloom} — {domains}")
    lines.append("")

    lines.append(f"## [Cross-Reference Graph](crossref_graph.md)")
    lines.append(f"{xref_count} semantic links between documents")
    lines.append("")

    lines.append("## [High-Yield Study Guide](high_yield/top_50.md)")
    lines.append("Top 50 documents ranked by exam relevance")
    lines.append("")

    lines.append("## [Corpus Analytics](analytics.md)")
    lines.append("Distribution histograms for all classification axes")
    lines.append("")

    with open(OUTPUT_DIR / "INDEX.md", "w") as f:
        f.write("\n".join(lines))


def main():
    print("Loading pipeline manifests...")
    manifests = load_manifests()
    for name, m in manifests.items():
        print(f"  {name}: {len(m)} entries")

    print("\nLoading QA facts...")
    qa_facts = load_qa_facts(manifests)
    print(f"  {len(qa_facts)} docs with QA facts")

    print("Loading segment facts...")
    seg_facts = load_segment_facts(manifests)
    print(f"  {len(seg_facts)} docs with segment facts")

    print("Loading difficulty/yield scores...")
    dy_scores = load_difficulty_yield()
    print(f"  {len(dy_scores)} docs scored")

    print("Loading study paths...")
    study_paths = load_study_paths()
    print(f"  {len(study_paths)} study paths")

    print("\nGenerating dimension pages...")
    dimensions = generate_dimension_pages(manifests, qa_facts, seg_facts, dy_scores)
    print(f"  {len(dimensions)} dimension pages")

    print("Generating schema pages...")
    schemas = generate_schema_pages(manifests, qa_facts, dy_scores)
    print(f"  {len(schemas)} schema pages")

    print("Generating study path pages...")
    n_sp = generate_study_path_pages(study_paths)
    print(f"  {n_sp} study path pages")

    print("Generating cross-reference graph...")
    xref_count = generate_crossref_graph(manifests)
    print(f"  {xref_count} cross-references")

    print("Generating high-yield guide...")
    n_ranked = generate_high_yield_guide(manifests, qa_facts, dy_scores)
    print(f"  {n_ranked} documents ranked")

    print("Generating corpus analytics...")
    generate_analytics(manifests, dy_scores)

    print("Generating overview index...")
    generate_overview(manifests, dimensions, schemas, xref_count, study_paths, dy_scores)

    print(f"\nSemantic index generated: {OUTPUT_DIR}")
    print(f"  {len(dimensions)} dimensions, {len(schemas)} schemas, "
          f"{n_sp} study paths, {xref_count} crossrefs")


if __name__ == "__main__":
    main()
