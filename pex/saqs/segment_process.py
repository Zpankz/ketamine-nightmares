#!/usr/bin/env python3
"""
Step 5: Segment-level deep processing of the PEX/SAQ corpus.

Operates on the ~20,000 segments produced by enrichment (Step 0) and
applies classification + extractive QA at segment granularity to build
a fine-grained knowledge graph.

Produces:
  1. Segment classifications — each segment tagged with content dimensions
  2. Cross-document concept index — terms/concepts mapped to all segments
     where they appear, with relevance scores
  3. Segment-level QA — key facts extracted from individual segments
  4. Concept co-occurrence matrix — which concepts appear together

This is the most token-intensive step. Run it last.

Order of operations: Run AFTER enrich.py, classify.py, and embed.py.
"""

import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

from isaacus import Isaacus

API_KEY = os.environ.get(
    "ISAACUS_API_KEY",
    "iuak_v1_alcvXayhjV_8O92bz8S8kphmSOBpiNNoDDAQoexh6q1_feeb2b85",
)
OUTPUT_DIR = Path(__file__).parent / "segment_results"
ENRICHMENT_DIR = Path(__file__).parent / "enrichment_results"
MANIFEST_PATH = ENRICHMENT_DIR / "manifest.json"

# Minimum segment text length worth processing
MIN_SEGMENT_LENGTH = 50
MAX_RETRIES = 4
MANIFEST_FLUSH_INTERVAL = 10

# Core concept queries for segment-level classification.
# Deliberately fewer than the document-level ontology (Step 2) to manage
# token costs — only the dimensions most useful at segment granularity.
SEGMENT_CONCEPTS = {
    "mechanism_of_action": {
        "query": (
            "{This describes a specific mechanism of action, receptor "
            "interaction, or signal transduction pathway}"
        ),
        "is_iql": True,
    },
    "clinical_effect": {
        "query": (
            "{This describes a specific clinical effect, side effect, "
            "or physiological response to a drug or intervention}"
        ),
        "is_iql": True,
    },
    "pharmacokinetics": {
        "query": (
            "{This describes pharmacokinetic properties such as absorption, "
            "distribution, metabolism, excretion, half-life, or bioavailability}"
        ),
        "is_iql": True,
    },
    "quantitative_data": {
        "query": (
            "{This contains specific numerical values, normal ranges, "
            "equations, percentages, or dose information}"
        ),
        "is_iql": True,
    },
    "comparison": {
        "query": (
            "{This compares or contrasts two or more drugs, techniques, "
            "or physiological processes}"
        ),
        "is_iql": True,
    },
    "physiology_regulation": {
        "query": (
            "{This describes physiological regulation, homeostatic "
            "mechanisms, feedback loops, or control systems}"
        ),
        "is_iql": True,
    },
    "anatomy_structure": {
        "query": (
            "{This describes anatomical structures, spatial relationships, "
            "or physical features of organs or tissues}"
        ),
        "is_iql": True,
    },
    "equipment_physics": {
        "query": (
            "{This describes equipment function, physical principles, "
            "measurement techniques, or engineering specifications}"
        ),
        "is_iql": True,
    },
}

# Segment-level questions for key-fact extraction
SEGMENT_QUESTIONS = [
    "What is the key fact or definition stated here?",
    "What numerical value or normal range is given?",
    "What mechanism or process is described?",
]


def extract_processable_segments(doc: dict) -> list[dict]:
    """Extract segments worth deep processing."""
    text = doc["text"]
    segments = []
    for seg in doc["segments"]:
        if seg["category"] != "main":
            continue
        if seg["kind"] != "item":
            continue
        start, end = seg["span"]["start"], seg["span"]["end"]
        seg_text = text[start:end].strip()
        if len(seg_text) < MIN_SEGMENT_LENGTH:
            continue
        segments.append({
            "id": seg["id"],
            "text": seg_text,
            "start": start,
            "end": end,
            "level": seg["level"],
        })
    return segments


def classify_segment(client: Isaacus, text: str) -> dict:
    """Classify a segment against all concept queries."""
    results = {}
    for concept, config in SEGMENT_CONCEPTS.items():
        retries = 0
        score = None
        while retries < MAX_RETRIES:
            try:
                response = client.classifications.universal.create(
                    model="kanon-universal-classifier-mini",
                    query=config["query"],
                    texts=[text],
                    is_iql=config["is_iql"],
                    scoring_method="chunk_max",
                )
                score = response.classifications[0].score
                break
            except Exception as e:
                retries += 1
                if retries < MAX_RETRIES:
                    print(f"    Retry {retries}/{MAX_RETRIES} for {concept}: {e}")
                    time.sleep(2 ** retries)
                else:
                    print(f"    classify_segment failed on {concept}: {e}")
        results[concept] = score
    return results


def extract_segment_facts(
    client: Isaacus, text: str
) -> list[dict]:
    """Extract key facts from a single segment."""
    facts = []
    for question in SEGMENT_QUESTIONS:
        retries = 0
        while retries < MAX_RETRIES:
            try:
                response = client.extractions.qa.create(
                    model="kanon-answer-extractor-mini",
                    query=question,
                    texts=[text],
                    top_k=1,
                    ignore_inextractability=False,
                )
                extraction = response.extractions[0]
                if (
                    extraction.answers
                    and extraction.answers[0].score > extraction.inextractability_score
                ):
                    facts.append({
                        "question": question,
                        "answer": extraction.answers[0].text,
                        "score": extraction.answers[0].score,
                    })
                break
            except Exception as e:
                retries += 1
                if retries < MAX_RETRIES:
                    print(f"    Retry {retries}/{MAX_RETRIES} for extract: {e}")
                    time.sleep(2 ** retries)
                else:
                    print(f"    extract_segment_facts failed on '{question[:40]}': {e}")
    return facts


def main():
    client = Isaacus(api_key=API_KEY)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    # Load progress
    seg_manifest_path = OUTPUT_DIR / "manifest.json"
    if seg_manifest_path.exists():
        with open(seg_manifest_path) as f:
            seg_manifest = json.load(f)
    else:
        seg_manifest = {}

    remaining = [
        (path, info)
        for path, info in manifest.items()
        if path not in seg_manifest
    ]
    print(f"Segment processing pipeline: {len(remaining)} documents remaining")

    # Global concept index: concept -> [{source, segment_id, score, text_preview}]
    concept_index_path = OUTPUT_DIR / "concept_index.json"
    if concept_index_path.exists():
        with open(concept_index_path) as f:
            concept_index = json.load(f)
    else:
        concept_index = {c: [] for c in SEGMENT_CONCEPTS}

    processed = 0

    for path, info in remaining:
        with open(ENRICHMENT_DIR / info["output_file"]) as f:
            doc = json.load(f)

        segments = extract_processable_segments(doc)
        if not segments:
            seg_manifest[path] = {
                "output_file": None,
                "num_segments_processed": 0,
            }
            continue

        print(f"\n[{processed + 1}/{len(remaining)}] {path} ({len(segments)} segments)")

        segment_results = []
        for s_idx, seg in enumerate(segments):
            if s_idx % 10 == 0 and s_idx > 0:
                print(f"  Segment {s_idx}/{len(segments)}...")

            # Classify
            concept_scores = classify_segment(client, seg["text"])

            # Extract facts
            facts = extract_segment_facts(client, seg["text"])

            # Update concept index for concepts scoring > 0.5
            for concept, score in concept_scores.items():
                if score and score > 0.5:
                    concept_index[concept].append({
                        "source": path,
                        "segment_id": seg["id"],
                        "score": round(score, 4),
                        "text_preview": seg["text"][:150],
                    })

            segment_results.append({
                "id": seg["id"],
                "start": seg["start"],
                "end": seg["end"],
                "level": seg["level"],
                "text_preview": seg["text"][:100],
                "concept_scores": {
                    k: round(v, 4) if v else None
                    for k, v in concept_scores.items()
                },
                "active_concepts": [
                    k for k, v in concept_scores.items() if v and v > 0.5
                ],
                "key_facts": facts,
            })

        # Save document segment results
        output_name = info["output_file"].replace(".json", "_segments.json")
        result_doc = {
            "source": path,
            "num_segments_processed": len(segment_results),
            "segments": segment_results,
        }
        with open(OUTPUT_DIR / output_name, "w") as f:
            json.dump(result_doc, f, indent=2, ensure_ascii=False)

        seg_manifest[path] = {
            "output_file": output_name,
            "num_segments_processed": len(segment_results),
        }

        # Save manifests periodically and on last document
        if processed % MANIFEST_FLUSH_INTERVAL == 0 or processed == len(remaining):
            with open(seg_manifest_path, "w") as f:
                json.dump(seg_manifest, f, indent=2, ensure_ascii=False)
            with open(concept_index_path, "w") as f:
                json.dump(concept_index, f, indent=2, ensure_ascii=False)

        processed += 1
        print(f"  -> {len(segment_results)} segments processed, "
              f"{sum(1 for s in segment_results for _ in s['active_concepts'])} concept tags")

    # --- Build co-occurrence matrix ---
    print("\nBuilding concept co-occurrence matrix...")
    cooccurrence = defaultdict(Counter)

    for path, info in seg_manifest.items():
        if not info.get("output_file"):
            continue
        with open(OUTPUT_DIR / info["output_file"]) as f:
            doc = json.load(f)
        for seg in doc["segments"]:
            concepts = seg["active_concepts"]
            for i, c1 in enumerate(concepts):
                for c2 in concepts[i + 1 :]:
                    cooccurrence[c1][c2] += 1
                    cooccurrence[c2][c1] += 1

    # Convert to serialisable format
    cooccurrence_matrix = {
        concept: dict(counts) for concept, counts in cooccurrence.items()
    }
    with open(OUTPUT_DIR / "concept_cooccurrence.json", "w") as f:
        json.dump(cooccurrence_matrix, f, indent=2, ensure_ascii=False)

    # --- Build global statistics ---
    concept_counts = {
        concept: len(entries) for concept, entries in concept_index.items()
    }
    stats = {
        "total_documents_processed": processed,
        "total_segments_processed": sum(
            info["num_segments_processed"]
            for info in seg_manifest.values()
        ),
        "concept_segment_counts": concept_counts,
        "concept_definitions": {
            k: v["query"] for k, v in SEGMENT_CONCEPTS.items()
        },
    }
    with open(OUTPUT_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\nSegment processing complete! {processed} documents.")
    print(f"Concept index entries: {sum(concept_counts.values())}")
    print(f"Co-occurrence pairs: {sum(len(v) for v in cooccurrence.values()) // 2}")
    print(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
