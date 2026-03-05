#!/usr/bin/env python3
"""
Step 5: Segment-level fact extraction from the PEX/SAQ corpus.

Extracts structured key facts from ~10,000 segments using extractive QA.
Concept tagging is derived from Step 2 document-level classifications
(which already cover the same dimensions), avoiding redundant API calls.

Produces:
  1. Segment-level key facts — definitions, values, and mechanisms
  2. Concept index — derived from Step 2 classifications mapped to segments
  3. Concept co-occurrence matrix — derived from Step 2 data

Pareto-optimised: 2 QA calls per segment (was 11 with classifications).
Saves ~13M tokens and ~6h vs. the classification-heavy approach.

Order of operations: Run AFTER enrich.py and classify.py.
"""

import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

from isaacus import Isaacus

from text_utils import DIMENSION_THRESHOLD, SKIP_PATHS

API_KEY = os.environ.get(
    "ISAACUS_API_KEY",
    "iuak_v1_alcvXayhjV_8O92bz8S8kphmSOBpiNNoDDAQoexh6q1_feeb2b85",
)
OUTPUT_DIR = Path(__file__).parent / "segment_results"
ENRICHMENT_DIR = Path(__file__).parent / "enrichment_results"
CLASSIFICATION_DIR = Path(__file__).parent / "classification_results"
MANIFEST_PATH = ENRICHMENT_DIR / "manifest.json"

# Minimum segment text length worth processing
MIN_SEGMENT_LENGTH = 50
MAX_RETRIES = 4
MANIFEST_FLUSH_INTERVAL = 10

# Focused fact extraction questions (high unique value per API call)
SEGMENT_QUESTIONS = [
    "What is the key fact or definition stated here?",
    "What numerical value or normal range is given?",
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


def load_document_classifications(path: str) -> dict:
    """Load Step 2 classification summary for a document.

    Returns the active content dimensions from the document-level ontology,
    which are inherited by all segments in that document.
    """
    class_manifest_path = CLASSIFICATION_DIR / "manifest.json"
    if not class_manifest_path.exists():
        return {"content_dimensions": []}

    with open(class_manifest_path) as f:
        class_manifest = json.load(f)

    entry = class_manifest.get(path)
    if not entry:
        return {"content_dimensions": []}

    return {
        "content_dimensions": entry.get("content_dimensions", []),
        "cognitive_domain": entry.get("cognitive_domain"),
        "clinical_relevance": entry.get("clinical_relevance"),
    }


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
        if path not in seg_manifest and path not in SKIP_PATHS
    ]
    print(f"Segment fact extraction: {len(remaining)} documents remaining")
    print(f"  {len(SEGMENT_QUESTIONS)} questions per segment "
          f"(classification inherited from Step 2)")

    # Concept index: built from Step 2 classifications mapped to segments
    concept_index = defaultdict(list)

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

        # Inherit document-level classifications from Step 2
        doc_class = load_document_classifications(path)
        doc_dimensions = doc_class.get("content_dimensions", [])

        print(f"\n[{processed + 1}/{len(remaining)}] {path} "
              f"({len(segments)} segs, dims: {doc_dimensions})")

        segment_results = []
        for s_idx, seg in enumerate(segments):
            if s_idx % 10 == 0 and s_idx > 0:
                print(f"  Segment {s_idx}/{len(segments)}...")

            # Extract facts (the unique high-value work)
            facts = extract_segment_facts(client, seg["text"])

            # Map document dimensions to this segment for the concept index
            for dim in doc_dimensions:
                concept_index[dim].append({
                    "source": path,
                    "segment_id": seg["id"],
                    "text_preview": seg["text"][:150],
                })

            segment_results.append({
                "id": seg["id"],
                "start": seg["start"],
                "end": seg["end"],
                "level": seg["level"],
                "text_preview": seg["text"][:100],
                "inherited_dimensions": doc_dimensions,
                "key_facts": facts,
            })

        # Save document segment results
        output_name = info["output_file"].replace(".json", "_segments.json")
        result_doc = {
            "source": path,
            "num_segments_processed": len(segment_results),
            "document_classification": doc_class,
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

        processed += 1
        facts_count = sum(len(s["key_facts"]) for s in segment_results)
        print(f"  -> {len(segment_results)} segments, {facts_count} facts extracted")

    # Save concept index
    concept_index_path = OUTPUT_DIR / "concept_index.json"
    with open(concept_index_path, "w") as f:
        json.dump(dict(concept_index), f, indent=2, ensure_ascii=False)

    # --- Build co-occurrence matrix from Step 2 classifications ---
    print("\nBuilding concept co-occurrence matrix from Step 2 classifications...")
    class_manifest_path = CLASSIFICATION_DIR / "manifest.json"
    cooccurrence = defaultdict(Counter)

    if class_manifest_path.exists():
        with open(class_manifest_path) as f:
            class_manifest = json.load(f)
        for entry in class_manifest.values():
            dims = entry.get("content_dimensions", [])
            for i, d1 in enumerate(dims):
                for d2 in dims[i + 1:]:
                    cooccurrence[d1][d2] += 1
                    cooccurrence[d2][d1] += 1

    cooccurrence_matrix = {
        concept: dict(counts) for concept, counts in cooccurrence.items()
    }
    with open(OUTPUT_DIR / "concept_cooccurrence.json", "w") as f:
        json.dump(cooccurrence_matrix, f, indent=2, ensure_ascii=False)

    # --- Build global statistics ---
    total_segments = sum(
        info["num_segments_processed"] for info in seg_manifest.values()
    )
    stats = {
        "total_documents_processed": processed,
        "total_segments_processed": total_segments,
        "questions_per_segment": len(SEGMENT_QUESTIONS),
        "api_calls_per_segment": len(SEGMENT_QUESTIONS),
        "concept_index_source": "Step 2 document-level classifications (inherited)",
        "concept_segment_counts": {
            k: len(v) for k, v in concept_index.items()
        },
    }
    with open(OUTPUT_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\nSegment processing complete! {processed} documents.")
    print(f"Total segments: {total_segments}")
    print(f"Concept index dimensions: {len(concept_index)}")
    print(f"Co-occurrence pairs: {sum(len(v) for v in cooccurrence.values()) // 2}")
    print(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
