#!/usr/bin/env python3
"""
Incremental reclassification: only re-runs changed or new ontology queries.

Compares the current ONTOLOGY in classify.py against the saved ontology.json
to determine which queries changed. Then re-runs only those queries across
all documents, patching the existing classification results in-place.

This avoids re-running all 24 queries when only a few changed.
"""

import json
import os
import time
from pathlib import Path

from isaacus import Isaacus

from classify import ONTOLOGY, classify_document, flatten_ontology
from text_utils import DIMENSION_THRESHOLD, SKIP_PATHS, strip_footer

API_KEY = os.environ.get(
    "ISAACUS_API_KEY",
    "iuak_v1_alcvXayhjV_8O92bz8S8kphmSOBpiNNoDDAQoexh6q1_feeb2b85",
)
OUTPUT_DIR = Path(__file__).parent / "classification_results"
ENRICHMENT_DIR = Path(__file__).parent / "enrichment_results"


def find_changed_queries():
    """Compare current ontology with saved one to find changed/new queries."""
    old_path = OUTPUT_DIR / "ontology.json"
    if not old_path.exists():
        print("No saved ontology — full reclassification needed.")
        return flatten_ontology()

    with open(old_path) as f:
        old_ontology = json.load(f)

    changed = []
    for axis, labels in ONTOLOGY.items():
        old_axis = old_ontology.get(axis, {})
        for label, config in labels.items():
            old_config = old_axis.get(label, {})
            if config["query"] != old_config.get("query", ""):
                changed.append((axis, label, config))
                print(f"  CHANGED: {axis}/{label}")
            elif label not in old_axis:
                changed.append((axis, label, config))
                print(f"  NEW: {axis}/{label}")

    return changed


def main():
    print("Detecting ontology changes...")
    changed = find_changed_queries()
    if not changed:
        print("No changes detected. Nothing to reclassify.")
        return

    print(f"\n{len(changed)} queries to re-run across all documents.\n")

    client = Isaacus(api_key=API_KEY)

    # Load enrichment manifest
    with open(ENRICHMENT_DIR / "manifest.json") as f:
        enrich_manifest = json.load(f)

    # Load classification manifest
    class_manifest_path = OUTPUT_DIR / "manifest.json"
    with open(class_manifest_path) as f:
        class_manifest = json.load(f)

    # Determine which axes contain changed queries
    changed_axes = {axis for axis, _, _ in changed}
    multi_axes = {"content_dimensions", "explanatory_schema"}
    single_axes = {"cognitive_domain", "clinical_relevance"}

    total_calls = len(enrich_manifest) * len(changed)
    print(f"Total API calls: {total_calls}")
    total_tokens = 0
    processed = 0

    for path, info in enrich_manifest.items():
        if path in SKIP_PATHS:
            continue

        with open(ENRICHMENT_DIR / info["output_file"]) as f:
            doc = json.load(f)

        text = strip_footer(doc["text"])
        if not text:
            continue

        processed += 1
        if processed % 50 == 1:
            print(f"\n[{processed}/{len(enrich_manifest)}] Processing...")

        # Load existing classification result
        class_entry = class_manifest.get(path)
        if not class_entry:
            continue
        class_file = OUTPUT_DIR / class_entry.get("output_file", "")
        if not class_file.exists():
            continue
        with open(class_file) as f:
            class_doc = json.load(f)

        # Re-run only changed queries
        for axis, label, config in changed:
            retries = 0
            while retries < 4:
                try:
                    result = classify_document(
                        client, text, config["query"], config["is_iql"]
                    )
                    break
                except Exception as e:
                    retries += 1
                    wait = 2 ** retries
                    print(f"  Error on {axis}/{label}: {e}. Retry {retries}/4 in {wait}s...")
                    time.sleep(wait)
            else:
                result = {"score": None, "top_chunks": [], "input_tokens": 0, "error": "failed"}

            # Patch the classification result
            if axis not in class_doc["classifications"]:
                class_doc["classifications"][axis] = {}
            class_doc["classifications"][axis][label] = {
                "score": result["score"],
                "description": config["description"],
                "top_chunks": result["top_chunks"],
            }
            total_tokens += result.get("input_tokens", 0)

        # Recompute summaries for affected axes
        for axis in changed_axes:
            axis_scores = class_doc["classifications"].get(axis, {})
            if axis in single_axes:
                best = max(axis_scores.items(), key=lambda x: x[1]["score"] or 0)
                class_doc["summary"][axis] = {
                    "primary": best[0],
                    "score": best[1]["score"],
                }
                if best[1]["score"]:
                    secondaries = [
                        (k, v["score"])
                        for k, v in axis_scores.items()
                        if k != best[0] and v["score"] and v["score"] >= best[1]["score"] - 0.1
                    ]
                    if secondaries:
                        class_doc["summary"][axis]["secondary"] = [s[0] for s in secondaries]
            elif axis in multi_axes:
                active = [
                    (k, v["score"])
                    for k, v in axis_scores.items()
                    if v["score"] and v["score"] > DIMENSION_THRESHOLD
                ]
                active.sort(key=lambda x: x[1], reverse=True)
                class_doc["summary"][axis] = [d[0] for d in active]

        # Save patched result
        with open(class_file, "w") as f:
            json.dump(class_doc, f, indent=2, ensure_ascii=False)

        # Update manifest entry
        class_manifest[path] = {
            "output_file": class_entry["output_file"],
            "cognitive_domain": class_doc["summary"]["cognitive_domain"]["primary"],
            "clinical_relevance": class_doc["summary"]["clinical_relevance"]["primary"],
            "content_dimensions": class_doc["summary"]["content_dimensions"],
            "explanatory_schema": class_doc["summary"]["explanatory_schema"],
            "num_active_dimensions": len(class_doc["summary"]["content_dimensions"]),
        }

    # Save updated manifest
    with open(class_manifest_path, "w") as f:
        json.dump(class_manifest, f, indent=2, ensure_ascii=False)

    # Save updated ontology
    with open(OUTPUT_DIR / "ontology.json", "w") as f:
        json.dump(ONTOLOGY, f, indent=2, ensure_ascii=False)

    print(f"\nDelta reclassification complete!")
    print(f"  {processed} documents, {len(changed)} queries each")
    print(f"  Total tokens: {total_tokens}")

    # Show new dimension stats
    from collections import Counter
    dim_counts = Counter()
    for entry in class_manifest.values():
        for d in entry.get("content_dimensions", []):
            dim_counts[d] += 1
    print(f"\nUpdated dimension distribution:")
    for d, c in dim_counts.most_common():
        print(f"  {d}: {c} ({100*c/len(class_manifest):.0f}%)")

    ndim = Counter(e.get("num_active_dimensions", 0) for e in class_manifest.values())
    zero_dim = ndim.get(0, 0)
    print(f"\nDocs with 0 dimensions: {zero_dim} (was 50)")


if __name__ == "__main__":
    main()
