#!/usr/bin/env python3
"""Enrich PEX/SAQ .htm documents using the Isaacus Kanon 2 Enricher API."""

import json
import os
import sys
import time
from pathlib import Path

from bs4 import BeautifulSoup
from isaacus import Isaacus

API_KEY = "iuak_v1_alcvXayhjV_8O92bz8S8kphmSOBpiNNoDDAQoexh6q1_feeb2b85"
BATCH_SIZE = 8
OUTPUT_DIR = Path(__file__).parent / "enrichment_results"
SAQS_DIR = Path(__file__).parent


def extract_text_from_html(filepath: Path) -> str:
    """Extract plain text from an HTML file."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n", strip=True)


def collect_htm_files(base_dir: Path) -> list[Path]:
    """Collect all .htm files recursively from base_dir."""
    return sorted(base_dir.rglob("*.htm"))


def enrich_batch(client: Isaacus, texts: list[str]) -> list:
    """Enrich a batch of texts using Kanon 2 Enricher."""
    response = client.enrichments.create(
        model="kanon-2-enricher",
        texts=texts,
        overflow_strategy="auto",
    )
    return response


def serialize_document(doc) -> dict:
    """Convert an ILGS document to a JSON-serializable dict."""
    return doc.model_dump(mode="json")


def main():
    client = Isaacus(api_key=API_KEY)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    htm_files = collect_htm_files(SAQS_DIR)
    # Exclude files inside enrichment_results
    htm_files = [f for f in htm_files if "enrichment_results" not in str(f)]
    print(f"Found {len(htm_files)} .htm files to enrich")

    # Track progress via a manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {}

    # Filter out already-processed files
    remaining = [f for f in htm_files if str(f.relative_to(SAQS_DIR)) not in manifest]
    print(f"{len(remaining)} files remaining to process")

    total_input_tokens = 0
    processed = 0

    for i in range(0, len(remaining), BATCH_SIZE):
        batch_files = remaining[i : i + BATCH_SIZE]
        batch_texts = []
        batch_rel_paths = []

        for filepath in batch_files:
            text = extract_text_from_html(filepath)
            if not text.strip():
                print(f"  Skipping empty file: {filepath}")
                continue
            batch_texts.append(text)
            batch_rel_paths.append(str(filepath.relative_to(SAQS_DIR)))

        if not batch_texts:
            continue

        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\nBatch {batch_num}/{total_batches}: enriching {len(batch_texts)} documents...")

        try:
            response = enrich_batch(client, batch_texts)
        except Exception as e:
            print(f"  Error enriching batch: {e}")
            # Retry once after a pause
            time.sleep(5)
            try:
                response = enrich_batch(client, batch_texts)
            except Exception as e2:
                print(f"  Retry also failed: {e2}. Skipping batch.")
                continue

        total_input_tokens += response.usage.input_tokens

        for result in response.results:
            rel_path = batch_rel_paths[result.index]
            doc = result.document

            # Save enriched document as JSON
            output_path = OUTPUT_DIR / (rel_path.replace("/", "__").replace(".htm", ".json"))
            doc_dict = serialize_document(doc)
            with open(output_path, "w") as f:
                json.dump(doc_dict, f, indent=2, ensure_ascii=False)

            manifest[rel_path] = {
                "output_file": str(output_path.name),
                "title": doc.title and doc.text[doc.title.start : doc.title.end] or None,
                "type": doc.type,
                "jurisdiction": doc.jurisdiction,
                "num_segments": len(doc.segments),
                "num_persons": len(doc.persons),
                "num_locations": len(doc.locations),
                "num_terms": len(doc.terms),
            }
            processed += 1

        # Save manifest after each batch
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"  Processed {processed}/{len(remaining)} files. Tokens used so far: {total_input_tokens}")

    print(f"\nDone! Enriched {processed} documents. Total input tokens: {total_input_tokens}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
