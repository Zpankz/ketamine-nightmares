#!/usr/bin/env python3
"""
Step 1: Embed PEX/SAQ documents and segments using Isaacus Kanon 2 Embedder.

Produces dense vector embeddings at two granularities:
  - Document-level: full text embedded as retrieval/document
  - Segment-level: each main-body segment embedded individually

These enable semantic search, clustering, similarity detection, and serve
as the foundation for the reranking pipeline (Step 4).

Order of operations: Run AFTER enrich.py, BEFORE classify.py.
"""

import json
import os
import time
from pathlib import Path

from isaacus import Isaacus

from text_utils import SKIP_PATHS, strip_footer

API_KEY = os.environ.get(
    "ISAACUS_API_KEY",
    "iuak_v1_alcvXayhjV_8O92bz8S8kphmSOBpiNNoDDAQoexh6q1_feeb2b85",
)
BATCH_SIZE = 32  # Embedding supports up to 128 texts per request
OUTPUT_DIR = Path(__file__).parent / "embedding_results"
ENRICHMENT_DIR = Path(__file__).parent / "enrichment_results"
MANIFEST_PATH = ENRICHMENT_DIR / "manifest.json"


def load_enrichment(filename: str) -> dict:
    """Load a single enrichment JSON file."""
    with open(ENRICHMENT_DIR / filename) as f:
        return json.load(f)


def extract_main_segments(doc: dict) -> list[dict]:
    """Extract main-body segments with their text spans."""
    text = doc["text"]
    segments = []
    for seg in doc["segments"]:
        if seg["category"] != "main" or seg["kind"] != "item":
            continue
        start, end = seg["span"]["start"], seg["span"]["end"]
        seg_text = text[start:end].strip()
        if len(seg_text) < 20:  # Skip trivially small segments
            continue
        segments.append({
            "id": seg["id"],
            "text": seg_text,
            "start": start,
            "end": end,
            "level": seg["level"],
        })
    return segments


def embed_batch(client: Isaacus, texts: list[str], task: str) -> tuple[list[list[float]], int]:
    """Embed a batch of texts, returning vectors in input order."""
    response = client.embeddings.create(
        model="kanon-2-embedder",
        texts=texts,
        task=task,
        overflow_strategy="drop_end",
    )
    # Sort by index to guarantee input order
    sorted_embeddings = sorted(response.embeddings, key=lambda e: e.index)
    return [e.embedding for e in sorted_embeddings], response.usage.input_tokens


def embed_in_batches(
    client: Isaacus, texts: list[str], task: str, label: str
) -> tuple[list[list[float]], int]:
    """Embed a list of texts in batches with progress reporting."""
    all_embeddings = []
    total_tokens = 0
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"  {label} batch {batch_num}/{total_batches} ({len(batch)} texts)...")

        retries = 0
        while retries < 4:
            try:
                embeddings, tokens = embed_batch(client, batch, task)
                all_embeddings.extend(embeddings)
                total_tokens += tokens
                break
            except Exception as e:
                retries += 1
                wait = 2 ** retries
                print(f"    Error: {e}. Retry {retries}/4 in {wait}s...")
                time.sleep(wait)
        else:
            print(f"    FAILED after 4 retries. Padding with empty vectors.")
            all_embeddings.extend([[] for _ in batch])

    return all_embeddings, total_tokens


def main():
    client = Isaacus(api_key=API_KEY)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load enrichment manifest
    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    # Load embedding progress
    embed_manifest_path = OUTPUT_DIR / "manifest.json"
    if embed_manifest_path.exists():
        with open(embed_manifest_path) as f:
            embed_manifest = json.load(f)
    else:
        embed_manifest = {}

    remaining = [
        (path, info)
        for path, info in manifest.items()
        if path not in embed_manifest and path not in SKIP_PATHS
    ]
    print(f"Embedding pipeline: {len(remaining)} documents remaining of {len(manifest)} total")

    total_tokens = 0
    processed = 0

    for path, info in remaining:
        doc = load_enrichment(info["output_file"])
        doc_text = strip_footer(doc["text"])
        if not doc_text:
            continue

        print(f"\n[{processed + 1}/{len(remaining)}] {path}")

        # --- Document-level embedding ---
        doc_embeddings, doc_tokens = embed_batch(client, [doc_text], "retrieval/document")
        total_tokens += doc_tokens

        # --- Segment-level embeddings ---
        segments = extract_main_segments(doc)
        seg_embeddings = []
        if segments:
            seg_texts = [s["text"] for s in segments]
            seg_vecs, seg_tokens = embed_in_batches(
                client, seg_texts, "retrieval/document", "Segments"
            )
            total_tokens += seg_tokens
            for seg, vec in zip(segments, seg_vecs):
                seg_embeddings.append({
                    "id": seg["id"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "level": seg["level"],
                    "text_preview": seg["text"][:100],
                    "embedding": vec,
                })

        # Save result
        output_name = info["output_file"].replace(".json", "_embeddings.json")
        result = {
            "source": path,
            "document_embedding": doc_embeddings[0],
            "num_segments_embedded": len(seg_embeddings),
            "segment_embeddings": seg_embeddings,
        }
        with open(OUTPUT_DIR / output_name, "w") as f:
            json.dump(result, f, ensure_ascii=False)

        embed_manifest[path] = {
            "output_file": output_name,
            "doc_embedding_dim": len(doc_embeddings[0]),
            "num_segments_embedded": len(seg_embeddings),
        }

        # Save manifest incrementally
        with open(embed_manifest_path, "w") as f:
            json.dump(embed_manifest, f, indent=2, ensure_ascii=False)

        processed += 1
        print(f"  Done. {len(seg_embeddings)} segments embedded. Tokens so far: {total_tokens}")

    print(f"\nEmbedding complete! {processed} documents. Total tokens: {total_tokens}")
    print(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
