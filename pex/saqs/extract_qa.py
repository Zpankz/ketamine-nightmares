#!/usr/bin/env python3
"""
Step 3: Build an Extractive QA index over the PEX/SAQ corpus.

Pre-extracts answers to a bank of high-yield questions across the entire
corpus, producing a structured Q&A knowledge base. Also provides a
query_corpus() function for ad-hoc questions at runtime.

The question bank is derived from the classification ontology (Step 2)
and the directory taxonomy — generating targeted questions for each
topic/subtopic combination.

Order of operations: Run AFTER enrich.py. Independent of embed.py/classify.py.
"""

import json
import os
import time
from pathlib import Path

from isaacus import Isaacus

API_KEY = os.environ.get(
    "ISAACUS_API_KEY",
    "iuak_v1_alcvXayhjV_8O92bz8S8kphmSOBpiNNoDDAQoexh6q1_feeb2b85",
)
OUTPUT_DIR = Path(__file__).parent / "qa_results"
ENRICHMENT_DIR = Path(__file__).parent / "enrichment_results"
MANIFEST_PATH = ENRICHMENT_DIR / "manifest.json"

# ============================================================================
# QUESTION BANK
#
# High-yield questions organised by the corpus taxonomy. Each question is
# run against ALL documents — the extractor's inextractability_score
# naturally filters irrelevant documents (score near 1.0 = no answer).
# ============================================================================

QUESTION_BANK = {
    # --- Pharmacology-focused questions ---
    "pharmacology": [
        "What is the mechanism of action of this drug?",
        "What are the main pharmacokinetic properties including half-life, metabolism, and clearance?",
        "What are the cardiovascular effects?",
        "What are the respiratory effects?",
        "What are the central nervous system effects?",
        "What are the adverse effects and toxicity concerns?",
        "What are the clinical indications and contraindications?",
        "What are the important drug interactions?",
        "How does the dose-response relationship work?",
        "What is the recommended dosing and route of administration?",
    ],
    # --- Physiology-focused questions ---
    "physiology": [
        "What is the normal physiological mechanism described here?",
        "What are the key regulatory mechanisms and feedback loops?",
        "How does this system respond to anaesthesia or critical illness?",
        "What are the normal values and how are they measured?",
        "What happens when this system fails or is impaired?",
        "How does this change in pregnancy, neonates, or the elderly?",
        "What is the relationship between this system and oxygen delivery?",
        "What are the hormonal or neural control mechanisms?",
    ],
    # --- Equipment & Physics questions ---
    "other": [
        "How does this equipment or device work?",
        "What are the safety features and failure modes?",
        "What physical principles underlie this measurement or device?",
        "What are the sources of error or inaccuracy?",
        "How is this calibrated or maintained?",
    ],
    # --- Universal cross-cutting questions ---
    "universal": [
        "What are the key facts that must be memorised for the exam?",
        "What equations or mathematical relationships are described?",
        "What graphs or curves are discussed and what do they show?",
        "What comparisons are made between different agents or systems?",
    ],
}


def get_questions_for_document(source_path: str) -> list[str]:
    """Select relevant questions based on the document's directory category."""
    questions = list(QUESTION_BANK["universal"])  # Always include universal
    category = source_path.split("/")[0]  # pharmacology, physiology, or other
    if category in QUESTION_BANK:
        questions.extend(QUESTION_BANK[category])
    return questions


def extract_answers(
    client: Isaacus,
    text: str,
    query: str,
    top_k: int = 3,
) -> dict:
    """Extract top-k answers for a single query from a single document."""
    response = client.extractions.qa.create(
        model="kanon-answer-extractor",
        query=query,
        texts=[text],
        top_k=top_k,
        ignore_inextractability=False,
    )
    extraction = response.extractions[0]
    return {
        "answers": [
            {
                "text": a.text,
                "start": a.start,
                "end": a.end,
                "score": a.score,
            }
            for a in extraction.answers
        ],
        "inextractability_score": extraction.inextractability_score,
        "input_tokens": response.usage.input_tokens,
    }


def main():
    client = Isaacus(api_key=API_KEY)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    # Load QA progress
    qa_manifest_path = OUTPUT_DIR / "manifest.json"
    if qa_manifest_path.exists():
        with open(qa_manifest_path) as f:
            qa_manifest = json.load(f)
    else:
        qa_manifest = {}

    remaining = [
        (path, info)
        for path, info in manifest.items()
        if path not in qa_manifest
    ]
    print(f"QA extraction pipeline: {len(remaining)} documents remaining")

    total_tokens = 0
    processed = 0

    for path, info in remaining:
        with open(ENRICHMENT_DIR / info["output_file"]) as f:
            doc = json.load(f)

        text = doc["text"].strip()
        if not text:
            continue

        questions = get_questions_for_document(path)
        print(f"\n[{processed + 1}/{len(remaining)}] {path} ({len(questions)} questions)")

        qa_results = []
        answerable_count = 0

        for q_idx, question in enumerate(questions):
            retries = 0
            while retries < 4:
                try:
                    result = extract_answers(client, text, question)
                    break
                except Exception as e:
                    retries += 1
                    wait = 2 ** retries
                    print(f"  Error on Q{q_idx}: {e}. Retry {retries}/4 in {wait}s...")
                    time.sleep(wait)
            else:
                result = {
                    "answers": [],
                    "inextractability_score": 1.0,
                    "input_tokens": 0,
                    "error": "failed",
                }

            total_tokens += result.get("input_tokens", 0)

            # Only keep questions where extraction found meaningful answers
            has_answer = (
                result["answers"]
                and result["answers"][0]["score"] > result["inextractability_score"]
            )
            qa_results.append({
                "question": question,
                "answerable": has_answer,
                "inextractability_score": result["inextractability_score"],
                "answers": result["answers"] if has_answer else [],
            })
            if has_answer:
                answerable_count += 1

        # Save result
        output_name = info["output_file"].replace(".json", "_qa.json")
        result_doc = {
            "source": path,
            "num_questions": len(questions),
            "num_answerable": answerable_count,
            "extractions": qa_results,
        }
        with open(OUTPUT_DIR / output_name, "w") as f:
            json.dump(result_doc, f, indent=2, ensure_ascii=False)

        qa_manifest[path] = {
            "output_file": output_name,
            "num_questions": len(questions),
            "num_answerable": answerable_count,
        }

        with open(qa_manifest_path, "w") as f:
            json.dump(qa_manifest, f, indent=2, ensure_ascii=False)

        processed += 1
        print(f"  -> {answerable_count}/{len(questions)} questions answered. "
              f"Tokens so far: {total_tokens}")

    # Save question bank alongside results
    with open(OUTPUT_DIR / "question_bank.json", "w") as f:
        json.dump(QUESTION_BANK, f, indent=2, ensure_ascii=False)

    print(f"\nQA extraction complete! {processed} documents. Total tokens: {total_tokens}")
    print(f"Results: {OUTPUT_DIR}")


def query_corpus(question: str, top_k: int = 5) -> list[dict]:
    """
    Ad-hoc query: search the entire corpus for answers to a question.

    Returns the top-k answers ranked by score across all documents.
    Requires embedding results for pre-filtering (Step 4 reranking).
    Falls back to brute-force extraction if embeddings unavailable.
    """
    client = Isaacus(api_key=API_KEY)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    all_answers = []
    for path, info in manifest.items():
        with open(ENRICHMENT_DIR / info["output_file"]) as f:
            doc = json.load(f)

        text = doc["text"].strip()
        if not text:
            continue

        try:
            result = extract_answers(client, text, question, top_k=1)
        except Exception:
            continue

        if (
            result["answers"]
            and result["answers"][0]["score"] > result["inextractability_score"]
        ):
            for ans in result["answers"]:
                all_answers.append({
                    "source": path,
                    "answer": ans["text"],
                    "score": ans["score"],
                    "inextractability": result["inextractability_score"],
                })

    all_answers.sort(key=lambda x: x["score"], reverse=True)
    return all_answers[:top_k]


if __name__ == "__main__":
    main()
