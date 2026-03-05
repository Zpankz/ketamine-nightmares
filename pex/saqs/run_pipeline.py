#!/usr/bin/env python3
"""
Master pipeline orchestrator for PEX/SAQ corpus processing.

Executes all 5 processing steps in optimal dependency order:

  Step 0: enrich.py         — ALREADY COMPLETE (prerequisite)
  Step 1: embed.py          — Document + segment embeddings
  Step 2: classify.py       — Hierarchical ontology classification
  Step 3: extract_qa.py     — Extractive QA knowledge base
  Step 4: rerank_search.py  — Similarity matrix + search index (needs Step 1)
  Step 5: segment_process.py — Segment-level deep processing

Dependency graph:
                 ┌─── embed.py ───────┐
                 │                    │
  enrich.py ─────┼─── classify.py     ├─── rerank_search.py
  (done)         │                    │
                 ├─── extract_qa.py   │
                 │                    │
                 └─── segment_process.py

Steps 1, 2, 3 are independent and could run concurrently.
Step 4 depends on Step 1.
Step 5 is independent but most token-intensive — runs last.

Usage:
    python run_pipeline.py           # Run all steps sequentially
    python run_pipeline.py 1         # Run only Step 1
    python run_pipeline.py 1 2 3     # Run Steps 1, 2, 3
    python run_pipeline.py 4         # Run Step 4 (requires Step 1 complete)
"""

import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent

STEPS = {
    1: ("embed.py", "Embedding (documents + segments)"),
    2: ("classify.py", "Classification (ontology tagging)"),
    3: ("extract_qa.py", "Extractive QA (knowledge base)"),
    4: ("rerank_search.py", "Reranking search index"),
    5: ("segment_process.py", "Segment-level deep processing"),
}

DEPENDENCIES = {
    1: [],
    2: [],
    3: [],
    4: [1],  # Needs embeddings
    5: [],   # Independent, but runs last by convention
}


def check_enrichment_complete():
    """Verify that enrichment (Step 0) has been run."""
    manifest = SCRIPTS_DIR / "enrichment_results" / "manifest.json"
    if not manifest.exists():
        print("ERROR: Enrichment results not found. Run enrich.py first.")
        sys.exit(1)
    import json
    with open(manifest) as f:
        data = json.load(f)
    print(f"Enrichment verified: {len(data)} documents available")
    return True


def check_step_complete(step: int) -> bool:
    """Check if a step's output manifest exists."""
    output_dirs = {
        1: "embedding_results",
        2: "classification_results",
        3: "qa_results",
        4: "search_index",
        5: "segment_results",
    }
    manifest = SCRIPTS_DIR / output_dirs[step] / "manifest.json"
    return manifest.exists()


def run_step(step: int):
    """Run a single pipeline step."""
    script, label = STEPS[step]
    script_path = SCRIPTS_DIR / script

    # Check dependencies
    for dep in DEPENDENCIES[step]:
        if not check_step_complete(dep):
            dep_label = STEPS[dep][1]
            print(f"  Dependency not met: Step {dep} ({dep_label}) must complete first.")
            print(f"  Running Step {dep} first...")
            run_step(dep)

    print(f"\n{'='*60}")
    print(f"STEP {step}: {label}")
    print(f"Script: {script}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(SCRIPTS_DIR),
    )
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"\nSTEP {step} FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        return False

    print(f"\nSTEP {step} COMPLETE in {elapsed:.1f}s")
    return True


def main():
    check_enrichment_complete()

    if len(sys.argv) > 1:
        # Run specific steps
        steps_to_run = [int(s) for s in sys.argv[1:]]
    else:
        # Run all steps in optimal order
        steps_to_run = [1, 2, 3, 4, 5]

    print(f"Pipeline: running steps {steps_to_run}")
    start = time.time()

    for step in steps_to_run:
        if step not in STEPS:
            print(f"Unknown step: {step}. Valid steps: {list(STEPS.keys())}")
            continue
        success = run_step(step)
        if not success:
            print(f"\nPipeline halted at Step {step}.")
            sys.exit(1)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE in {elapsed:.1f}s")
    print(f"{'='*60}")

    # Summary
    print("\nOutput directories:")
    for step in steps_to_run:
        if step in STEPS:
            output_dirs = {
                1: "embedding_results/",
                2: "classification_results/",
                3: "qa_results/",
                4: "search_index/",
                5: "segment_results/",
            }
            print(f"  Step {step}: {output_dirs[step]}")


if __name__ == "__main__":
    main()
