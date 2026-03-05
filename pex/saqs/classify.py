#!/usr/bin/env python3
"""
Step 2: Classify PEX/SAQ documents using Isaacus Universal Classification.

Applies a hierarchical classification ontology designed specifically for
anaesthesia primary exam SAQs. Uses zero-shot IQL queries — no training
data required.

The ontology has three axes:
  1. COGNITIVE DOMAIN — what type of thinking the question demands
  2. CLINICAL RELEVANCE — how directly applicable to clinical practice
  3. CONTENT DIMENSIONS — cross-cutting themes that span categories

These axes are orthogonal to the existing directory-based topic taxonomy
(pharmacology/, physiology/, other/) and add a layer of pedagogical and
clinical metadata that the enrichment alone cannot provide.

Order of operations: Run AFTER enrich.py. Independent of embed.py.
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
OUTPUT_DIR = Path(__file__).parent / "classification_results"
ENRICHMENT_DIR = Path(__file__).parent / "enrichment_results"
MANIFEST_PATH = ENRICHMENT_DIR / "manifest.json"

# ============================================================================
# CLASSIFICATION ONTOLOGY
#
# Each axis contains queries written as plain-English statements that the
# Isaacus Universal Classifier scores against document text (0–1 scale).
# Scores > 0.5 indicate positive classification.
#
# IQL operators (AND, OR, NOT, +, >, <) compose compound queries.
# ============================================================================

ONTOLOGY = {
    # ------------------------------------------------------------------
    # AXIS 1: COGNITIVE DOMAIN
    # Maps Bloom's taxonomy levels to exam question demands.
    # ------------------------------------------------------------------
    "cognitive_domain": {
        "factual_recall": {
            "query": (
                "{This requires listing, naming, or defining specific facts, "
                "values, or classifications}"
            ),
            "is_iql": True,
            "description": "Pure recall of facts, values, definitions, or lists",
        },
        "mechanistic_explanation": {
            "query": (
                "{This requires explaining a physiological mechanism or "
                "pharmacological mechanism of action} AND "
                "{This describes how or why something occurs at a molecular, "
                "cellular, or systems level}"
            ),
            "is_iql": True,
            "description": "Explain how/why at mechanistic level",
        },
        "comparative_analysis": {
            "query": (
                "{This requires comparing or contrasting two or more drugs, "
                "agents, techniques, or physiological processes}"
            ),
            "is_iql": True,
            "description": "Compare/contrast multiple entities",
        },
        "quantitative_reasoning": {
            "query": (
                "{This requires calculating, deriving equations, interpreting "
                "graphs, or applying mathematical relationships}"
            ),
            "is_iql": True,
            "description": "Mathematical or graphical reasoning required",
        },
        "applied_clinical": {
            "query": (
                "{This requires applying pharmacological or physiological "
                "knowledge to a clinical scenario, patient management decision, "
                "or practical technique}"
            ),
            "is_iql": True,
            "description": "Apply knowledge to clinical scenarios",
        },
    },

    # ------------------------------------------------------------------
    # AXIS 2: CLINICAL RELEVANCE TIER
    # How directly the content maps to bedside anaesthetic practice.
    # ------------------------------------------------------------------
    "clinical_relevance": {
        "immediately_applicable": {
            "query": (
                "{This content directly describes drug dosing, monitoring "
                "techniques, equipment operation, or patient management "
                "decisions used during anaesthesia}"
            ),
            "is_iql": True,
            "description": "Directly used at the bedside",
        },
        "foundational_science": {
            "query": (
                "{This content describes basic science mechanisms, cellular "
                "physiology, or molecular pharmacology that underpins clinical "
                "practice but is not directly applied at the bedside}"
            ),
            "is_iql": True,
            "description": "Underlying science that informs practice",
        },
        "integrative": {
            "query": (
                "{This content bridges basic science and clinical practice by "
                "explaining how physiological or pharmacological principles "
                "translate into clinical effects or treatment decisions}"
            ),
            "is_iql": True,
            "description": "Bridges basic science to clinical application",
        },
    },

    # ------------------------------------------------------------------
    # AXIS 3: CROSS-CUTTING CONTENT DIMENSIONS
    # Thematic labels that cut across the directory-based taxonomy.
    # A document can match multiple dimensions.
    # ------------------------------------------------------------------
    "content_dimensions": {
        "dose_response_kinetics": {
            "query": (
                "{This discusses dose-response relationships, pharmacokinetic "
                "parameters, half-life, clearance, volume of distribution, "
                "bioavailability, or drug concentration curves}"
            ),
            "is_iql": True,
            "description": "PK/PD parameters and dose-response",
        },
        "receptor_pharmacology": {
            "query": (
                "{This discusses specific receptors, receptor subtypes, "
                "agonists, antagonists, receptor binding, signal transduction "
                "pathways, or second messenger systems}"
            ),
            "is_iql": True,
            "description": "Receptor mechanisms and ligand interactions",
        },
        "organ_system_effects": {
            "query": (
                "{This discusses effects on the cardiovascular system, "
                "respiratory system, central nervous system, renal system, "
                "hepatic system, or other organ systems}"
            ),
            "is_iql": True,
            "description": "Multi-organ or systems-level effects",
        },
        "special_populations": {
            "query": (
                "{This discusses altered physiology or pharmacology in "
                "pregnancy, neonates, the elderly, obese patients, or "
                "patients with renal or hepatic impairment}"
            ),
            "is_iql": True,
            "description": "Pregnancy, neonatal, elderly, obesity, organ failure",
        },
        "monitoring_and_measurement": {
            "query": (
                "{This discusses clinical measurement techniques, monitoring "
                "devices, transducers, waveform interpretation, or "
                "laboratory investigations}"
            ),
            "is_iql": True,
            "description": "Clinical measurement and monitoring",
        },
        "physics_and_engineering": {
            "query": (
                "{This discusses physical principles, gas laws, fluid "
                "dynamics, electrical safety, laser physics, or the "
                "engineering principles of anaesthetic equipment}"
            ),
            "is_iql": True,
            "description": "Physics and equipment engineering principles",
        },
        "acid_base_and_electrolytes": {
            "query": (
                "{This discusses acid-base balance, pH regulation, buffer "
                "systems, electrolyte homeostasis, or disorders of sodium, "
                "potassium, calcium, or magnesium}"
            ),
            "is_iql": True,
            "description": "Acid-base and electrolyte physiology",
        },
        "autonomic_nervous_system": {
            "query": (
                "{This discusses the sympathetic nervous system, "
                "parasympathetic nervous system, autonomic reflexes, "
                "catecholamines, cholinergic transmission, or autonomic "
                "pharmacology}"
            ),
            "is_iql": True,
            "description": "ANS physiology and pharmacology",
        },
        "haemostasis_and_transfusion": {
            "query": (
                "{This discusses coagulation pathways, platelet function, "
                "anticoagulant drugs, thrombolysis, blood products, "
                "transfusion reactions, or point-of-care coagulation testing}"
            ),
            "is_iql": True,
            "description": "Coagulation, bleeding, and transfusion",
        },
        "pain_and_analgesia": {
            "query": (
                "{This discusses nociception, pain pathways, analgesic "
                "mechanisms, opioid pharmacology, local anaesthetic "
                "mechanisms, or regional anaesthesia techniques}"
            ),
            "is_iql": True,
            "description": "Pain physiology and analgesic pharmacology",
        },
        "respiratory_gas_exchange": {
            "query": (
                "{This discusses oxygen transport, carbon dioxide transport, "
                "ventilation-perfusion matching, the oxygen-haemoglobin "
                "dissociation curve, dead space, or shunt}"
            ),
            "is_iql": True,
            "description": "Gas exchange and oxygen delivery",
        },
        "neuromuscular_and_cns": {
            "query": (
                "{This discusses neuromuscular transmission, muscle "
                "relaxants, neuromuscular monitoring, the blood-brain "
                "barrier, cerebral blood flow, intracranial pressure, "
                "or mechanisms of general anaesthesia}"
            ),
            "is_iql": True,
            "description": "Neuromuscular and central nervous system",
        },
    },
}


def flatten_ontology() -> list[tuple[str, str, dict]]:
    """Flatten ontology into (axis, label, config) tuples."""
    items = []
    for axis, labels in ONTOLOGY.items():
        for label, config in labels.items():
            items.append((axis, label, config))
    return items


def classify_document(
    client: Isaacus,
    text: str,
    query: str,
    is_iql: bool,
) -> dict:
    """Classify a single document against a single query."""
    response = client.classifications.universal.create(
        model="kanon-universal-classifier",
        query=query,
        texts=[text],
        is_iql=is_iql,
        scoring_method="chunk_max",
    )
    classification = response.classifications[0]
    return {
        "score": classification.score,
        "top_chunks": [
            {
                "text": c.text[:200] if c.text else None,
                "score": c.score,
                "start": c.start,
                "end": c.end,
            }
            for c in (classification.chunks or [])[:3]
        ],
        "input_tokens": response.usage.input_tokens,
    }


def main():
    client = Isaacus(api_key=API_KEY)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    # Load classification progress
    class_manifest_path = OUTPUT_DIR / "manifest.json"
    if class_manifest_path.exists():
        with open(class_manifest_path) as f:
            class_manifest = json.load(f)
    else:
        class_manifest = {}

    ontology_items = flatten_ontology()
    remaining = [
        (path, info)
        for path, info in manifest.items()
        if path not in class_manifest
    ]
    print(f"Classification pipeline: {len(remaining)} documents, {len(ontology_items)} queries each")
    print(f"Total classifications to run: {len(remaining) * len(ontology_items)}")

    total_tokens = 0
    processed = 0

    for path, info in remaining:
        with open(ENRICHMENT_DIR / info["output_file"]) as f:
            doc = json.load(f)

        text = doc["text"].strip()
        if not text:
            continue

        print(f"\n[{processed + 1}/{len(remaining)}] {path}")

        classifications = {}
        for axis, label, config in ontology_items:
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

            if axis not in classifications:
                classifications[axis] = {}
            classifications[axis][label] = {
                "score": result["score"],
                "description": config["description"],
                "top_chunks": result["top_chunks"],
            }
            total_tokens += result.get("input_tokens", 0)

        # Derive summary labels (winner-take-all per axis for single-select axes)
        summary = {}
        for axis in ["cognitive_domain", "clinical_relevance"]:
            axis_scores = classifications[axis]
            best = max(axis_scores.items(), key=lambda x: x[1]["score"] or 0)
            summary[axis] = {
                "primary": best[0],
                "score": best[1]["score"],
            }
            # Secondary if close (within 0.1 of primary)
            if best[1]["score"]:
                secondaries = [
                    (k, v["score"])
                    for k, v in axis_scores.items()
                    if k != best[0] and v["score"] and v["score"] >= best[1]["score"] - 0.1
                ]
                if secondaries:
                    summary[axis]["secondary"] = [s[0] for s in secondaries]

        # Content dimensions: all above threshold
        dim_scores = classifications["content_dimensions"]
        active_dims = [
            (k, v["score"])
            for k, v in dim_scores.items()
            if v["score"] and v["score"] > 0.5
        ]
        active_dims.sort(key=lambda x: x[1], reverse=True)
        summary["content_dimensions"] = [d[0] for d in active_dims]

        # Save full result
        output_name = info["output_file"].replace(".json", "_classifications.json")
        result_doc = {
            "source": path,
            "summary": summary,
            "classifications": classifications,
        }
        with open(OUTPUT_DIR / output_name, "w") as f:
            json.dump(result_doc, f, indent=2, ensure_ascii=False)

        class_manifest[path] = {
            "output_file": output_name,
            "cognitive_domain": summary["cognitive_domain"]["primary"],
            "clinical_relevance": summary["clinical_relevance"]["primary"],
            "content_dimensions": summary["content_dimensions"],
            "num_active_dimensions": len(summary["content_dimensions"]),
        }

        with open(class_manifest_path, "w") as f:
            json.dump(class_manifest, f, indent=2, ensure_ascii=False)

        processed += 1
        print(f"  -> {summary['cognitive_domain']['primary']} | "
              f"{summary['clinical_relevance']['primary']} | "
              f"dims: {summary['content_dimensions']}")
        print(f"  Tokens so far: {total_tokens}")

    # Save ontology definition alongside results for reproducibility
    with open(OUTPUT_DIR / "ontology.json", "w") as f:
        json.dump(ONTOLOGY, f, indent=2, ensure_ascii=False)

    print(f"\nClassification complete! {processed} documents. Total tokens: {total_tokens}")
    print(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
