"""
rank.py
Ranking execution pipeline for the Redrob candidate ranker.
Retrieves and reranks candidates to output top 100 results.
Follows PEP 8 style guide.
"""

import argparse
import csv
import gzip
import json
import os
import sys
import time
import subprocess
from pathlib import Path

# Auto-re-execute using the virtual environment interpreter if available and not already inside it
if not os.environ.get("VIRTUAL_ENV"):
    venv_python_bin = os.path.join(os.path.dirname(__file__), ".venv", "bin", "python")
    venv_python_win = os.path.join(os.path.dirname(__file__), ".venv", "Scripts", "python.exe")
    venv_python = venv_python_bin if os.path.exists(venv_python_bin) else venv_python_win
    if os.path.exists(venv_python) and sys.executable != venv_python:
        try:
            sys.exit(subprocess.call([venv_python] + sys.argv))
        except Exception:
            pass

import numpy as np

# We import modules from src package
from src.config import (
    FAISS_INDEX_PATH, JD_EMBEDDING_PATH, CANDIDATE_MAP_PATH,
    PRECOMPUTED_FEATURES_PATH, WEIGHTS, CORE_AI_SKILLS,
    VECTOR_DB_SKILLS, EMBEDDING_RETRIEVAL_SKILLS
)
from src.io_utils import (
    load_faiss_index, load_numpy_array, stream_candidates
)
from src.precompute_features import extract_features
from src.text_utils import build_candidate_rich_text, count_keyword_matches
from src.reasoning import generate_reasoning
from src.validation_helpers import verify_ranking_output


def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Rank candidates for Job Description.")
    parser.add_argument(
        "--candidates",
        type=str,
        default="data/candidates.jsonl",
        help="Path to candidates jsonl or jsonl.gz file"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="outputs/submission.csv",
        help="Path to write the final ranked CSV"
    )
    parser.add_argument(
        "--sample-mode",
        action="store_true",
        help="Run ranking in artifact-free mode for demo/sandbox check"
    )
    return parser.parse_args()


def check_artifacts():
    """
    Checks if all required precomputed artifacts exist.
    Fails with clear run instructions if anything is missing.
    """
    missing = []
    for path in [FAISS_INDEX_PATH, JD_EMBEDDING_PATH, CANDIDATE_MAP_PATH, PRECOMPUTED_FEATURES_PATH]:
        if not path.exists():
            missing.append(path.name)

    if missing:
        msg = (
            f"Required precomputed artifacts are missing ({', '.join(missing)}). "
            f"Please run the precomputation script first:\n"
            f"python precompute.py --candidates data/candidates.jsonl.gz --job data/job_description.docx"
        )
        print(f"Error: {msg}", file=sys.stderr)
        sys.exit(1)


def main():
    args = parse_args()
    start_time = time.time()

    # Step 1: Initialize candidates processing based on mode
    ranked_candidates = []

    if args.sample_mode:
        print("Running in SAMPLE MODE (artifact-free mode)...")
        # Load all candidates from inputs, run feature extraction dynamically,
        # and compute a proxy dense retrieval score using keyword matching.
        candidates_file = Path(args.candidates)
        if not candidates_file.exists():
            if args.candidates.endswith(".gz"):
                # Try uncompressed fallback
                uncompressed = Path(args.candidates[:-3])
                if uncompressed.exists():
                    candidates_file = uncompressed
            else:
                # Try compressed fallback
                compressed = Path(args.candidates + ".gz")
                if compressed.exists():
                    candidates_file = compressed

        if not candidates_file.exists():
            print(f"Error: Candidate file not found: {args.candidates}", file=sys.stderr)
            sys.exit(1)

        print(f"Loading candidates dynamically from {candidates_file}...")
        cands = list(stream_candidates(candidates_file))
        print(f"Loaded {len(cands)} candidates.")

        # Keywords for dense score approximation
        jd_keywords = CORE_AI_SKILLS + VECTOR_DB_SKILLS + EMBEDDING_RETRIEVAL_SKILLS

        for c in cands:
            features = extract_features(c)
            
            # Approximate dense semantic score using keyword match frequency
            rich_text = build_candidate_rich_text(c)
            match_count = count_keyword_matches(rich_text, jd_keywords)
            dense_score = min(1.0, match_count / 10.0)  # Normalize
            
            features["dense_score"] = dense_score
            ranked_candidates.append(features)

    else:
        # Standard Production Mode
        print("Verifying precomputed artifacts...")
        check_artifacts()

        print("Step 2: Loading FAISS index and JD embeddings...")
        index = load_faiss_index(FAISS_INDEX_PATH)
        jd_embedding = load_numpy_array(JD_EMBEDDING_PATH)
        
        with open(CANDIDATE_MAP_PATH, "r", encoding="utf-8") as f:
            candidate_id_map = json.load(f)

        # Retrieve top 50,000 matches via FAISS
        # Query embedding is shape (1, dimension). We search for top 50000.
        print("Searching FAISS index (recall top 50,000)...")
        distances, indices = index.search(jd_embedding, 50000)
        
        # Flatten outputs
        flat_distances = distances[0]
        flat_indices = indices[0]

        # Map FAISS scores (cosine similarities) to normalized dense_scores
        retrieved_candidates = {}
        for rank_idx, (idx, dist) in enumerate(zip(flat_indices, flat_distances)):
            if idx == -1:
                continue
            cand_id = candidate_id_map[idx]
            # Map typical cosine similarity range [0.2, 0.7] to [0.0, 1.0]
            norm_dense = (dist - 0.2) / 0.5
            norm_dense = max(0.0, min(1.0, norm_dense))
            retrieved_candidates[cand_id] = norm_dense

        print(f"Retrieved {len(retrieved_candidates)} candidate IDs from FAISS.")

        print("Step 3: Loading precomputed candidates features...")
        # Read the gzipped candidate features and score all candidates
        with gzip.open(PRECOMPUTED_FEATURES_PATH, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                cand_id = row["candidate_id"]
                
                # Check if they pass the qualification gate
                if not row.get("is_qualified", False):
                    continue
                
                row["dense_score"] = retrieved_candidates.get(cand_id, 0.0)
                ranked_candidates.append(row)

    print(f"Scoring {len(ranked_candidates)} candidates...")
    # Step 4: Run hybrid scoring formula on all candidates
    for cand in ranked_candidates:
        dense_score = cand.get("dense_score", 0.0)
        
        # Weighted base score calculation
        base_score = (
            WEIGHTS["engineering_role_score"] * cand.get("engineering_role_score", 0.0) +
            WEIGHTS["production_evidence_score"] * cand["production_evidence_score"] +
            WEIGHTS["ranking_eval_score"] * cand["ranking_eval_score"] +
            WEIGHTS["vector_search_score"] * cand["vector_search_score"] +
            WEIGHTS["dense_score"] * dense_score +
            WEIGHTS["embedding_retrieval_score"] * cand["embedding_retrieval_score"] +
            WEIGHTS["python_score"] * cand["python_score"] +
            WEIGHTS["startup_shipper_score"] * cand["startup_shipper_score"] +
            WEIGHTS["product_company_score"] * cand["product_company_score"] +
            WEIGHTS["experience_fit_score"] * cand["experience_fit_score"] +
            WEIGHTS["behavioral_signal_score"] * cand["behavioral_signal_score"] +
            WEIGHTS["location_fit_score"] * cand["location_fit_score"]
            - cand["trap_risk_penalty"]
            - cand["disqualifier_penalty"]
        )
        base_score = max(0.0, min(1.0, base_score))
        
        # Map final score to non-overlapping ranges based on Tier
        # Tier 1 (A): [0.75, 1.0]
        # Tier 2 (B): [0.50, 0.74]
        # Tier 3 (C): [0.25, 0.49]
        # Tier 4 (D): [0.05, 0.24]
        # Tier 5 (Disqualified): 0.0
        tier = cand.get("tier", 4)
        if tier == 1:
            final_score = 0.75 + 0.25 * base_score
        elif tier == 2:
            final_score = 0.50 + 0.24 * base_score
        elif tier == 3:
            final_score = 0.25 + 0.24 * base_score
        elif tier == 4:
            final_score = 0.05 + 0.19 * base_score
        else:
            final_score = 0.0
            
        cand["final_score"] = round(final_score, 6)

    # Step 5: Sort candidates based on the composite priority rules
    # We sort strictly by final_score descending, with secondary tie-breakers:
    # technical_core desc, production_evidence desc, ranking_eval desc, behavioral_signal desc, candidate_id asc.
    print("Sorting and generating rankings...")
    ranked_candidates.sort(
        key=lambda x: (
            -x["final_score"],
            -x["technical_core_score"],
            -x["production_evidence_score"],
            -x["ranking_eval_score"],
            -x["behavioral_signal_score"],
            x["candidate_id"]
        )
    )

    # Step 6: Select top 100 candidates with strict rank-quality guardrails
    def select_top_100_with_guardrails(ranked_cands):
        selected = []
        neg_keywords = [
            "hr", "recruiter", "sales", "marketing", "accountant", "accounting", 
            "designer", "graphic designer", "civil", "mechanical", "support", 
            "customer support", "content writer", "operations manager", "operations"
        ]
        
        idx = 0
        while len(selected) < 100 and idx < len(ranked_cands):
            cand = ranked_cands[idx]
            title = cand["evidence"]["title"].lower()
            years = cand["evidence"]["years"]
            tech = cand["technical_core_score"]
            prod = cand["production_evidence_score"]
            score = cand["final_score"]
            
            # Guardrail 0: Zero score check (never select 0.0 unless absolutely necessary and no other positive candidates exist)
            if score <= 0.0:
                idx += 1
                continue
            
            # Guardrail 1: Top 10 & Top 20 has no disqualified titles
            if len(selected) < 20:
                if any(kw in title for kw in neg_keywords):
                    idx += 1
                    continue
            
            # Guardrail 2: Top 20 has no candidates under 3 years unless exceptional
            if len(selected) < 20 and years < 3.0:
                is_exceptional = (tech >= 0.5 and prod >= 0.5)
                if not is_exceptional:
                    idx += 1
                    continue
            
            # Guardrail 3: Top 20 has no candidates under 2 years
            if len(selected) < 20 and years < 2.0:
                idx += 1
                continue
                
            # Guardrail 4: Top 50 should not include Business Analyst or Project Manager unless explicit production ML evidence
            if len(selected) < 50:
                if "business analyst" in title or "project manager" in title or "pm" == title or "ba" == title:
                    if prod <= 0.0:
                        idx += 1
                        continue

            # General: Top 100 has no disqualified titles
            if any(kw in title for kw in neg_keywords):
                idx += 1
                continue
                
            selected.append(cand)
            idx += 1
            
        return selected

    top_100 = select_top_100_with_guardrails(ranked_candidates)
    
    # Check if we have enough candidates (backfill if necessary from real candidates)
    if len(top_100) < 100:
        needed = 100 - len(top_100)
        print(
            f"Warning: Only found {len(top_100)} candidates. "
            f"Backfilling with {needed} real candidate(s) from candidate pool."
        )
        
        # Determine candidates to load for backfill
        existing_ids = {c["candidate_id"] for c in ranked_candidates}
        
        # Locate candidate files
        fallback_files = [
            Path("data/candidates.jsonl"),
            Path("data/candidates.jsonl.gz"),
            Path("data/candidates.json"),
            Path("data/sample_candidates.json")
        ]
        
        backfill_candidates = []
        for p in fallback_files:
            if p.exists() and p != Path(args.candidates):
                try:
                    for cand in stream_candidates(p):
                        cand_id = cand["candidate_id"]
                        if cand_id not in existing_ids:
                            features = extract_features(cand)
                            features["dense_score"] = 0.0  # Proxy score
                            
                            # Score backfill candidate
                            base_score = (
                                WEIGHTS["engineering_role_score"] * features.get("engineering_role_score", 0.0) +
                                WEIGHTS["production_evidence_score"] * features["production_evidence_score"] +
                                WEIGHTS["ranking_eval_score"] * features["ranking_eval_score"] +
                                WEIGHTS["vector_search_score"] * features["vector_search_score"] +
                                WEIGHTS["dense_score"] * features.get("dense_score", 0.0) +
                                WEIGHTS["embedding_retrieval_score"] * features["embedding_retrieval_score"] +
                                WEIGHTS["python_score"] * features["python_score"] +
                                WEIGHTS["startup_shipper_score"] * features["startup_shipper_score"] +
                                WEIGHTS["product_company_score"] * features["product_company_score"] +
                                WEIGHTS["experience_fit_score"] * features["experience_fit_score"] +
                                WEIGHTS["behavioral_signal_score"] * features["behavioral_signal_score"] +
                                WEIGHTS["location_fit_score"] * features["location_fit_score"]
                                - features["trap_risk_penalty"]
                                - features["disqualifier_penalty"]
                            )
                            base_score = max(0.0, min(1.0, base_score))
                            
                            tier = features.get("tier", 4)
                            if tier == 1:
                                final_score = 0.75 + 0.25 * base_score
                            elif tier == 2:
                                final_score = 0.50 + 0.24 * base_score
                            elif tier == 3:
                                final_score = 0.25 + 0.24 * base_score
                            elif tier == 4:
                                final_score = 0.05 + 0.19 * base_score
                            else:
                                final_score = 0.0
                                
                            features["final_score"] = final_score
                            backfill_candidates.append(features)
                            existing_ids.add(cand_id)
                            
                            if len(backfill_candidates) >= needed:
                                break
                except Exception as e:
                    print(f"Skipping backfill path {p} due to read error: {e}")
                if len(backfill_candidates) >= needed:
                    break
        
        # Append backfill results to top_100
        top_100.extend(backfill_candidates[:needed])

    # Round final scores to 6 decimals
    for cand in top_100:
        cand["final_score"] = round(cand["final_score"], 6)

    # To satisfy the validator's tie-break rule:
    # Stable sort by candidate_id ascending first, then by final_score descending.
    # This guarantees that equal scores will be ordered by candidate_id ascending.
    top_100.sort(key=lambda x: x["candidate_id"])
    top_100.sort(key=lambda x: x["final_score"], reverse=True)

    # Compile the final rows
    output_rows = []
    for rank_pos, cand in enumerate(top_100, start=1):
        cand_id = cand["candidate_id"]
        score = cand["final_score"]
        evidence = cand["evidence"]
        
        # Generate rank-appropriate factual reasoning
        reasoning = generate_reasoning(evidence, rank_pos, cand_id)
        
        output_rows.append({
            "candidate_id": cand_id,
            "rank": rank_pos,
            "score": score,
            "reasoning": reasoning
        })

    # Ensure monotonicity of scores (if any float rounding caused a slight bump)
    for i in range(len(output_rows) - 1):
        if output_rows[i]["score"] < output_rows[i + 1]["score"]:
            output_rows[i + 1]["score"] = output_rows[i]["score"]

    # Step 7: Write to final CSV
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Ranked CSV successfully written to: {out_path}")

    # Step 8: Run post-generation validation checks
    print("Running validation check on written CSV...")
    is_valid, errors = verify_ranking_output(out_path)
    if is_valid:
        print("Validation succeeded. Submission is completely valid.")
    else:
        print("Validation failed with errors:", file=sys.stderr)
        for err in errors:
            print(f"- {err}", file=sys.stderr)
        sys.exit(1)

    elapsed = time.time() - start_time
    print(f"Ranking runtime: {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
