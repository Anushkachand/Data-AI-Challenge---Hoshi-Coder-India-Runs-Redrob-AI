"""
precompute.py
Precomputation pipeline for the Redrob candidate ranker.
Generates FAISS index, JD embeddings, and candidate features.
Follows PEP 8 style guide.
"""

import argparse
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
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import (
    ARTIFACTS_DIR, FAISS_INDEX_PATH, JD_EMBEDDING_PATH,
    CANDIDATE_MAP_PATH, PRECOMPUTED_FEATURES_PATH,
    PRECOMPUTE_MANIFEST_PATH, EMBEDDING_MODEL_NAME
)
from src.io_utils import load_docx_text, save_faiss_index, save_numpy_array
from src.text_utils import build_candidate_rich_text
from src.precompute_features import extract_features


def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Precompute candidates features and embeddings.")
    parser.add_argument(
        "--candidates",
        type=str,
        default="data/candidates.jsonl",
        help="Path to candidates jsonl or jsonl.gz file"
    )
    parser.add_argument(
        "--job",
        type=str,
        default="data/job_description.docx",
        help="Path to job description docx file"
    )
    parser.add_argument(
        "--features-only",
        action="store_true",
        help="Only recompute candidate features, skip sentence transformer embedding generation"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    start_time = time.time()

    # Ensure artifacts directory exists
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.features_only:
        print("Skipping dense embeddings generation. Recomputing candidate features only...")
        candidates_file = Path(args.candidates)
        if not candidates_file.exists():
            if args.candidates.endswith(".gz"):
                uncompressed = Path(args.candidates[:-3])
                if uncompressed.exists():
                    candidates_file = uncompressed
            else:
                compressed = Path(args.candidates + ".gz")
                if compressed.exists():
                    candidates_file = compressed

        if not candidates_file.exists():
            raise FileNotFoundError(f"Candidates file not found: {args.candidates}")

        print(f"Streaming from: {candidates_file}")
        features_out = gzip.open(PRECOMPUTED_FEATURES_PATH, "wt", encoding="utf-8")

        def candidate_generator(path):
            if path.suffix == ".gz":
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            yield json.loads(line)
            else:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            yield json.loads(line)

        count = 0
        for cand in tqdm(candidate_generator(candidates_file), desc="Recomputing features"):
            features = extract_features(cand)
            features_out.write(json.dumps(features) + "\n")
            count += 1

        features_out.close()
        elapsed = time.time() - start_time
        print(f"Features recomputation complete in {elapsed:.2f} seconds. Processed {count} candidates.")
        return

    print("Step 1: Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    dimension = model.get_sentence_embedding_dimension()
    print(f"Model loaded. Dimension: {dimension}")

    print("Step 2: Processing job description...")
    if not os.path.exists(args.job):
        raise FileNotFoundError(f"Job description docx not found: {args.job}")
    jd_text = load_docx_text(args.job)
    print("Generating JD embedding...")
    jd_emb = model.encode([jd_text], convert_to_numpy=True)
    faiss.normalize_L2(jd_emb)  # Normalize for cosine similarity
    save_numpy_array(jd_emb, JD_EMBEDDING_PATH)
    print(f"JD embedding saved to {JD_EMBEDDING_PATH}")

    print("Step 3: Processing candidates...")
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
        raise FileNotFoundError(f"Candidates file not found: {args.candidates}")

    print(f"Streaming from: {candidates_file}")

    # Set up FAISS IndexFlatIP (inner product on normalized vectors = cosine similarity)
    index = faiss.IndexFlatIP(dimension)
    candidate_id_map = []
    
    # Open compressed features file for writing
    features_out = gzip.open(PRECOMPUTED_FEATURES_PATH, "wt", encoding="utf-8")

    batch_size = 1024
    current_batch_texts = []
    current_batch_ids = []

    # Stream candidates, compute features, build texts for embeddings
    # We use a custom parser to read line by line to keep RAM usage small
    def candidate_generator(path):
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)
        else:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)

    count = 0
    for cand in tqdm(candidate_generator(candidates_file), desc="Processing candidates"):
        cand_id = cand["candidate_id"]
        
        # Extract features and write to gzip features file
        features = extract_features(cand)
        features_out.write(json.dumps(features) + "\n")

        # Compile rich text for sentence transformers
        rich_text = build_candidate_rich_text(cand)
        current_batch_texts.append(rich_text)
        current_batch_ids.append(cand_id)

        # Batch encode to avoid holding all embeddings in RAM
        if len(current_batch_texts) >= batch_size:
            embeddings = model.encode(current_batch_texts, convert_to_numpy=True)
            faiss.normalize_L2(embeddings)
            index.add(embeddings)
            candidate_id_map.extend(current_batch_ids)
            
            current_batch_texts = []
            current_batch_ids = []
            
        count += 1

    # Encode remaining items in the final batch
    if current_batch_texts:
        embeddings = model.encode(current_batch_texts, convert_to_numpy=True)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        candidate_id_map.extend(current_batch_ids)

    # Close features file
    features_out.close()

    print("Step 4: Saving FAISS index and ID maps...")
    save_faiss_index(index, FAISS_INDEX_PATH)
    
    with open(CANDIDATE_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(candidate_id_map, f)

    # Save metadata manifest
    manifest = {
        "embedding_model": EMBEDDING_MODEL_NAME,
        "vector_dimension": dimension,
        "total_candidates": count,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "faiss_index_size": index.ntotal
    }
    with open(PRECOMPUTE_MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - start_time
    print(f"Precomputation complete in {elapsed:.2f} seconds.")
    print(f"Total candidates indexed: {count}")
    print(f"Artifacts successfully written to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
