# Redrob AI Ranker — Intelligent Candidate Discovery & Ranking

A hybrid candidate ranking system designed to identify the top 100 candidates from a 100,000-candidate pool for a **Senior AI Engineer — Founding Team** role. Built for the Redrob Hackathon.

## Approach & Design

The system implements a **hybrid retrieval + structured scoring + trap-aware reranking** architecture. 

It does not rely solely on raw embeddings, as the candidate pool contains keyword stuffers, domain mismatches, and synthetic honeypots. It combines initial dense semantic similarity (retrieved via FAISS) with 11 structured feature scores:
- **Technical Core Match**: Keywords and proficiencies matching ranking, retrieval, recommendation systems, evaluations.
- **Production & Startup Shipper Evidence**: Search for deployment and scale-focused phrases in career history descriptions.
- **Redrob Behavioral Signals**: Uses recruiter response rate, login activity, email/phone verification, and notice period as modifiers.
- **Trap Detection Heuristics**: Detects and heavily penalizes honeypots (impossible salaries, zero-duration expert skills, stated years of experience vs career history length discrepancies) and keyword stuffers.

---

## Installation & Setup

1. Create a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Pipeline Execution

### 1. Precomputation (Heavy / GPU-optional)
Runs the dense embedding model (`sentence-transformers/all-MiniLM-L6-v2`) over candidate profiles and job description texts to create the FAISS search index and precompute candidate feature scores. This can run outside the 5-minute constraint.
```bash
python precompute.py --candidates data/candidates.jsonl --job data/job_description.docx
```
*(Supports `.jsonl` or `.jsonl.gz` compressed inputs).*

### 2. Ranking Execution (CPU-only, Fast, Network-free)
Executes within the 5-minute constraint. It queries the FAISS index to retrieve the top 10,000 candidates, applies the hybrid scoring formula, filters out trap candidates, deterministic-break ties, generates factual reasons, and writes exactly 100 candidates to the output CSV.
```bash
python rank.py --candidates data/candidates.jsonl --out outputs/submission.csv
```
**Constraint Warning**: If the precomputed artifacts are missing, `rank.py` fails clearly and alerts the user to run `precompute.py`.

### 3. Submission Format Validation
Verifies formatting, monotonicity of scores, rank sequence, unique candidates, and reasoning constraints:
```bash
python validate_submission.py outputs/submission.csv
```

---

## Sandbox & Demo Mode (Lightweight)
Organizers or developers can verify the end-to-end functionality of the ranker on sample files without requiring the full precomputed FAISS artifacts. In sample mode, dense semantic retrieval is approximated using keyword match frequencies:
```bash
python rank.py --candidates data/sample_candidates.json --out outputs/sample_submission.csv --sample-mode
```

---

## File Structure

```txt
redrob-ai-ranker/
├── data/
│   ├── candidates.jsonl
│   ├── job_description.docx
│   └── ...
├── artifacts/
│   ├── candidate_index.faiss
│   ├── candidate_id_map.json
│   ├── candidate_features.jsonl.gz
│   └── jd_embedding.npy
├── src/
│   ├── config.py             # Global configurations, weights, targets
│   ├── io_utils.py           # Word DOCX parsing and compressed IO
│   ├── text_utils.py         # Candidate text builders & keyword counters
│   ├── scoring.py            # Feature subscore calculators
│   ├── trap_detection.py     # Honeypot & keyword stuffers checks
│   ├── reasoning.py          # Deterministic factual reasoning templates
│   └── validation_helpers.py # Post-run sanity check assertions
├── precompute.py             # Offline embedding and features compiler
├── rank.py                   # Reranker entrypoint
├── validate_submission.py    # Hackathon validation script
├── requirements.txt          # Python package requirements
├── approach.md               # Detailed architectural description
└── submission_metadata.yaml  # Sandbox details, declarations
```
