"""
app.py
Streamlit web UI for HR users to rank candidates using the scoring pipeline.
Allows uploading candidate datasets and job descriptions, adjusting weights,
previewing candidates, viewing validation diagnostics, and downloading CSVs.
Follows PEP 8 style guide.
"""

import csv
import gzip
import json
import os
import re
import time
from pathlib import Path
import pandas as pd
import streamlit as st

# Workspace modules imports
from src.config import (
    CANDIDATE_MAP_PATH, FAISS_INDEX_PATH, CORE_AI_SKILLS,
    VECTOR_DB_SKILLS, EMBEDDING_RETRIEVAL_SKILLS, PYTHON_KEYWORDS,
    EVAL_FRAMEWORK_SKILLS, EMBEDDING_MODEL_NAME
)
from src.io_utils import (
    stream_candidates, load_docx_text, load_faiss_index
)
from src.precompute_features import extract_features
from src.text_utils import build_candidate_rich_text, count_keyword_matches
from src.reasoning import generate_reasoning


def check_dataset_matches_original(uploaded_file_path):
    """
    Checks if the uploaded dataset is a subset of the original candidate dataset
    by comparing candidate IDs against the precomputed candidate map.
    """
    if not CANDIDATE_MAP_PATH.exists():
        return False
    try:
        with open(CANDIDATE_MAP_PATH, "r", encoding="utf-8") as f:
            original_ids = set(json.load(f))

        # Check sample of candidate IDs from uploaded dataset
        uploaded_ids = set()
        count = 0
        for cand in stream_candidates(uploaded_file_path):
            cid = cand.get("candidate_id")
            if cid:
                uploaded_ids.add(cid)
            count += 1
            if count >= 100:
                break

        if uploaded_ids and uploaded_ids.issubset(original_ids):
            return True
    except Exception:
        pass
    return False


def rank_candidates_interactive(candidate_path, jd_path, params, top_n=100, progress_callback=None):
    """
    Ranks candidates interactively based on uploaded files and parameter weights.
    Returns a pandas DataFrame containing ranked candidates.
    """
    # 1. Parse Job Description Text
    jd_path = Path(jd_path)
    if jd_path.suffix == ".docx":
        jd_text = load_docx_text(jd_path)
    else:
        with open(jd_path, "r", encoding="utf-8", errors="ignore") as f:
            jd_text = f.read()

    # Determine relevant keywords for proxy dense score
    all_possible_keywords = (
        CORE_AI_SKILLS + VECTOR_DB_SKILLS + EMBEDDING_RETRIEVAL_SKILLS +
        PYTHON_KEYWORDS + EVAL_FRAMEWORK_SKILLS
    )
    jd_keywords = [kw for kw in all_possible_keywords if kw.lower() in jd_text.lower()]
    if not jd_keywords:
        jd_keywords = CORE_AI_SKILLS + VECTOR_DB_SKILLS + EMBEDDING_RETRIEVAL_SKILLS

    # Check if advanced FAISS mode is available
    advanced_available = False
    candidate_id_map = None
    if FAISS_INDEX_PATH.exists() and CANDIDATE_MAP_PATH.exists():
        try:
            with open(CANDIDATE_MAP_PATH, "r", encoding="utf-8") as f:
                candidate_id_map = json.load(f)
            advanced_available = True
        except Exception:
            pass

    use_advanced = params.get("use_advanced", False) and advanced_available
    retrieved_candidates = {}

    if use_advanced:
        try:
            import faiss
            from sentence_transformers import SentenceTransformer

            # Load model and encode JD text
            model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            jd_emb = model.encode([jd_text], convert_to_numpy=True)
            faiss.normalize_L2(jd_emb)

            # Load precomputed FAISS index
            index = load_faiss_index(FAISS_INDEX_PATH)

            # Perform FAISS search (recall top 50,000 matches)
            distances, indices = index.search(jd_emb, min(50000, len(candidate_id_map)))
            flat_distances = distances[0]
            flat_indices = indices[0]

            for idx, dist in zip(flat_indices, flat_distances):
                if idx == -1:
                    continue
                cand_id = candidate_id_map[idx]
                norm_dense = (dist - 0.2) / 0.5
                norm_dense = max(0.0, min(1.0, norm_dense))
                retrieved_candidates[cand_id] = norm_dense
        except Exception as e:
            # Fallback to structured scoring
            use_advanced = False

    # Stream candidates and extract features
    ranked_candidates = []
    cands_generator = stream_candidates(candidate_path)

    # Estimate total count for progress reporting
    total_count = None
    try:
        if candidate_path.suffix != ".gz":
            with open(candidate_path, "r", encoding="utf-8") as f:
                total_count = sum(1 for line in f if line.strip())
    except Exception:
        pass

    count = 0
    for cand in cands_generator:
        count += 1
        if progress_callback and total_count and count % 500 == 0:
            progress_callback(min(0.95, count / total_count))

        features = extract_features(cand)
        cand_id = features["candidate_id"]

        # Calculate dense semantic score
        if use_advanced:
            dense_score = retrieved_candidates.get(cand_id, 0.0)
        else:
            rich_text = build_candidate_rich_text(cand)
            match_count = count_keyword_matches(rich_text, jd_keywords)
            dense_score = min(1.0, match_count / 10.0)

        features["dense_score"] = dense_score

        # Check qualification gate
        penalize_non_eng = params.get("penalize_non_eng", True)
        if not penalize_non_eng:
            features["is_qualified"] = True

        if not features["is_qualified"]:
            continue

        # Subscores
        eng_role_score = features.get("engineering_role_score", 0.0)
        prod_score = features.get("production_evidence_score", 0.0)
        eval_score = features.get("ranking_eval_score", 0.0)
        vector_score = features.get("vector_search_score", 0.0)
        embed_score = features.get("embedding_retrieval_score", 0.0)
        python_score = features.get("python_score", 0.0)
        startup_score = features.get("startup_shipper_score", 0.0)
        prod_company_score = features.get("product_company_score", 0.0)

        # Experience preference
        prefer_experience = params.get("prefer_experience", True)
        if prefer_experience:
            exp_fit_score = features.get("experience_fit_score", 0.0)
        else:
            exp_fit_score = 1.0

        # Location preference
        loc_fit_score = features.get("location_fit_score", 0.0)

        # Behavioral signals
        use_behavioral = params.get("use_behavioral", True)
        if use_behavioral:
            behavior_score = features.get("behavioral_signal_score", 0.0)
        else:
            behavior_score = 0.5

        # Penalties
        remove_traps = params.get("remove_traps", True)
        trap_risk_penalty = features.get("trap_risk_penalty", 0.0) if remove_traps else 0.0
        disqualifier_penalty = features.get("disqualifier_penalty", 0.0) if penalize_non_eng else 0.0

        # Extract weights from params
        tech_core_w = params.get("technical_core_weight", 0.15)
        prod_evidence_w = params.get("production_evidence_weight", 0.18)
        ranking_eval_w = params.get("ranking_eval_weight", 0.14)

        # Search / Retrieval weights mapping
        search_retrieval_val = params.get("search_retrieval_weight", 0.12)
        vector_search_w = search_retrieval_val
        embedding_retrieval_w = 0.5 * search_retrieval_val
        dense_score_w = search_retrieval_val

        # Python / Engineering strength weights mapping
        python_engineering_val = params.get("python_engineering_weight", 0.20)
        eng_role_w = python_engineering_val
        python_score_w = 0.25 * python_engineering_val

        # Startup / Product fit weights mapping
        startup_product_val = params.get("startup_product_weight", 0.07)
        startup_shipper_w = (4 / 7) * startup_product_val
        product_company_w = (3 / 7) * startup_product_val

        experience_fit_w = params.get("experience_fit_weight", 0.03)
        location_fit_w = params.get("location_fit_weight", 0.01)
        behavioral_signal_w = params.get("behavioral_signal_weight", 0.02)

        trap_penalty_strength = params.get("trap_penalty_strength", 1.0)
        non_eng_title_penalty = params.get("non_eng_title_penalty", 1.0)

        # Composite base score formula
        base_score = (
            eng_role_w * eng_role_score +
            prod_evidence_w * prod_score +
            ranking_eval_w * eval_score +
            vector_search_w * vector_score +
            embedding_retrieval_w * embed_score +
            tech_core_w * features.get("technical_core_score", 0.0) +
            python_score_w * python_score +
            dense_score_w * dense_score +
            startup_shipper_w * startup_score +
            product_company_w * prod_company_score +
            experience_fit_w * exp_fit_score +
            location_fit_w * loc_fit_score +
            behavioral_signal_w * behavior_score
            - trap_penalty_strength * trap_risk_penalty
            - non_eng_title_penalty * disqualifier_penalty
        )
        base_score = max(0.0, min(1.0, base_score))

        # Adjust Tier mappings if titles aren't penalized
        tier = features.get("tier", 4)
        if not penalize_non_eng and tier == 5:
            tier = 4

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

        features["final_score"] = round(final_score, 6)
        ranked_candidates.append(features)

    # Sort candidates (Score descending primary, others as secondary tiebreakers)
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

    # Select top candidates using guardrails if requested
    require_prod_evidence = params.get("require_production_evidence", True)

    if require_prod_evidence:
        selected = []
        neg_keywords = [
            "hr", "recruiter", "sales", "marketing", "accountant", "accounting",
            "designer", "graphic designer", "civil", "mechanical", "support",
            "customer support", "content writer", "operations manager", "operations"
        ]

        idx = 0
        while len(selected) < top_n and idx < len(ranked_candidates):
            cand = ranked_candidates[idx]
            title = cand["evidence"]["title"].lower()
            years = cand["evidence"]["years"]
            tech = cand["technical_core_score"]
            prod = cand["production_evidence_score"]
            score = cand["final_score"]

            if score <= 0.0:
                idx += 1
                continue

            if len(selected) < 20:
                if any(kw in title for kw in neg_keywords):
                    idx += 1
                    continue

            if len(selected) < 20 and years < 3.0:
                is_exceptional = (tech >= 0.5 and prod >= 0.5)
                if not is_exceptional:
                    idx += 1
                    continue

            if len(selected) < 20 and years < 2.0:
                idx += 1
                continue

            if len(selected) < 50:
                if "business analyst" in title or "project manager" in title or "pm" == title or "ba" == title:
                    if prod <= 0.0:
                        idx += 1
                        continue

            if any(kw in title for kw in neg_keywords):
                idx += 1
                continue

            selected.append(cand)
            idx += 1
    else:
        # Without production ML/search guardrails
        selected = [c for c in ranked_candidates if c["final_score"] > 0.0][:top_n]
        if not selected:
            selected = ranked_candidates[:top_n]

    # Stable sorting to enforce standard rules
    selected.sort(key=lambda x: x["candidate_id"])
    selected.sort(key=lambda x: x["final_score"], reverse=True)

    # Build response rows
    output_rows = []
    for rank_pos, cand in enumerate(selected, start=1):
        cand_id = cand["candidate_id"]
        score = cand["final_score"]
        evidence = cand["evidence"]

        reasoning = generate_reasoning(evidence, rank_pos, cand_id)

        output_rows.append({
            "candidate_id": cand_id,
            "rank": rank_pos,
            "score": score,
            "reasoning": reasoning,
            # Preview extra columns
            "Title": evidence.get("title", ""),
            "Experience (Yrs)": evidence.get("years", 0.0),
            "Location": evidence.get("location", ""),
            "Skills": ", ".join(evidence.get("skills", []))
        })

    # Enforce strict score monotonicity
    for i in range(len(output_rows) - 1):
        if output_rows[i]["score"] < output_rows[i + 1]["score"]:
            output_rows[i + 1]["score"] = output_rows[i]["score"]

    return pd.DataFrame(output_rows)


def main():
    """
    Renders the Streamlit frontend layout and captures inputs to run ranking.
    """
    # Setup page configurations with wide layout
    st.set_page_config(
        page_title="AI Candidate Ranking System",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Premium Custom CSS Injection
    st.markdown(
        """
        <style>
        .reportview-container {
            font-family: 'Inter', sans-serif;
        }
        .hero-banner {
            background: linear-gradient(135deg, #1e1b4b, #311042);
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 25px;
            border: 1px solid #4338ca;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        }
        .hero-banner h1 {
            margin: 0 !important;
            color: #f8fafc !important;
            font-family: 'Outfit', 'Inter', sans-serif !important;
            font-size: 2.2rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.5px !important;
        }
        .hero-banner h2 {
            margin: 5px 0 15px 0 !important;
            color: #a5b4fc !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 1.25rem !important;
            font-weight: 500 !important;
            opacity: 0.9 !important;
        }
        .hero-banner p {
            margin: 0 !important;
            color: #cbd5e1 !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 0.95rem !important;
            line-height: 1.5 !important;
            max-width: 850px !important;
        }
        .card-step {
            background-color: #1e293b;
            padding: 20px;
            border-radius: 10px;
            border: 1px solid #334155;
            height: 100%;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        }
        .card-step h4 {
            margin-top: 0 !important;
            font-size: 1.05rem !important;
            font-weight: 600 !important;
        }
        .card-step p {
            margin: 0 !important;
            font-size: 0.85rem !important;
            color: #94a3b8 !important;
            line-height: 1.4 !important;
        }
        .metric-box {
            background-color: #1e293b;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #334155;
            text-align: center;
            box-shadow: 0 2px 4px rgb(0 0 0 / 0.1);
        }
        .metric-box-title {
            font-size: 0.75rem;
            color: #94a3b8;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .metric-box-value {
            font-size: 1.5rem;
            color: #3b82f6;
            font-weight: 700;
            margin-top: 5px;
        }
        .metric-box-value.pass {
            color: #10b981;
        }
        .metric-box-value.fail {
            color: #ef4444;
        }
        .download-card {
            background-color: #1e293b;
            padding: 25px;
            border-radius: 12px;
            border: 1px solid #334155;
            text-align: center;
            margin-top: 15px;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        }
        .download-card h4 {
            margin: 0 0 10px 0 !important;
            color: #f8fafc !important;
            font-size: 1.15rem !important;
        }
        .download-card p {
            margin: 0 0 20px 0 !important;
            color: #94a3b8 !important;
            font-size: 0.88rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Hero Header Rendering
    st.markdown(
        """
        <div class="hero-banner">
            <h1>Devansh Sharma &middot; Hoshi Coders</h1>
            <h2>The Data & AI Challenge | AI Candidate Ranking System</h2>
            <p>
                Upload a candidate dataset, tune ranking priorities, preview the shortlist, and export a recruiter-ready CSV — all locally.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Define column layout: control panel on left, scoring workspace on right
    control_col, workspace_col = st.columns([1, 2], gap="large")

    # Left Column - Control Panel
    with control_col:
        st.markdown("### 🎛️ Control Panel")

        # File uploads
        cand_file = st.file_uploader(
            "Upload candidate dataset (.json, .jsonl, .jsonl.gz)",
            type=["json", "jsonl", "jsonl.gz"],
            help="Supports Redrob candidate export files."
        )

        jd_file = st.file_uploader(
            "Upload job description (.txt, .md, .docx)",
            type=["txt", "md", "docx"],
            help="Used to dynamically extract scoring keywords and perform semantic matches."
        )

        out_filename = st.text_input(
            "Output filename",
            value="ranked_candidates.csv",
            help="Filename for the exported recruiter-ready CSV."
        )

        top_n_export = st.number_input(
            "Top N candidates to export",
            min_value=1,
            max_value=1000,
            value=100,
            help="Number of top ranked candidates to select for the final list."
        )

        # Sliders and controls partitioned into Expandable sections
        with st.expander("⚖️ Scoring Weights", expanded=False):
            tech_core_w = st.slider(
                "Technical Match",
                0.0, 1.0, 0.15,
                help="Evaluation of core AI/ML algorithms, fine-tuning, and information retrieval skills."
            )
            prod_evidence_w = st.slider(
                "Production ML Evidence",
                0.0, 1.0, 0.18,
                help="Prefers candidates with history of deploying machine learning models at scale."
            )
            search_retrieval_w = st.slider(
                "Search / Retrieval Experience",
                0.0, 1.0, 0.12,
                help="Weights familiarity with vector DBs (FAISS, Pinecone, Qdrant) and search indexing."
            )
            ranking_eval_w = st.slider(
                "Ranking & Evaluation Experience",
                0.0, 1.0, 0.14,
                help="Prioritizes candidate knowledge of ranking metrics (NDCG, MAP) and A/B testing."
            )
            python_engineering_w = st.slider(
                "Python / Engineering Strength",
                0.0, 1.0, 0.20,
                help="Stresses general Python expertise and engineering role experience."
            )
            startup_product_w = st.slider(
                "Startup / Product Fit",
                0.0, 1.0, 0.07,
                help="Favors scrappy, zero-to-one shippers and product-company background."
            )

        with st.expander("🛡️ Penalty Controls", expanded=False):
            trap_strength = st.slider(
                "Trap / Honeypot Penalty",
                0.0, 5.0, 1.0,
                help="Strength of penalty for discrepant profiles or suspicious keyword patterns."
            )
            non_eng_strength = st.slider(
                "Non-Engineering Title Penalty",
                0.0, 5.0, 1.0,
                help="Strength of penalty applied to non-engineering current/past roles."
            )

        with st.expander("⚙️ Recruiter Preferences", expanded=False):
            behavioral_signal_w = st.slider(
                "Behavioral Availability",
                0.0, 1.0, 0.02,
                help="Factor in platform signals like notice periods, recruiter response rates, and activity."
            )
            experience_fit_w = st.slider(
                "Experience Fit",
                0.0, 1.0, 0.03,
                help="Preference for candidates matching the target range of experience."
            )
            location_fit_w = st.slider(
                "Location Fit",
                0.0, 1.0, 0.01,
                help="Adjusts score based on willingness to relocate or proximity to preferred cities."
            )
            penalize_non_eng = st.checkbox("Penalize non-engineering titles", value=True)
            prefer_experience = st.checkbox("Prefer 5–9 years experience", value=True)
            use_behavioral = st.checkbox("Use behavioral signals if available", value=True)
            remove_traps = st.checkbox("Remove obvious honeypots/traps", value=True)
            require_prod_evidence = st.checkbox(
                "Require production ML/search/retrieval evidence for top ranks",
                value=True
            )

        # Enable advanced toggle only if artifacts match
        advanced_supported = False
        is_original_dataset = False
        temp_cand_path = None
        temp_jd_path = None

        if cand_file and jd_file:
            os.makedirs(".tmp_uploads", exist_ok=True)

            temp_cand_path = Path(".tmp_uploads") / cand_file.name
            with open(temp_cand_path, "wb") as f:
                f.write(cand_file.getbuffer())

            temp_jd_path = Path(".tmp_uploads") / jd_file.name
            with open(temp_jd_path, "wb") as f:
                f.write(jd_file.getbuffer())

            # Check if the uploaded file matches original candidate dataset
            is_original_dataset = check_dataset_matches_original(temp_cand_path)

            if FAISS_INDEX_PATH.exists() and CANDIDATE_MAP_PATH.exists() and is_original_dataset:
                advanced_supported = True

        if advanced_supported:
            use_advanced = st.checkbox("Run in Advanced Semantic Mode (FAISS dense lookup)", value=True)
        else:
            use_advanced = False

        # Main run button
        run_btn = st.button(
            "🚀 Generate Candidate Shortlist",
            type="primary",
            disabled=not (cand_file and jd_file),
            use_container_width=True
        )

    # Right Column - Workspace & Output Results
    with workspace_col:
        # Render explanation cards if no results exist in session state
        if "df_ranked" not in st.session_state:
            st.session_state.df_ranked = None
        if "diagnostics" not in st.session_state:
            st.session_state.diagnostics = None

        if st.session_state.df_ranked is None:
            st.markdown("### 📋 How It Works")
            s1, s2, s3 = st.columns(3)
            with s1:
                st.markdown(
                    """
                    <div class="card-step">
                        <h4 style="color: #3b82f6;">1. Upload Dataset</h4>
                        <p>Provide your candidates export dataset and the target job description to match skills and details.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with s2:
                st.markdown(
                    """
                    <div class="card-step">
                        <h4 style="color: #8b5cf6;">2. Tune Priorities</h4>
                        <p>Customize scoring component weights and penalty filters in the control panel to fit your team's needs.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with s3:
                st.markdown(
                    """
                    <div class="card-step">
                        <h4 style="color: #ec4899;">3. Generate Shortlist</h4>
                        <p>Run the local ranker to apply filters, view automatic diagnostics, and download the recruitment-ready CSV.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            st.info("👈 Please upload the candidate dataset and job description on the left to activate the ranking pipeline.")

        # Execution flow triggered by button click
        if run_btn:
            progress_bar = st.progress(0.0)

            def update_progress(frac):
                progress_bar.progress(frac)

            start_time = time.time()

            # Sequenced recruiter progress messaging
            status_placeholder = st.empty()
            status_placeholder.markdown("📥 **Loading dataset...**")
            time.sleep(0.3)
            status_placeholder.markdown("🔍 **Extracting candidate signals...**")
            time.sleep(0.3)
            status_placeholder.markdown("⚖️ **Applying scoring weights...**")
            time.sleep(0.3)
            status_placeholder.markdown("🛡️ **Detecting honeypots & traps...**")
            time.sleep(0.3)
            status_placeholder.markdown("📝 **Generating CSV...**")

            params = {
                "technical_core_weight": tech_core_w,
                "production_evidence_weight": prod_evidence_w,
                "ranking_eval_weight": ranking_eval_w,
                "search_retrieval_weight": search_retrieval_w,
                "python_engineering_weight": python_engineering_w,
                "startup_product_weight": startup_product_w,
                "experience_fit_weight": experience_fit_w,
                "location_fit_weight": location_fit_w,
                "behavioral_signal_weight": behavioral_signal_w,
                "trap_penalty_strength": trap_strength,
                "non_eng_title_penalty": non_eng_strength,
                "penalize_non_eng": penalize_non_eng,
                "prefer_experience": prefer_experience,
                "use_behavioral": use_behavioral,
                "remove_traps": remove_traps,
                "require_production_evidence": require_prod_evidence,
                "use_advanced": use_advanced
            }

            with st.spinner("Analyzing candidates and generating ranked shortlist..."):
                df = rank_candidates_interactive(
                    temp_cand_path,
                    temp_jd_path,
                    params,
                    top_n=top_n_export,
                    progress_callback=update_progress
                )

            progress_bar.progress(1.0)
            elapsed = time.time() - start_time
            status_placeholder.empty()

            # Compile diagnostics statistics
            total_loaded = 0
            try:
                for _ in stream_candidates(temp_cand_path):
                    total_loaded += 1
            except Exception:
                total_loaded = len(df)

            total_ranked = len(df)

            # Check candidate IDs format
            candidate_id_pattern = re.compile(r"^CAND_[0-9]{7}$")
            fake_ids_count = 0
            for cid in df["candidate_id"]:
                if not candidate_id_pattern.match(str(cid)):
                    fake_ids_count += 1

            # Check duplicate candidate IDs
            duplicates_count = len(df["candidate_id"]) - df["candidate_id"].nunique()

            # Score descending check
            is_descending = True
            scores = df["score"].tolist()
            for i in range(len(scores) - 1):
                if scores[i] < scores[i + 1]:
                    is_descending = False
                    break

            # Save outputs in session state
            st.session_state.df_ranked = df
            st.session_state.diagnostics = {
                "total_loaded": total_loaded,
                "total_ranked": total_ranked,
                "fake_ids": fake_ids_count,
                "duplicates": duplicates_count,
                "is_descending": is_descending,
                "runtime": elapsed
            }

        # Render results panels
        if st.session_state.df_ranked is not None:
            df = st.session_state.df_ranked
            diag = st.session_state.diagnostics

            st.markdown("### 📊 Shortlist Diagnostics")

            # Metric summary row
            m1, m2, m3, m4, m5, m6 = st.columns(6)

            with m1:
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-box-title">Total candidates loaded</div>
                        <div class="metric-box-value">{diag['total_loaded']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with m2:
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-box-title">Candidates ranked</div>
                        <div class="metric-box-value">{diag['total_ranked']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with m3:
                status_cls = "pass" if diag['fake_ids'] == 0 else "fail"
                status_txt = "0" if diag['fake_ids'] == 0 else f"{diag['fake_ids']}"
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-box-title">Fake IDs</div>
                        <div class="metric-box-value {status_cls}">{status_txt}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with m4:
                status_cls = "pass" if diag['duplicates'] == 0 else "fail"
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-box-title">Duplicate IDs</div>
                        <div class="metric-box-value {status_cls}">{diag['duplicates']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with m5:
                desc_status = "Pass" if diag['is_descending'] else "Fail"
                status_cls = "pass" if diag['is_descending'] else "fail"
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-box-title">Score order valid</div>
                        <div class="metric-box-value {status_cls}">{desc_status}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with m6:
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-box-title">Runtime</div>
                        <div class="metric-box-value">{diag['runtime']:.2f}s</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.write("")

            # Create tabbed workspace sections
            tab_preview, tab_diag, tab_download = st.tabs([
                "🎯 Preview Top Candidates",
                "🔎 Diagnostics Details",
                "📥 Download CSV"
            ])

            with tab_preview:
                st.markdown("#### Candidate Shortlist Preview")
                st.dataframe(
                    df[["rank", "candidate_id", "score", "reasoning"]],
                    use_container_width=True
                )

            with tab_diag:
                st.markdown("#### Validation Verification Logs")
                d1, d2 = st.columns(2)
                with d1:
                    st.write("**Data Checks**")
                    st.write(f"- Total uploaded candidates: `{diag['total_loaded']}`")
                    st.write(f"- Selected export shortlist count: `{diag['total_ranked']}`")
                    st.write(f"- Duplicate IDs detected: `{diag['duplicates']}`")
                with d2:
                    st.write("**Validation Standard Checks**")
                    st.write(f"- Zero fake candidate IDs: `{'Pass' if diag['fake_ids'] == 0 else 'Fail'}`")
                    st.write(f"- Monotonic score sorting: `{'Pass' if diag['is_descending'] else 'Fail'}`")
                    st.write(f"- Vector / Advanced Mode utilized: `{'Yes' if use_advanced else 'No'}`")

            with tab_download:
                # Format and prepare CSV for download
                csv_df = df[["candidate_id", "rank", "score", "reasoning"]]
                csv_data = csv_df.to_csv(index=False)

                st.markdown(
                    f"""
                    <div class="download-card">
                        <h4>Download recruiter-ready CSV</h4>
                        <p>The output file contains candidate_id, rank, score, and reasoning columns. Formatted perfectly for submissions.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.download_button(
                    label="⬇️ Download Ranked CSV",
                    data=csv_data,
                    file_name=out_filename,
                    mime="text/csv",
                    use_container_width=True
                )


if __name__ == "__main__":
    main()
