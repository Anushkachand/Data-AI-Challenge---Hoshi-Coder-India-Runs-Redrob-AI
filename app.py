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

# Setup page configuration
st.set_page_config(
    page_title="AI Candidate Ranking System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

        # Weights
        tech_core_w = params.get("technical_core_weight", 0.15)
        prod_evidence_w = params.get("production_evidence_weight", 0.18)
        ranking_eval_w = params.get("ranking_eval_weight", 0.14)
        vector_search_w = params.get("vector_search_weight", 0.12)
        embedding_retrieval_w = params.get("embedding_retrieval_weight", 0.06)
        python_engineering_w = params.get("python_engineering_weight", 0.20)
        experience_fit_w = params.get("experience_fit_weight", 0.03)
        location_fit_w = params.get("location_fit_weight", 0.01)
        behavioral_signal_w = params.get("behavioral_signal_weight", 0.02)

        trap_penalty_strength = params.get("trap_penalty_strength", 1.0)
        non_eng_title_penalty = params.get("non_eng_title_penalty", 1.0)

        # Composite base score formula
        base_score = (
            python_engineering_w * eng_role_score +
            prod_evidence_w * prod_score +
            ranking_eval_w * eval_score +
            vector_search_w * vector_score +
            embedding_retrieval_w * embed_score +
            tech_core_w * features.get("technical_core_score", 0.0) +
            0.05 * python_score +
            0.12 * dense_score +
            0.04 * startup_score +
            0.03 * prod_company_score +
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


# UI Layout and Rendering
st.markdown(
    """
    <style>
    .reportview-container {
        font-family: 'Inter', sans-serif;
    }
    .main-title {
        background-color: #0f172a;
        padding: 24px;
        border-radius: 12px;
        margin-bottom: 25px;
        text-align: center;
        border-left: 6px solid #3b82f6;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
    }
    .main-title h3 {
        margin: 0 !important;
        color: #f8fafc !important;
        font-size: 1.35rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        border: 1px solid #334155;
    }
    .metric-card-title {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 4px;
        font-weight: 500;
        text-transform: uppercase;
    }
    .metric-card-value {
        font-size: 1.5rem;
        color: #f8fafc;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="main-title">
        <h3>Devansh Sharma &middot; Hoshi Coders | The Data & AI Challenge | AI Candidate Ranking System</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# Define column layout: parameters panel on left, scoring output on right
left_col, right_col = st.columns([1, 2])

# Left Column - File Uploads and Parameter controls
with left_col:
    st.subheader("1. Upload Section")

    # Candidate dataset upload
    cand_file = st.file_uploader(
        "Upload candidate dataset (.json, .jsonl, .jsonl.gz)",
        type=["json", "jsonl", "jsonl.gz"]
    )

    # Job description upload
    jd_file = st.file_uploader(
        "Upload job description (.txt, .md, .docx)",
        type=["txt", "md", "docx"]
    )

    out_filename = st.text_input(
        "Output filename",
        value="ranked_candidates.csv"
    )

    st.subheader("2. Ranking Parameters")

    # Subscore weight sliders
    with st.expander("Scoring Component Weights", expanded=True):
        tech_core_w = st.slider("Technical Core Weight", 0.0, 1.0, 0.15)
        prod_evidence_w = st.slider("Production Evidence Weight", 0.0, 1.0, 0.18)
        ranking_eval_w = st.slider("Ranking / Evaluation Weight", 0.0, 1.0, 0.14)
        vector_search_w = st.slider("Vector Search Weight", 0.0, 1.0, 0.12)
        embedding_retrieval_w = st.slider("Embedding / Retrieval Weight", 0.0, 1.0, 0.06)
        python_engineering_w = st.slider("Python / Engineering Weight", 0.0, 1.0, 0.20)
        experience_fit_w = st.slider("Experience Fit Weight", 0.0, 1.0, 0.03)
        location_fit_w = st.slider("Location Fit Weight", 0.0, 1.0, 0.01)
        behavioral_signal_w = st.slider("Behavioral Signal Weight", 0.0, 1.0, 0.02)

    # Penalty & Configuration Sliders
    with st.expander("Penalty Adjustments", expanded=False):
        trap_strength = st.slider("Trap Penalty Strength", 0.0, 5.0, 1.0)
        non_eng_strength = st.slider("Non-Engineering Title Penalty", 0.0, 5.0, 1.0)

    top_n_export = st.number_input(
        "Top N candidates to export",
        min_value=1,
        max_value=1000,
        value=100
    )

    # Heuristic rules selection checkboxes
    st.subheader("Rule Checkboxes")
    penalize_non_eng = st.checkbox("Penalize non-engineering titles", value=True)
    prefer_experience = st.checkbox("Prefer 5–9 years experience", value=True)
    use_behavioral = st.checkbox("Use behavioral signals if available", value=True)
    remove_traps = st.checkbox("Remove obvious honeypots/traps", value=True)
    require_prod_evidence = st.checkbox(
        "Require production ML/search/retrieval evidence for top ranks",
        value=True
    )

# Right Column - Output table, Diagnostics and Actions
with right_col:
    st.subheader("3. Execution & Preview")

    # Enable advanced mode toggle only if artifacts match
    advanced_supported = False
    uploaded_path_str = None
    is_original_dataset = False

    if cand_file and jd_file:
        # Create temp dir and save files
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

    # Mode notification
    if advanced_supported:
        use_advanced = st.checkbox("Run in Advanced Semantic Mode (FAISS dense lookup)", value=True)
    else:
        st.info("Structured scoring mode enabled. Advanced Semantic Mode requires the original dataset and FAISS index artifacts.")
        use_advanced = False

    run_btn = st.button("Run Candidate Ranking 🚀", type="primary", disabled=not (cand_file and jd_file))

    # Initialize session state for storage
    if "df_ranked" not in st.session_state:
        st.session_state.df_ranked = None
    if "diagnostics" not in st.session_state:
        st.session_state.diagnostics = None

    if run_btn:
        progress_bar = st.progress(0.0)

        def update_progress(frac):
            progress_bar.progress(frac)

        start_time = time.time()

        params = {
            "technical_core_weight": tech_core_w,
            "production_evidence_weight": prod_evidence_w,
            "ranking_eval_weight": ranking_eval_w,
            "vector_search_weight": vector_search_w,
            "embedding_retrieval_weight": embedding_retrieval_w,
            "python_engineering_weight": python_engineering_w,
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

        with st.spinner("Processing candidate profiles and compiling features..."):
            df = rank_candidates_interactive(
                temp_cand_path,
                temp_jd_path,
                params,
                top_n=top_n_export,
                progress_callback=update_progress
            )

        progress_bar.progress(1.0)
        elapsed = time.time() - start_time

        # Calculate diagnostics details
        # Check total loaded candidates
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

        # Save results in streamlit session state
        st.session_state.df_ranked = df
        st.session_state.diagnostics = {
            "total_loaded": total_loaded,
            "total_ranked": total_ranked,
            "fake_ids": fake_ids_count,
            "duplicates": duplicates_count,
            "is_descending": is_descending,
            "runtime": elapsed
        }

        st.success(f"Ranking complete! Processed {total_ranked} qualified candidates in {elapsed:.2f} seconds.")

    # Render Preview and Diagnostics if df exists in session state
    if st.session_state.df_ranked is not None:
        df = st.session_state.df_ranked
        diag = st.session_state.diagnostics

        st.subheader("4. Diagnostics Summary")
        diag_cols = st.columns(6)

        with diag_cols[0]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-title">Loaded</div>
                    <div class="metric-card-value">{diag['total_loaded']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with diag_cols[1]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-title">Ranked</div>
                    <div class="metric-card-value">{diag['total_ranked']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with diag_cols[2]:
            status_text = "Yes" if diag['fake_ids'] == 0 else f"No ({diag['fake_ids']})"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-title">Zero Fake IDs</div>
                    <div class="metric-card-value">{status_text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with diag_cols[3]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-title">Duplicates</div>
                    <div class="metric-card-value">{diag['duplicates']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with diag_cols[4]:
            desc_status = "Pass" if diag['is_descending'] else "Fail"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-title">Monotonic Score</div>
                    <div class="metric-card-value">{desc_status}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with diag_cols[5]:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-title">Runtime</div>
                    <div class="metric-card-value">{diag['runtime']:.2f}s</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.subheader("5. Top Ranked Candidates Preview")
        # Display full candidate preview table
        st.dataframe(
            df[[
                "rank", "candidate_id", "score", "Title",
                "Experience (Yrs)", "Location", "Skills", "reasoning"
            ]],
            use_container_width=True
        )

        st.subheader("6. Download Options")

        # Compile CSV with only required columns
        csv_df = df[["candidate_id", "rank", "score", "reasoning"]]
        csv_data = csv_df.to_csv(index=False)

        st.download_button(
            label=f"Download {out_filename} 💾",
            data=csv_data,
            file_name=out_filename,
            mime="text/csv"
        )
