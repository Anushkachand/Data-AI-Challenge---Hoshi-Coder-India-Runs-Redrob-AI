"""
precompute_features.py
Functions to process candidate profiles and compile their structured features.
Follows PEP 8 style guide.
"""

from src.scoring import (
    calculate_technical_core_score,
    calculate_vector_search_score,
    calculate_embedding_retrieval_score,
    calculate_ranking_eval_score,
    calculate_python_score,
    calculate_production_evidence_score,
    calculate_product_company_score,
    calculate_startup_shipper_score,
    calculate_experience_fit_score,
    calculate_location_fit_score,
    calculate_behavioral_signal_score,
    calculate_engineering_role_score
)
from src.trap_detection import detect_traps
from src.config import (
    CORE_AI_SKILLS, VECTOR_DB_SKILLS, EMBEDDING_RETRIEVAL_SKILLS,
    EVAL_FRAMEWORK_SKILLS, PRODUCTION_KEYWORDS, STARTUP_KEYWORDS
)


def extract_features(candidate):
    """
    Extracts all subscores, penalties, and compiles the evidence dictionary
    for a single candidate record.
    """
    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    history = candidate.get("career_history", [])
    signals = candidate.get("redrob_signals", {})

    # Extract raw properties
    cand_id = candidate.get("candidate_id")
    years = profile.get("years_of_experience", 0.0)
    title = profile.get("current_title", "")
    location = f"{profile.get('location', '')}, {profile.get('country', '')}"
    willing_reloc = signals.get("willing_to_relocate", False)
    work_mode = signals.get("preferred_work_mode", "")
    summary = profile.get("summary", "")

    # Calculate subscores
    tech_score = calculate_technical_core_score(skills, history)
    vector_score = calculate_vector_search_score(skills, history)
    embed_score = calculate_embedding_retrieval_score(skills, history)
    eval_score = calculate_ranking_eval_score(skills, history)
    python_score = calculate_python_score(skills, history)
    prod_score = calculate_production_evidence_score(summary, history)
    prod_company_score = calculate_product_company_score(history)
    startup_score = calculate_startup_shipper_score(summary, history)
    exp_fit = calculate_experience_fit_score(years)
    loc_fit = calculate_location_fit_score(
        profile.get("location", ""),
        profile.get("country", ""),
        willing_reloc,
        work_mode
    )
    behavior_score = calculate_behavioral_signal_score(signals)
    role_score = calculate_engineering_role_score(title, history, skills, summary)

    # Qualified candidate gate and tier calculations
    def is_engineering_title(t):
        t_clean = t.lower()
        pos_words = ["engineer", "developer", "programmer", "scientist", "architect", "nlp", "retrieval", "search", "ranking", "recommender", "recommendation"]
        neg_words = ["civil", "mechanical", "graphic designer", "hr", "recruiter", "sales", "marketing", "accountant", "accounting", "support", "customer support", "operations", "business analyst", "project manager", "product manager", "analyst"]
        has_pos = any(pw in t_clean for pw in pos_words)
        has_neg = any(nw in t_clean for nw in neg_words)
        return has_pos and not has_neg

    current_title = title
    career_titles = [h.get("title", "") for h in history]
    has_any_eng = is_engineering_title(current_title) or any(is_engineering_title(t) for t in career_titles)
    
    has_production_ml = (prod_score > 0.0)
    is_qualified = has_any_eng or has_production_ml

    if not is_qualified:
        tier = 5
    elif has_any_eng and has_production_ml:
        tier = 1
    elif has_any_eng and (tech_score >= 0.3 or vector_score > 0.0 or embed_score > 0.0 or eval_score > 0.0) and (behavior_score >= 0.4):
        tier = 2
    elif tech_score > 0.0 or vector_score > 0.0 or embed_score > 0.0 or eval_score > 0.0 or prod_score > 0.0:
        tier = 3
    else:
        tier = 4

    # Calculate penalties
    trap_risk, disqualifier = detect_traps(candidate)

    # Compile evidence details for reasoning engine
    # A. Relevant skills
    rel_skills_set = set(
        CORE_AI_SKILLS + VECTOR_DB_SKILLS +
        EMBEDDING_RETRIEVAL_SKILLS + EVAL_FRAMEWORK_SKILLS
    )
    matching_skills = []
    for s in skills:
        sname = s.get("name", "")
        if sname.lower() in rel_skills_set:
            matching_skills.append(sname)
    # Deduplicate and limit to top 4 skills
    matching_skills = list(dict.fromkeys(matching_skills))[:4]

    # B. Match production evidence phrases
    prod_phrases = []
    combined_text = (summary + " " + " ".join([h.get("description", "") for h in history])).lower()
    for kw in (PRODUCTION_KEYWORDS + STARTUP_KEYWORDS):
        if kw in combined_text:
            # Capitalize word for neat formatting in reasoning
            prod_phrases.append(kw.title())
    prod_phrases = list(dict.fromkeys(prod_phrases))[:3]

    # C. Track concerns
    concerns = []
    notice = signals.get("notice_period_days", 0)
    if notice > 90:
        concerns.append(f"notice period of {notice} days")

    resp_rate = signals.get("recruiter_response_rate", 1.0)
    if resp_rate < 0.15:
        concerns.append("low response rate")

    sal = signals.get("expected_salary_range_inr_lpa", {})
    if sal.get("max", 0) < sal.get("min", 0):
        concerns.append("salary requirements inconsistency")

    for s in skills:
        if s.get("proficiency") in ["expert", "advanced"] and s.get("duration_months", 0) == 0:
            concerns.append(f"implausible duration for skill {s.get('name')}")

    # Build final feature row
    feature_row = {
        "candidate_id": cand_id,
        "engineering_role_score": round(role_score, 4),
        "technical_core_score": round(tech_score, 4),
        "embedding_retrieval_score": round(embed_score, 4),
        "vector_search_score": round(vector_score, 4),
        "ranking_eval_score": round(eval_score, 4),
        "python_score": round(python_score, 4),
        "production_evidence_score": round(prod_score, 4),
        "product_company_score": round(prod_company_score, 4),
        "startup_shipper_score": round(startup_score, 4),
        "experience_fit_score": round(exp_fit, 4),
        "location_fit_score": round(loc_fit, 4),
        "behavioral_signal_score": round(behavior_score, 4),
        "trap_risk_penalty": round(trap_risk, 4),
        "disqualifier_penalty": round(disqualifier, 4),
        "tier": tier,
        "is_qualified": is_qualified,
        "evidence": {
            "years": round(years, 1),
            "title": title,
            "location": location,
            "skills": matching_skills,
            "production_phrases": prod_phrases,
            "concerns": concerns
        }
    }

    return feature_row
