"""
trap_detection.py
Heuristics to identify candidate profile traps, keyword stuffers, and honeypots.
Follows PEP 8 style guide.
"""

from src.config import (
    SERVICES_COMPANIES, AI_CURIO_KEYWORDS, CORE_AI_SKILLS,
    EVAL_FRAMEWORK_SKILLS, PRODUCTION_KEYWORDS, DISQUALIFIED_TITLES,
    VECTOR_DB_SKILLS
)


def detect_traps(candidate):
    """
    Analyzes candidate profile for honeypots, keyword stuffing, and plain-language traps.
    Returns:
        trap_risk_penalty (float): Penalty for suspicious keyword patterns or discrepancies.
        disqualifier_penalty (float): Penalty for poor profile matches (e.g., services only, CV-only).
    """
    trap_risk_penalty = 0.0
    disqualifier_penalty = 0.0

    profile = candidate.get("profile", {})
    skills = candidate.get("skills", [])
    history = candidate.get("career_history", [])
    signals = candidate.get("redrob_signals", {})

    years_exp = profile.get("years_of_experience", 0.0)
    title = profile.get("current_title", "").lower()
    summary = profile.get("summary", "").lower()

    # Create joint text for keyword searches
    skills_names = [s.get("name", "").lower() for s in skills]
    skills_str = " ".join(skills_names)
    career_desc_str = " ".join([h.get("description", "").lower() for h in history])
    career_titles_str = " ".join([h.get("title", "").lower() for h in history])

    # 1. HARD HONEYPOT DETECTION (Penalty = 1.0 -> Auto-disqualify from Top 100)
    is_honeypot = False
    honeypot_reasons = []

    # A. Stated expert/advanced skill with 0 months duration
    for s in skills:
        if s.get("proficiency") in ["expert", "advanced"] and s.get("duration_months", 0) == 0:
            is_honeypot = True
            honeypot_reasons.append(f"Skill {s.get('name')} is {s.get('proficiency')} with 0 months")

    # B. Expected salary min is greater than max
    sal = signals.get("expected_salary_range_inr_lpa", {})
    if sal.get("max", 0.0) < sal.get("min", 0.0):
        is_honeypot = True
        honeypot_reasons.append(f"Salary Max ({sal.get('max')}) < Min ({sal.get('min')})")

    # C. Major experience discrepancy (stated experience vs summed career history months)
    total_months = sum(h.get("duration_months", 0) for h in history)
    calc_years = total_months / 12.0
    if abs(years_exp - calc_years) > 2.5:
        is_honeypot = True
        honeypot_reasons.append(f"Experience mismatch: Profile {years_exp} vs calculated {calc_years:.2f}")

    if is_honeypot:
        # Hard penalty to completely exclude from top 100
        trap_risk_penalty += 1.0

    # 2. KEYWORD STUFFER DETECTION
    # High concentration of AI terms in skills but zero production ML context in history
    has_ai_skills = any(kw in skills_str for kw in CORE_AI_SKILLS)
    has_prod_evidence = any(kw in career_desc_str for kw in PRODUCTION_KEYWORDS)

    if has_ai_skills and not has_prod_evidence:
        # Strong penalty for lack of actual shipping experience despite claiming skills
        trap_risk_penalty += 0.20

    # 3. PLAIN-LANGUAGE FALSE POSITIVES
    # Checks if candidate states "curious about AI", "using ChatGPT", etc. but has no professional AI engineering
    has_curiosity_keywords = any(kw in summary for kw in AI_CURIO_KEYWORDS)
    # Check if they have zero core AI terms in their career history descriptions
    has_real_ai_history = any(kw in career_desc_str for kw in CORE_AI_SKILLS)

    if has_curiosity_keywords and not has_real_ai_history:
        disqualifier_penalty += 0.25

    # 4. IT SERVICES ONLY DISQUALIFIER
    # Entire career at TCS/Wipro/Infosys/etc. with no product company experience
    if history:
        all_services = True
        for h in history:
            company_name = h.get("company", "").lower()
            if not any(serv in company_name for serv in SERVICES_COMPANIES):
                all_services = False
                break
        if all_services:
            # We apply a penalty for pure services-company background (except if they are early-career/exceptional)
            disqualifier_penalty += 0.20

    # 5. DOMAIN MISMATCH (CV/Speech/Robotics without NLP/Retrieval/Ranking)
    cv_keywords = ["computer vision", "image classification", "object detection", "yolo", "opencv", "robotics", "speech recognition", "tts"]
    has_cv = any(kw in skills_str or kw in career_desc_str for kw in cv_keywords)
    has_retrieval_ranking = any(kw in skills_str or kw in career_desc_str for kw in ["retrieval", "ranking", "search", "recommender", "recommendation", "embeddings", "faiss"])

    if has_cv and not has_retrieval_ranking:
        disqualifier_penalty += 0.15

    # 6. MANAGEMENT ONLY (No longer writing code)
    mgmt_titles = ["manager", "director", "head of", "lead", "vp", "chief", "cto", "architect"]
    is_mgmt = any(m_t in title for m_t in mgmt_titles)
    # If they are currently lead/manager/director and summary implies hands-off / no coding
    if is_mgmt and "architect" not in title and years_exp > 10.0:
        # Check if they mention hands-on coding
        coding_terms = ["code", "python", "hands-on", "develop", "implement"]
        has_coding = any(term in summary or term in career_desc_str for term in coding_terms)
        if not has_coding:
            disqualifier_penalty += 0.15

    # 7. LOW ASSESSMENT SCORES FOR CORE SKILLS
    assessment_scores = signals.get("skill_assessment_scores", {})
    for s in skills:
        sname = s.get("name", "")
        prof = s.get("proficiency", "")
        if sname in assessment_scores:
            score = assessment_scores[sname]
            if prof == "expert" and score < 50:
                trap_risk_penalty += 0.10
            elif prof == "advanced" and score < 40:
                trap_risk_penalty += 0.05

    # 8. DISQUALIFIED ROLE / CURRENT TITLE CHECK (using engineering role score)
    from src.scoring import calculate_engineering_role_score, calculate_production_evidence_score, calculate_technical_core_score
    role_score = calculate_engineering_role_score(profile.get("current_title", ""), history, skills, summary)
    if role_score == 0.0:
        disqualifier_penalty += 0.50
    elif role_score <= 0.1:
        disqualifier_penalty += 0.45
    elif role_score <= 0.5:
        disqualifier_penalty += 0.25

    # 9. ISOLATED SKILL KEYWORDS CHECK
    core_libs = ["pinecone", "qdrant", "faiss", "milvus", "weaviate", "nlp", "semantic search", "learning to rank", "embeddings"]
    has_isolated_skill = False
    for sname in skills_names:
        if any(lib in sname for lib in core_libs):
            lib_clean = sname.split()[0]
            if lib_clean not in career_desc_str and lib_clean not in career_titles_str:
                has_isolated_skill = True
                break
    if has_isolated_skill:
        trap_risk_penalty += 0.50

    # 9B. GENERIC ML MENTION WITHOUT SPECIFIC SKILLS
    specific_keywords = ["search", "retrieval", "ranking", "recommender", "recommendation", "nlp", "faiss", "pinecone", "qdrant", "weaviate", "milvus", "elasticsearch", "opensearch", "ndcg", "mrr", "map", "embeddings"]
    has_specific = any(kw in skills_str or kw in career_desc_str for kw in specific_keywords)
    has_generic = any(kw in skills_str or kw in career_desc_str or kw in summary for kw in ["machine learning", "applied machine learning", "ai", "artificial intelligence", "data science"])
    if has_generic and not has_specific:
        disqualifier_penalty += 0.30

    # 10. MINIMUM SENIORITY GUARDRAILS
    tech_score = calculate_technical_core_score(skills, history)
    prod_score = calculate_production_evidence_score(summary, history)
    if years_exp < 3.0:
        if not (tech_score >= 0.5 and prod_score >= 0.5):
            # No exceptional evidence
            disqualifier_penalty += 0.30
        else:
            disqualifier_penalty += 0.10
    if years_exp < 2.0:
        disqualifier_penalty += 0.45

    # 11. TITLE-PROFILE MISMATCH PENALTY
    is_non_tech = any(disq in title for disq in DISQUALIFIED_TITLES)
    ai_skills_count = sum(1 for sname in skills_names if any(core in sname for core in CORE_AI_SKILLS + VECTOR_DB_SKILLS))
    if is_non_tech and ai_skills_count >= 3:
        if prod_score < 0.5:
            trap_risk_penalty += 0.35

    return trap_risk_penalty, disqualifier_penalty
