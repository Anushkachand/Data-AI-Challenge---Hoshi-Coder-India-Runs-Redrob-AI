"""
scoring.py
Scoring functions for different candidate profile attributes.
Follows PEP 8 style guide.
"""

import datetime
from src.config import (
    CORE_AI_SKILLS, VECTOR_DB_SKILLS, EMBEDDING_RETRIEVAL_SKILLS,
    EVAL_FRAMEWORK_SKILLS, PRODUCTION_KEYWORDS, SERVICES_COMPANIES,
    STARTUP_KEYWORDS, CURRENT_DATE
)


def calculate_technical_core_score(skills, career_history):
    """
    Computes a score based on presence of core AI/ML skills
    in skill profile and career history descriptions.
    """
    career_desc_str = " ".join([h.get("description", "").lower() for h in career_history])
    career_titles_str = " ".join([h.get("title", "").lower() for h in career_history])
    full_career_str = career_desc_str + " " + career_titles_str

    skill_matches = 0
    for s in skills:
        name = s.get("name", "").lower()
        for core in CORE_AI_SKILLS:
            if core in name:
                prof = s.get("proficiency", "beginner")
                weight = {
                    "expert": 1.0,
                    "advanced": 0.8,
                    "intermediate": 0.5,
                    "beginner": 0.2
                }.get(prof, 0.2)
                
                # Check if it is supported by career history
                is_supported = core in full_career_str
                if not is_supported:
                    weight *= 0.1  # penalize isolated skill keywords
                
                skill_matches += weight
                break

    career_matches = 0
    for h in career_history:
        desc = h.get("description", "").lower()
        title = h.get("title", "").lower()
        for core in CORE_AI_SKILLS:
            if core in desc:
                career_matches += 0.5
            if core in title:
                career_matches += 1.0

    raw_score = skill_matches * 1.5 + career_matches
    # Soft normalize using 15.0 as denominator
    return min(1.0, raw_score / 15.0)


def calculate_vector_search_score(skills, career_history):
    """
    Computes search and vector database proficiency.
    """
    career_desc_str = " ".join([h.get("description", "").lower() for h in career_history])
    career_titles_str = " ".join([h.get("title", "").lower() for h in career_history])
    full_career_str = career_desc_str + " " + career_titles_str

    skill_matches = 0
    for s in skills:
        name = s.get("name", "").lower()
        for db in VECTOR_DB_SKILLS:
            if db in name:
                prof = s.get("proficiency", "beginner")
                weight = {
                    "expert": 1.0,
                    "advanced": 0.8,
                    "intermediate": 0.5,
                    "beginner": 0.2
                }.get(prof, 0.2)
                
                is_supported = db in full_career_str
                if not is_supported:
                    weight *= 0.1
                
                skill_matches += weight
                break

    career_matches = 0
    has_career_match = False
    for h in career_history:
        desc = h.get("description", "").lower()
        title = h.get("title", "").lower()
        for db in VECTOR_DB_SKILLS:
            if db in desc:
                career_matches += 0.5
                has_career_match = True
            if db in title:
                career_matches += 1.0
                has_career_match = True

    raw_score = skill_matches * 1.5 + career_matches
    score = min(1.0, raw_score / 5.0)
    
    # "vector_search_score only when supported by career descriptions"
    if not has_career_match:
        score *= 0.1
        
    return score


def calculate_embedding_retrieval_score(skills, career_history):
    """
    Computes embedding-based retrieval model proficiency.
    """
    career_desc_str = " ".join([h.get("description", "").lower() for h in career_history])
    career_titles_str = " ".join([h.get("title", "").lower() for h in career_history])
    full_career_str = career_desc_str + " " + career_titles_str

    skill_matches = 0
    for s in skills:
        name = s.get("name", "").lower()
        for model in EMBEDDING_RETRIEVAL_SKILLS:
            if model in name:
                prof = s.get("proficiency", "beginner")
                weight = {
                    "expert": 1.0,
                    "advanced": 0.8,
                    "intermediate": 0.5,
                    "beginner": 0.2
                }.get(prof, 0.2)
                
                is_supported = model in full_career_str
                if not is_supported:
                    weight *= 0.1
                
                skill_matches += weight
                break

    career_matches = 0
    for h in career_history:
        desc = h.get("description", "").lower()
        title = h.get("title", "").lower()
        for model in EMBEDDING_RETRIEVAL_SKILLS:
            if model in desc:
                career_matches += 0.5
            if model in title:
                career_matches += 1.0

    raw_score = skill_matches * 1.5 + career_matches
    return min(1.0, raw_score / 5.0)


def calculate_ranking_eval_score(skills, career_history):
    """
    Computes model evaluation and metrics proficiency (NDCG, MAP, MRR, A/B).
    """
    career_desc_str = " ".join([h.get("description", "").lower() for h in career_history])
    career_titles_str = " ".join([h.get("title", "").lower() for h in career_history])
    full_career_str = career_desc_str + " " + career_titles_str

    skill_matches = 0
    for s in skills:
        name = s.get("name", "").lower()
        for ev in EVAL_FRAMEWORK_SKILLS:
            if ev in name:
                prof = s.get("proficiency", "beginner")
                weight = {
                    "expert": 1.0,
                    "advanced": 0.8,
                    "intermediate": 0.5,
                    "beginner": 0.2
                }.get(prof, 0.2)
                
                is_supported = ev in full_career_str
                if not is_supported:
                    weight *= 0.1
                
                skill_matches += weight
                break

    career_matches = 0
    for h in career_history:
        desc = h.get("description", "").lower()
        title = h.get("title", "").lower()
        for ev in EVAL_FRAMEWORK_SKILLS:
            if ev in desc:
                career_matches += 0.5
            if ev in title:
                career_matches += 1.0

    raw_score = skill_matches * 1.5 + career_matches
    return min(1.0, raw_score / 5.0)


def calculate_python_score(skills, career_history):
    """
    Computes Python expertise based on skill and career context.
    """
    python_in_skills = False
    skill_weight = 0.0
    for s in skills:
        name = s.get("name", "").lower()
        if "python" in name:
            python_in_skills = True
            prof = s.get("proficiency", "beginner")
            skill_weight = {
                "expert": 1.0,
                "advanced": 0.9,
                "intermediate": 0.7,
                "beginner": 0.4
            }.get(prof, 0.4)
            break

    desc_matches = 0
    for h in career_history:
        desc = h.get("description", "").lower()
        if any(py in desc for py in ["python", "pyspark", "numpy", "pandas"]):
            desc_matches += 1

    if python_in_skills:
        return min(1.0, skill_weight + desc_matches * 0.1)
    elif desc_matches > 0:
        return min(1.0, 0.5 + desc_matches * 0.1)
    else:
        return 0.1


def calculate_production_evidence_score(summary, career_history):
    """
    Identifies production deployment evidence.
    Evidence is only valid if a production/deployment term and an ML-related term
    appear within the same career history entry or summary.
    """
    valid_evidence_count = 0
    prod_terms = ["production", "deploy", "deployed", "shipped", "infrastructure", "scale", "serving", "inference", "ab test", "a/b test"]
    ml_terms = ["ml", "machine learning", "search", "retrieval", "ranking", "recommender", "recommendation", "embeddings", "vector", "nlp", "faiss", "model", "ndcg", "mrr", "map", "eval", "fine-tuning", "lora", "qdrant", "weaviate", "pinecone", "milvus"]

    for h in career_history:
        desc_lower = h.get("description", "").lower()
        title_lower = h.get("title", "").lower()
        full_text = desc_lower + " " + title_lower
        
        has_prod = any(pt in full_text for pt in prod_terms)
        has_ml = any(mt in full_text for mt in ml_terms)
        
        if has_prod and has_ml:
            valid_evidence_count += 1
            
    summary_lower = summary.lower()
    has_summary_prod = any(pt in summary_lower for pt in prod_terms)
    has_summary_ml = any(mt in summary_lower for mt in ml_terms)
    if has_summary_prod and has_summary_ml:
        valid_evidence_count += 0.5

    return min(1.0, valid_evidence_count / 2.0)


def calculate_product_company_score(career_history):
    """
    Computes product company score by checking the fraction of non-services companies.
    """
    if not career_history:
        return 0.5
    product_companies = 0
    total_companies = len(career_history)
    for h in career_history:
        company = h.get("company", "").lower()
        # If it is not in the services company blacklist, increment
        if not any(serv in company for serv in SERVICES_COMPANIES):
            product_companies += 1
    return product_companies / total_companies


def calculate_startup_shipper_score(summary, career_history):
    """
    Computes alignment with early-stage shipping attributes.
    """
    text = (summary + " " + " ".join([h.get("description", "").lower() for h in career_history])).lower()
    matches = 0
    for kw in STARTUP_KEYWORDS:
        if kw in text:
            matches += 1
    return min(1.0, matches / 4.0)


def calculate_experience_fit_score(years_of_experience):
    """
    Scores alignment with target experience brackets:
    - 5-9 years: 1.0 (highest score)
    - 3-5 years: 0.5 (qualify with good evidence)
    - other: 0.1
    """
    years = years_of_experience
    if 5.0 <= years <= 9.0:
        return 1.0
    elif 3.0 <= years < 5.0:
        return 0.5
    else:
        return 0.1


def calculate_location_fit_score(location, country, willing_to_relocate, preferred_work_mode):
    """
    Preferred location: Pune/Noida (1.0).
    Tier-1 Indian cities: Hyderabad/Bangalore/Mumbai/Chennai (0.8).
    Willing to relocate or flexible mode: (0.6).
    Otherwise: (0.0 - 0.4).
    """
    loc_lower = location.lower()
    country_lower = country.lower()

    # Preferred location
    if any(pref in loc_lower for pref in ["pune", "noida", "delhi ncr", "gurgaon", "gurugram"]):
        return 1.0
    # Tier 1
    if any(t1 in loc_lower for t1 in ["hyderabad", "mumbai", "bangalore", "bengaluru", "chennai"]):
        return 0.8

    is_india = "india" in country_lower or any(
        city in loc_lower for city in [
            "pune", "noida", "delhi", "gurgaon", "hyderabad", "mumbai",
            "bangalore", "bengaluru", "chennai", "kolkata", "bhubaneswar",
            "ahmedabad", "jaipur"
        ]
    )

    if is_india:
        if willing_to_relocate or preferred_work_mode in ["hybrid", "flexible"]:
            return 0.6
        return 0.4
    else:
        if willing_to_relocate:
            return 0.3
        return 0.0


def calculate_behavioral_signal_score(redrob_signals):
    """
    Aggregates active platform engagement signals into a normalized [0, 1] score.
    """
    score = 0.5  # Neutral baseline

    # Stated availability
    if redrob_signals.get("open_to_work_flag") is True:
        score += 0.10
    else:
        score -= 0.10

    # Last active date
    last_active_str = redrob_signals.get("last_active_date")
    if last_active_str:
        try:
            last_active = datetime.datetime.strptime(last_active_str, "%Y-%m-%d").date()
            delta = (CURRENT_DATE - last_active).days
            if delta <= 60:
                score += 0.15
            elif delta > 180:
                score -= 0.15
        except Exception:
            pass

    # Recruiter response rate
    response_rate = redrob_signals.get("recruiter_response_rate", 0.0)
    if response_rate >= 0.5:
        score += 0.10
    elif response_rate < 0.10:
        score -= 0.15

    # Average response time
    avg_resp_time = redrob_signals.get("avg_response_time_hours", 200.0)
    if avg_resp_time <= 48.0:
        score += 0.10
    elif avg_resp_time > 120.0:
        score -= 0.15

    # Interview completion rate
    int_completion = redrob_signals.get("interview_completion_rate", 0.0)
    if int_completion >= 0.75:
        score += 0.10
    elif int_completion < 0.40:
        score -= 0.15

    # GitHub contributions
    gh_score = redrob_signals.get("github_activity_score", -1.0)
    if gh_score > 20.0:
        score += 0.10
    elif gh_score == -1.0:
        score -= 0.05

    # Saved by recruiters
    saved_count = redrob_signals.get("saved_by_recruiters_30d", 0)
    if saved_count > 5:
        score += 0.10

    # Verification checks
    if redrob_signals.get("verified_email") is True:
        score += 0.05
    if redrob_signals.get("verified_phone") is True:
        score += 0.05
    if redrob_signals.get("linkedin_connected") is True:
        score += 0.05

    # Relocation
    if redrob_signals.get("willing_to_relocate") is True:
        score += 0.05

    # Notice period
    notice = redrob_signals.get("notice_period_days", 90)
    if notice <= 30:
        score += 0.10
    elif notice > 90:
        score -= 0.10

    return max(0.0, min(1.0, score))


def calculate_engineering_role_score(title, career_history, skills, summary):
    """
    Grades alignment of candidate title and history with engineering/ML roles.
    Strongly penalizes disqualified non-technical profiles.
    """
    title_lower = title.lower()

    # Positive targets
    pos_titles = [
        "ai engineer", "machine learning engineer", "ml engineer", "software engineer",
        "backend engineer", "search engineer", "ranking engineer", "recommendation systems engineer",
        "data scientist", "nlp engineer", "applied scientist", "full stack developer",
        "full-stack developer"
    ]

    # Score current title
    score = 0.5  # Neutral baseline

    # Check positive matches
    is_positive = False
    for pos in pos_titles:
        if pos in title_lower:
            is_positive = True
            if pos in ["full stack developer", "full-stack developer"]:
                # Only if ML/search/retrieval evidence exists
                text = (
                    summary + " " +
                    " ".join([h.get("description", "") for h in career_history]) + " " +
                    " ".join([s.get("name", "") for s in skills])
                ).lower()
                if any(kw in text for kw in ["ml", "machine learning", "search", "retrieval", "ranking", "recommender", "recommendation", "nlp", "embeddings"]):
                    score = 0.8
                else:
                    score = 0.3
            elif any(ai_term in pos for ai_term in ["ai", "machine learning", "ml", "nlp", "applied scientist", "ranking", "recommendation", "search"]):
                score = 1.0
            else:
                score = 0.8
            break

    # If it falls into non-technical categories, it receives low scores or 0.0
    from src.config import DISQUALIFIED_TITLES
    
    # If the current title contains any disqualified title, it's NOT positive
    for neg in DISQUALIFIED_TITLES:
        if neg in title_lower:
            is_positive = False
            if neg in ["project manager", "business analyst", "product manager"]:
                # Check if hands-on ML engineering evidence exists in descriptions
                text = (summary + " " + " ".join([h.get("description", "") for h in career_history])).lower()
                if any(kw in text for kw in ["ml", "machine learning", "search", "retrieval", "ranking"]):
                    score = 0.3
                else:
                    score = 0.0
            else:
                score = 0.0
            break

    # Also check if career history contains business analyst, project manager, or other disqualified roles
    # and adjust the score down
    for h in career_history:
        h_title = h.get("title", "").lower()
        for neg in DISQUALIFIED_TITLES:
            if neg in h_title:
                score *= 0.8
                break

    return score
