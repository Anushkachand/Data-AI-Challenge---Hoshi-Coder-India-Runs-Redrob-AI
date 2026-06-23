"""
text_utils.py
Text utilities for candidate text representation and lexical matching.
Follows PEP 8 style guide.
"""

import re


def clean_text(text):
    """
    Cleans raw text for keyword searching and model inputs.
    """
    if not text:
        return ""
    # Normalize whitespaces
    return " ".join(text.split()).strip()


def build_candidate_rich_text(candidate):
    """
    Constructs a rich text description of a candidate to feed into sentence-transformers.
    Gives appropriate weights to career history descriptions.
    """
    profile = candidate.get("profile", {})
    title = clean_text(profile.get("current_title", ""))
    headline = clean_text(profile.get("headline", ""))
    summary = clean_text(profile.get("summary", ""))
    years_exp = profile.get("years_of_experience", 0.0)
    location = clean_text(profile.get("location", ""))
    country = clean_text(profile.get("country", ""))
    industry = clean_text(profile.get("current_industry", ""))
    company = clean_text(profile.get("current_company", ""))

    # Format skills
    skills_list = []
    for s in candidate.get("skills", []):
        name = clean_text(s.get("name", ""))
        prof = s.get("proficiency", "")
        dur = s.get("duration_months", 0)
        skills_list.append(f"{name} ({prof}, {dur}m)")
    skills_str = ", ".join(skills_list)

    # Format career history
    career_list = []
    for h in candidate.get("career_history", []):
        c_title = clean_text(h.get("title", ""))
        c_comp = clean_text(h.get("company", ""))
        c_dur = h.get("duration_months", 0)
        c_desc = clean_text(h.get("description", ""))
        career_list.append(
            f"- Role: {c_title} at {c_comp} ({c_dur} months)\n"
            f"  Description: {c_desc}"
        )
    career_str = "\n".join(career_list)

    # Format education
    edu_list = []
    for e in candidate.get("education", []):
        inst = clean_text(e.get("institution", ""))
        deg = clean_text(e.get("degree", ""))
        field = clean_text(e.get("field_of_study", ""))
        tier = e.get("tier", "unknown")
        edu_list.append(f"{deg} in {field} from {inst} ({tier})")
    edu_str = "; ".join(edu_list)

    # Combine into a single structured rich text block
    rich_text = (
        f"Title: {title}\n"
        f"Headline: {headline}\n"
        f"Experience: {years_exp} years\n"
        f"Location: {location}, {country}\n"
        f"Industry: {industry} | Company: {company}\n"
        f"Skills: {skills_str}\n"
        f"Education: {edu_str}\n"
        f"Summary: {summary}\n"
        f"Career History:\n{career_str}"
    )

    return rich_text


def count_keyword_matches(text, keywords):
    """
    Counts how many of the target keywords/phrases appear in the text.
    Uses regex matching for whole words and exact phrase matches.
    """
    if not text:
        return 0

    text_lower = text.lower()
    matches = 0

    for kw in keywords:
        kw_lower = kw.lower()
        # Escaping regex characters in keywords
        pattern = r"\b" + re.escape(kw_lower) + r"\b"
        # Find all occurrences
        found = re.findall(pattern, text_lower)
        matches += len(found)

    return matches
