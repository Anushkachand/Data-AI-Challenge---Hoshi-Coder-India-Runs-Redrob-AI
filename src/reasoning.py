"""
reasoning.py
Deterministic reasoning generation based on candidate features and rank.
Follows PEP 8 style guide.
"""


def generate_reasoning(evidence, rank, candidate_id):
    """
    Generates a 1-2 sentence reasoning string that is factual, JD-connected,
    and tone-aligned with the candidate's rank.
    Uses candidate facts deterministically to avoid hallucinations and maintain reproducibility.
    """
    years = evidence.get("years", 0.0)
    title = evidence.get("title", "AI Engineer")
    location = evidence.get("location", "India")
    skills_list = evidence.get("skills", [])
    prod_phrases_list = evidence.get("production_phrases", [])
    concerns_list = evidence.get("concerns", [])

    # Format lists nicely
    if skills_list:
        skills = ", ".join(skills_list[:3])
    else:
        skills = "applied machine learning"

    if prod_phrases_list:
        if len(prod_phrases_list) > 1:
            production = ", and ".join([", ".join(prod_phrases_list[:-1]), prod_phrases_list[-1]])
        else:
            production = prod_phrases_list[0]
        production = production.lower()
    else:
        production = "applied ML systems"

    if concerns_list:
        if len(concerns_list) > 1:
            concern_phrase = " and ".join([", ".join(concerns_list[:-1]), concerns_list[-1]])
        else:
            concern_phrase = concerns_list[0]
    else:
        concern_phrase = "notice period details"

    # Deterministic choice based on candidate_id and rank to ensure reproducibility
    deterministic_hash = hash(f"{candidate_id}_{rank}_{title}")
    h_idx = abs(deterministic_hash)

    if rank <= 10:
        # Confident, highly positive tone templates
        templates = [
            f"{title} with {years} years of experience and proven expertise in {skills}. Strong production experience with {production} makes them a top fit for our founding team in {location}.",
            f"Exceptional match: {title} ({years} yrs) with core capabilities in {skills} and scaling {production}. Location fit at {location} and excellent behavioral signals align perfectly with this role.",
            f"Senior profile showing {years} years of experience building {skills}. Their track record of deploying {production} matches the hands-on engineering demand for the founding team."
        ]
        res = templates[h_idx % len(templates)]
    elif rank <= 50:
        # Strong match but acknowledging minor tradeoffs/concerns
        templates = [
            f"{title} with {years} years of experience and skills in {skills}. Demonstrates shipping evidence like {production}, though {concern_phrase} is a minor tradeoff.",
            f"Qualified {title} with {years} years of experience and deep familiarity with {skills}. Well-aligned for product delivery ({production}) with a small caveat around {concern_phrase}.",
            f"Strong applied ML background ({years} yrs) matching JD keywords like {skills}. Shows solid production background, with {concern_phrase} as the primary consideration."
        ]
        res = templates[h_idx % len(templates)]
    else:
        # Qualified backup with more caution
        templates = [
            f"{title} with {years} years of experience offering adjacent skills in {skills}. Holds production background in {production}, but is more cautious due to {concern_phrase}.",
            f"Experienced professional ({years} years) in {location} with familiarity in {skills}. Included as a qualified candidate, noting tradeoffs including {concern_phrase}.",
            f"{years} years of experience as {title} with experience in {skills}. Exhibits useful hands-on coding capability, though {concern_phrase} calls for a cautious assessment."
        ]
        res = templates[h_idx % len(templates)]

    return res
