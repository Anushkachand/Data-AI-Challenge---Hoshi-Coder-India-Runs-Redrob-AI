# Redrob AI Candidate Ranker: Architectural Design & Approach

## Overview

The Redrob Hackathon Candidate Ranker utilizes a hybrid retrieval, structured scoring, and trap-aware reranking system to solve the problem of selecting the top 100 candidates from a large pool of 100,000 candidates for a **Senior AI Engineer — Founding Team** role.

In web-scale recruitment, pure semantic search or embedding-based retrieval fails because:
1. **Keyword Stuffing**: Low-quality candidates copy-paste technical buzzwords into their skill sections or summaries.
2. **Synthetic Honeypots**: The dataset includes around 80 honeypots with impossible values (e.g., salary min > max, 0-month duration for expert skills) that would rank highly in a simple dense vector search.
3. **Availability & Engagement**: A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% response rate is, for hiring purposes, not actually available.

To address these challenges, we built a pipeline consisting of two phases:
1. **Precomputation Pipeline (`precompute.py`)**: Runs offline without execution time limits. It builds a dense semantic FAISS index for candidates, processes the JD document, and extracts compact structured feature scores.
2. **Ranking Pipeline (`rank.py`)**: A resource-constrained, fast execution script ($\le$ 5 minutes CPU-only, network-free) that recalls the top 10,000 matches from the FAISS index, computes the final hybrid composite score, handles trap penalties, and formats the top 100 ranked output.

---

## 1. Retrieval & Dense Match

During precomputation, candidate profiles are compiled into structured rich text documents highlighting:
- Stated current title, headline, summary, and years of experience.
- Detailed career history titles and descriptions (which carry more weight than simple skills).
- Education details (degrees, fields, and institution tiers).
- Listed skills with proficiency levels.

This text is embedded using `sentence-transformers/all-MiniLM-L6-v2` and indexed into a **FAISS Inner Product (Cosine Similarity) index** after L2 normalization. 

During ranking, the normalized Job Description embedding is queried against the FAISS index to retrieve the top 10,000 matches. The raw FAISS cosine similarity is normalized into a `dense_score` in the $[0.0, 1.0]$ range.

---

## 2. Structured Feature Scoring

To complement dense semantic search, we evaluate 11 distinct structured signals for each candidate:

1. **Technical Core Score**: Match against role-specific target terms in both skills (weighted by proficiency) and career descriptions (prioritizing title matches).
2. **Vector Search Score**: Matches on vector databases and search engines (FAISS, Pinecone, Qdrant, Milvus, Weaviate, OpenSearch, Elastic).
3. **Embedding Retrieval Score**: Matches on embedding architectures and retrievers (sentence-transformers, OpenAI embeddings, BGE, E5).
4. **Ranking Eval Score**: Focuses on search evaluation metrics and frameworks (NDCG, MAP, MRR, offline-online evaluations, A/B testing).
5. **Python Score**: Assesses python coding depth based on skill proficiency and project application.
6. **Production Evidence Score**: Scans career history descriptions for shipping phrases ("production scale", "deployed to real users", "low latency", "high throughput").
7. **Product-Company Score**: Graded score based on the history of product-company vs service/consulting companies (blacklisting TCS, Wipro, Infosys, Cognizant, etc.).
8. **Startup Shipper Score**: Keywords associated with scrappy, founding-team style shipping ("seed", "series a", "zero to one").
9. **Experience Fit Score**: Graded score matching the 5-9 year JD target:
   - 6–8 years: `1.0` (optimal)
   - 5–6 / 8–9 years: `0.9`
   - 4–5 / 9–11 years: `0.65`
   - 3–4 years: `0.35`
   - 11+ years: `0.45`
   - <3 years: `0.15`
10. **Location Fit Score**: Target-based scoring mapping preferred cities (Pune/Noida/Delhi NCR) to `1.0`, Tier-1 cities (Hyderabad/Bangalore/Chennai) to `0.8`, and relocation willingness/hybrid mode to `0.6`.
11. **Behavioral Signal Score**: Aggregates 23 platform activity signals (signups, logins, response rates, response speed, connection count, phone/email verification, notice period days).

---

## 3. Trap & Honeypot Detection

We implement a multi-layered verification system to detect and heavily penalize trap profiles:

- **Stated Expert/Advanced skill with 0 months duration**: Marked as a hard honeypot (penalty = 1.0).
- **Impossible Salary bounds (Max < Min)**: Marked as a hard honeypot (penalty = 1.0).
- **Years of Experience Mismatch**: Stated profile experience deviating from calculated career duration by > 2.5 years (penalty = 1.0).
- **Low Skill Assessment Scores**: Claims of expert/advanced core skills where Redrob assessment score is low (penalty = 0.05 - 0.10).
- **Keyword Stuffers**: High concentration of AI terms in listed skills but zero references to production, ML systems, or ranking in career descriptions (penalty = 0.20).
- **Plain-Language False Positives**: Summaries referencing AI curiosity or ChatGPT productivity without any underlying professional engineering experience (penalty = 0.25).
- **IT Services Only**: Entire career at services firms without product-company exposure (penalty = 0.20).

---

## 4. Hybrid Reranking Formula

Candidates are scored using a linear combination of dense and structured metrics, offset by trap penalties:

$$
\begin{aligned}
\text{final\_score} = & 0.18 \times \text{dense\_score} + 0.16 \times \text{technical\_core\_score} + 0.13 \times \text{production\_evidence\_score} \\
& + 0.12 \times \text{ranking\_eval\_score} + 0.10 \times \text{vector\_search\_score} + 0.08 \times \text{embedding\_retrieval\_score} \\
& + 0.07 \times \text{python\_score} + 0.06 \times \text{startup\_shipper\_score} + 0.04 \times \text{product\_company\_score} \\
& + 0.04 \times \text{experience\_fit\_score} + 0.04 \times \text{behavioral\_signal\_score} + 0.04 \times \text{location\_fit\_score} \\
& - \text{trap\_risk\_penalty} - \text{disqualifier\_penalty}
\end{aligned}
$$

The final score is clamped between $[0.0, 1.0]$. 

Candidates are sorted according to:
1. `final_score` (descending)
2. `technical_core_score` (descending)
3. `production_evidence_score` (descending)
4. `ranking_eval_score` (descending)
5. `behavioral_signal_score` (descending)
6. `candidate_id` (ascending, deterministic tie-breaker)

---

## 5. Factual Reasoning Generation

To meet Stage 4 manual reviews without calling hosted LLM APIs during ranking, we employ a deterministic, fact-driven template generator. The reasons:
- Mention only explicit candidate facts (current title, experience years, location, skills, production keywords).
- Align in tone with rank:
  - Ranks 1-10: Confident fit.
  - Ranks 11-50: Strong fit acknowledging minor tradeoffs (e.g. notice period).
  - Ranks 51-100: Cautious recommendation with identified concerns.
- Vary using a deterministic hash of the candidate ID to prevent repetition.
- Strictly adhere to a maximum length of two sentences.
