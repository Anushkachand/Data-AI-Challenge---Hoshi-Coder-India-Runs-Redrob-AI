"""
config.py
Configuration and scoring constants for the Redrob candidate ranker.
Follows PEP 8 style guide.
"""

import datetime
from pathlib import Path

# Workspace Directory Structure
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Target role details
JOB_DESCRIPTION_PATH = DATA_DIR / "job_description.docx"
REDROB_SIGNALS_DOC_PATH = DATA_DIR / "redrob_signals_doc.docx"

# Precomputed files paths
FAISS_INDEX_PATH = ARTIFACTS_DIR / "candidate_index.faiss"
JD_EMBEDDING_PATH = ARTIFACTS_DIR / "jd_embedding.npy"
CANDIDATE_MAP_PATH = ARTIFACTS_DIR / "candidate_id_map.json"
PRECOMPUTED_FEATURES_PATH = ARTIFACTS_DIR / "candidate_features.jsonl.gz"
PRECOMPUTE_MANIFEST_PATH = ARTIFACTS_DIR / "precompute_manifest.json"

# Reference current date for availability calculations (June 23, 2026)
CURRENT_DATE = datetime.date(2026, 6, 23)

# Model configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Scoring weights in final_score aggregation formula
# Must sum to 1.0
WEIGHTS = {
    "engineering_role_score": 0.20,
    "production_evidence_score": 0.18,
    "ranking_eval_score": 0.14,
    "vector_search_score": 0.12,
    "dense_score": 0.12,
    "embedding_retrieval_score": 0.06,
    "python_score": 0.05,
    "startup_shipper_score": 0.04,
    "product_company_score": 0.03,
    "experience_fit_score": 0.03,
    "behavioral_signal_score": 0.02,
    "location_fit_score": 0.01,
}

# Core search concepts from Job Description (JD)
CORE_AI_SKILLS = [
    "embeddings", "semantic search", "vector search", "retrieval",
    "ranking", "recommendation systems", "search infrastructure",
    "hybrid search", "bm25", "learning to rank", "xgboost ranking",
    "lightgbm ranking", "llm systems", "fine-tuning", "lora",
    "qlora", "peft", "nlp", "information retrieval"
]

VECTOR_DB_SKILLS = [
    "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "elasticsearch", "opensearch", "vector database", "vectordb"
]

EMBEDDING_RETRIEVAL_SKILLS = [
    "sentence-transformers", "openai embeddings", "bge", "e5",
    "embeddings", "dense retrieval", "neural search"
]

EVAL_FRAMEWORK_SKILLS = [
    "ndcg", "mrr", "map", "a/b testing", "offline evaluation",
    "online evaluation", "evaluation framework", "recommender evaluation",
    "retrieval quality", "offline-to-online correlation"
]

PYTHON_KEYWORDS = ["python", "pyspark", "numpy", "pandas", "scipy"]

# Startup Shipping Terms
STARTUP_KEYWORDS = [
    "startup", "founding", "owned end-to-end", "scrappy", "shipped",
    "early stage", "series a", "series b", "seed", "zero to one"
]

# Production Evidence Terms
PRODUCTION_KEYWORDS = [
    "production", "real users", "deploy", "deployed", "scale",
    "scalability", "infrastructure", "high throughput", "low latency",
    "ab test", "a/b testing", "shipped", "serving", "inference"
]

# Disqualified roles / fields (Strong Negative Signals)
DISQUALIFIED_TITLES = [
    "marketing", "seo", "content writing", "content writer", "operations",
    "accounting", "accountant", "civil engineering", "civil engineer",
    "mechanical engineering", "mechanical engineer", "customer support",
    "hr", "hr manager", "sales", "sales executive", "business analyst",
    "finance", "recruiter", "designer", "graphic designer"
]

# Excluded keywords for AI experience verification
AI_CURIO_KEYWORDS = [
    "curious about ai", "experimented with chatgpt", "chatgpt productivity",
    "langchain tutorial", "openai demo", "artificial intelligence enthusiast",
    "prompt engineering for productivity", "chatgpt wrapper"
]

# IT services companies with poor alignment history (if no other product ML exp)
SERVICES_COMPANIES = [
    "tcs", "tata consultancy services", "wipro", "infosys", "accenture",
    "cognizant", "capgemini", "hcltech", "hcl technologies", "ltts",
    "tech mahindra", "mindtree", "l&t infotech"
]

# Location preference tiers
LOCATION_TIERS = {
    "preferred": ["pune", "noida", "delhi ncr", "gurgaon", "gurugram"],
    "tier_1": ["hyderabad", "mumbai", "bangalore", "bengaluru", "chennai"]
}
