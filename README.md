# SHL Assessment Recommendation API

An AI-powered RAG (Retrieval-Augmented Generation) system that recommends the most relevant SHL assessments based on a provided Job Description (JD). This project solves the challenge of matching complex job requirements to a vast catalog of behavioral and technical assessments.

---

##  Features
* Hybrid Search Engine:** Combines semantic vector search (FAISS) with keyword-based matching (BM25) to ensure both conceptual and technical accuracy.
* MMR Reranking:** Implements *Maximal Marginal Relevance* to ensure the top recommendations are diverse, covering a mix of technical skills, personality, and aptitude.
* Seniority Intelligence:** Specifically tuned to recognize seniority levels (e.g., Executive, Lead, Junior) and recommend appropriate leadership assessments like the OPQ32.
* Chain-of-Thought Reranking:** Uses GPT-4o to analyze the JD context and finalize the top 5 recommendations based on strategic fit.

---

##  Dataset & Preprocessing
To meet the requirement of "Scraping, Parsing, and Storing" the SHL Product Catalog, the following pipeline was built:

* Data Parsing:** A custom script (`clean_data.py`) processes the raw catalog to standardize labels and fix taxonomy errors.
* Feature Engineering:** Merged `test_name`, `description`, and `test_type` to create a rich semantic context for the embedding model.
* Vector Storage:** Used OpenAI’s `text-embedding-3-small` (1536 dimensions) to store vectorized descriptions in a FAISS index.
* Data Integrity:** Performed deduplication and handled missing metadata to prevent "hallucinations" during the recommendation phase.



---

## Tech Stack
* Framework:** FastAPI (Python)
* AI Models:** OpenAI GPT-4o (Reasoning) & text-embedding-3-small (Embeddings)
* Vector DB:** FAISS (Facebook AI Similarity Search)
* Retrieval:** Rank-BM25 & Hybrid Retrieval Logic
* Deployment:** Render

---

##  API Specification (Appendix 2)
**Endpoint:** `POST /recommend`  
**Content-Type:** `application/json`

### Sample Request:
```json
{
  "text": "Senior Java Developer with experience in AWS and team management."
}