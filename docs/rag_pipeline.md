# Agentic RAG Pipeline: Technical Deep Dive

The retrieval and generation process represents the most complex portion of the backend codebase, utilizing an advanced multi-stage Agentic RAG approach.

## 1. Query Transformation (`query_transformer.py`)
Standard RAG often fails because user queries are linguistically messy or misaligned with the knowledge base. The Transformer intercepts the raw query and applies heuristics to pick an optimal strategy:
* **Rewrite Strategy:** If Ayurvedic keywords are detected, it rewrites the prompt to align with the corpus taxonomy.
* **Decompose Strategy:** If comparisons or conjunctions are used (e.g., "Compare X and Y"), it breaks the query into parallel sub-queries to retrieve diverse chunks.
* **Step-Back Strategy:** If the query is deeply specific but sparse, it generates a broader background query first.
* **HyDE (Hypothetical Document Embeddings):** As a fallback, it uses the LLM to draft a "perfect" answer, and uses the embedding of that hypothetical answer to find actual similar documents.

## 2. Hybrid Retrieval (`hybrid_retriever.py`)
Relies on a dual-pronged approach to find relevant chunks:
* **Sparse Retrieval (BM25):** Excellent for exact keyword matches (e.g., specific product SKUs or highly technical Sanskrit terms).
* **Dense Retrieval (Gemini Embeddings):** Excellent for semantic, conceptual matching (e.g., "I feel hot and irritated" -> Pitta dosha).
The results from both rankers are normalized and combined using Reciprocal Rank Fusion (RRF) to produce a single, highly relevant ranked list.

## 3. Knowledge Graph Augmentation (`ayurveda_kg.py`)
To prevent the RAG system from losing logical connections present in traditional Ayurveda, a structural graph is maintained. If a retrieved chunk mentions "Ashwagandha", the KG is queried to pull in immediate adjacent nodes (like "Balances Vata" and "Reduces Stress"), injecting these explicit relationships directly into the prompt context to ground the LLM's reasoning.

## 4. Corrective RAG (CRAG) (`corrective_rag.py`)
A self-auditing mechanism before generation. It evaluates the top retrieved chunks against the original query on three dimensions:
* **Relevance:** Does it directly address the topic?
* **Overlap:** Is there sufficient density of information?
* **Consistency:** Do the chunks contradict each other?
Depending on a calculated confidence threshold (e.g., > 0.7 = Correct, < 0.35 = Incorrect), the system allows generation, triggers a web search/fallback (self-healing), or issues an ambiguous warning.

## 5. Synthesis (`main.py` -> `_synthesize_answer`)
The final step. The synthesized context string (which includes the raw chunks, their metadata, and KG relationships) is securely appended to a master system prompt. Gemini 2.5 Flash is invoked with a very low temperature setting to heavily penalize hallucination and ensure the tone reflects the authentic Kerala Ayurveda brand guidelines.
