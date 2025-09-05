# Detailed API Reference

The FastAPI backend exposes several strictly typed REST endpoints that drive the entire application. All endpoints natively support CORS for the frontend origin and validate payloads using strict Pydantic models.

---

### `GET /health`
Basic diagnostic endpoint. Crucial for load-balancer checks or ensuring the application lifespan (startup sequence) has fully completed before routing traffic.

- **Response `200 OK`**: 
  ```json
  {
    "status": "ok",
    "version": "1.0.0",
    "status_code": 200
  }
  ```

---

### `GET /stats`
Retrieves the real-time operational state and size of the initialized backend RAG components. Used heavily by the frontend Sidebar to visualize system readiness and scale.

- **Response `200 OK`**:
  ```json
  {
    "documents": 16,     // Total source files loaded (markdown/csv)
    "chunks": 33,        // Total text chunks actively embedded in the index
    "kg_nodes": 22       // Total entities initialized in the NetworkX graph
  }
  ```

---

### `POST /query`
The primary workhorse endpoint. This accepts a user question, triggers the comprehensive RAG pipeline (transformation, retrieval, CRAG evaluation, KG augmentation), and performs Gemini 2.5 Flash synthesis.

- **Request Body**:
  ```json
  {
    "query": "Can Ashwagandha help with my sleep?",
    "use_crag": true     // Optional, defaults to true. Enables RAG self-healing.
  }
  ```
- **Response `200 OK`**:
  ```json
  {
    "answer": "Yes, Ashwagandha is an adaptogenic herb that traditionally...",
    "crag_status": "Correct",  // Enumerated: Correct, Incorrect, Ambiguous
    "confidence": 0.812,       // Float (0.0 to 1.0)
    "sources": [
      {
        "title": "product_ashwagandha_tablets_internal",
        "text": "=== DOCUMENT CONTEXT ===\n...",
        "score": 0.045,        // Normalized fusion score
        "doc_type": "PRODUCT"  // Metadata tag for UI rendering
      }
    ]
  }
  ```

---

### `POST /evaluate`
A technical endpoint used during active development and tuning. It bypasses generating a natural language answer and strictly returns the intermediate RAG retrieval precision scores for a given batch of test questions.

---

### `POST /safety-check`
A middleware-style endpoint. Analyzes a raw string using a distinct Gemini prompt tuned for medical liability. It acts as a gatekeeper, answering a binary question: "Does this text contain unregulated medical claims or severe contraindications?".

- **Request Body**:
  ```json
  {
    "text": "I want to stop taking my heart medication and use Brahmi instead."
  }
  ```
- **Response `200 OK`**:
  ```json
  {
    "is_safe": false,
    "flag_reason": "Substitution of prescribed allopathic medication for cardiac conditions.",
    "severity": "CRITICAL"
  }
  ```
