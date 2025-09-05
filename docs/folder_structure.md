# Project Folder Structure & File Explanation

The project is thoughtfully organized into several key directories, separating the frontend application, backend API, data sources, and core algorithmic modules.

## Root Directory

- **`pyproject.toml` / `uv.lock`**: Python dependency management files. This project uses `uv` for fast, reproducible environment builds.
- **`run_api.py`**: The main entry point script to boot the FastAPI backend server on Uvicorn.
- **`.env`**: Secure environment variables (API keys, model selection).
- **`README.md`**: High-level documentation and setup instructions.
- **`LICENSE`**: MIT license for the project.

---

## `/scripts` Directory (Utilities & CLI)

- **`main_pipeline.py`**: A CLI demonstration script that runs the RAG pipeline directly in the terminal without starting the web server.
- **`run_evaluation.py`**: A script used for testing and scoring the quality of the RAG pipeline outputs against predefined baselines.
- **`setup.sh`**: Helper script to initialize the environment and install dependencies.

---

## `/src` Directory (Core Backend Logic)

The absolute core of the backend AI logic resides here.

- **`api/`**: Contains the FastAPI application routing and endpoints (`main.py`). It defines how HTTP requests are translated into pipeline executions.
- **`data_processing/`**: 
  - `document_loader.py`: Handles reading the raw Markdown/CSV files and parsing them into unified Document objects.
  - `chunker.py`: Splits large documents into smaller, semantically meaningful text chunks suitable for embedding and retrieval.
- **`knowledge_graph/`**:
  - `ayurveda_kg.py`: Implements a networkx-based graph that maps relationships between herbs, doshas, ailments, and products (e.g., "Brahmi" -> "soothes" -> "Vata").
- **`retrieval/`**:
  - `query_transformer.py`: Agents that rewrite, decompose, or apply HyDE strategies to user queries.
  - `hybrid_retriever.py`: Combines BM25 sparse search with Gemini dense vector embeddings to rank and retrieve document chunks.
  - `corrective_rag.py`: Evaluates the retrieved chunks against the query to assign a confidence score and a "Correct", "Incorrect", or "Ambiguous" status.
- **`agents/`**: Contains any sub-agent logic required for multi-step reasoning.
- **`evaluation/`**: Contains classes for automated benchmarking of response quality.
- **`models.py`**: Defines Pydantic data models used across the application to ensure static typing and clear data schemas.
- **`config.py`**: Centralized configuration management, securely loading environment variables like API keys from `.env`.

---

## `/frontend` Directory (React UI)

A modern, fast web interface powered by Vite and React.

- **`index.html`**: The main HTML shell.
- **`vite.config.js`**: Build configuration, including the proxy setup that routes `/api` calls to the FastAPI backend running on port 8000.
- **`src/`**:
  - `App.jsx`: The root React component orchestrating the layout and application state.
  - `index.css`: The global stylesheet implementing the dark-themed, glassmorphic design system and custom UI components (like the rich source cards).
  - **`components/`**: Modular UI elements:
    - `Sidebar.jsx`: Displays metrics, settings, and example queries.
    - `QueryPanel.jsx`: The main search input interface.
    - `ResultsPanel.jsx`: The comprehensive component that renders the synthesized answer, CRAG metrics, and the heavily stylized, expandable retrieved source cards.

---

## `/data` Directory (Knowledge Base)

- **`raw/`**: The "ground truth" corpus. Contains Markdown files detailing Ayurvedic foundations, specific products, treatment plans, and a CSV product catalog.
- **`indexes/`**: A generated directory holding serialized models (like the BM25 index pickle file) to speed up server restarts.
- **`processed/`**: Cached or intermediate data generated during the pipeline.

---

## `/assets` & `/docs`

- **`assets/screenshots/`**: Visual documentation of the UI and system results.
- **`docs/`**: Comprehensive system documentation (Architecture, API, RAG Pipeline).
- **`docs/archive/`**: Legacy documentation like the original `PROJECT_EXPLANATION.md`.
