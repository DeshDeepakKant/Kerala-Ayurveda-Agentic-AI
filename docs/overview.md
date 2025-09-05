# Kerala Ayurveda Agentic AI - Comprehensive Overview

## Project Background & Context
The **Kerala Ayurveda Agentic AI** is a highly specialized digital assistant designed to serve experts, practitioners, and wellness seekers by providing precise, safe, and authentic answers regarding Ayurvedic practices, products, and principles. 

Unlike generalized LLM applications, this system is strictly grounded in a curated corpus of Kerala Ayurveda knowledge (stored locally as Markdown and CSV files). It utilizes a sophisticated **Agentic Retrieval-Augmented Generation (RAG)** architecture to ensure that every response is both factually accurate according to the corpus and tonally appropriate.

## Core Capabilities
1. **Multi-Strategy Query Transformation:** Vagueness in user queries is a common challenge. The system intelligently rewrites, decomposes, or performs "step-back" abstractions on user queries to maximize retrieval accuracy.
2. **Hybrid & Graph-Augmented Retrieval:** By combining traditional BM25 keyword matching with dense vector embeddings and a semantic Knowledge Graph, the system guarantees comprehensive context retrieval, even for complex Ayurvedic concepts.
3. **Corrective RAG (CRAG) Self-Healing:** Before any answer is generated, the retrieved context is evaluated for relevance and consistency. If the context is deemed poor, the system will attempt to "heal" the query or gracefully fallback to a safe, domain-restricted response.
4. **Google Gemini Synthesis:** The final generation step employs Google's `Gemini 2.5 Flash` model, strictly prompted to synthesize the retrieved context into a warm, professional, and authentic Ayurvedic tone.
5. **Real-time Metrics:** The system exposes internal processing metrics (CRAG confidence scores, chunk counts, graph nodes) directly to the user interface, ensuring complete transparency in how an answer was derived.

## Intended Audience
- **Practitioners:** Seeking quick references on contraindications, dosha affinities, and herbal formulations.
- **Wellness Seekers:** Looking for authentic guidance on self-care routines, diet, and lifestyle changes based on Ayurvedic principles.
- **Developers / Researchers:** Exploring advanced RAG methodologies, corrective evaluation, and agentic workflows in a specialized domain.
