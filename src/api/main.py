"""
FastAPI REST API for Kerala Ayurveda RAG System.

Endpoints:
- POST /generate - Generate content from a brief
- POST /query - Simple Q&A query
- POST /evaluate - Evaluate a response
- GET /health - Health check
- GET /stats - System statistics
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import time
import logging

from src.config import Config
from src.models import CRAGResult
from src.data_processing.document_loader import DocumentLoader
from src.data_processing.chunker import AyurvedaChunker
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.corrective_rag import CorrectiveRAG
from src.knowledge_graph.ayurveda_kg import AyurvedaKnowledgeGraph
from src.agents.langgraph_workflow import AyurvedaAgentWorkflow
from src.evaluation.metrics import RAGEvaluator

logger = logging.getLogger(__name__)


# ============ Pydantic Models ============

class GenerateRequest(BaseModel):
    """Request model for content generation."""
    brief: str = Field(..., description="Content brief or topic", min_length=10)
    content_type: str = Field(default="article", description="Type: article, faq, product_description")
    use_agents: bool = Field(default=True, description="Use multi-agent workflow")
    
    class Config:
        json_schema_extra = {
            "example": {
                "brief": "Write about the benefits of Ashwagandha for stress management",
                "content_type": "article",
                "use_agents": True
            }
        }


class QueryRequest(BaseModel):
    """Request model for simple Q&A."""
    query: str = Field(..., description="User question", min_length=5)
    use_crag: bool = Field(default=True, description="Use CRAG for retrieval")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the contraindications for Ashwagandha?",
                "use_crag": True,
                "top_k": 5
            }
        }


class SafetyCheckRequest(BaseModel):
    """Request model for safety/contraindication check."""
    herb_or_product: str = Field(..., description="Herb or product name")
    conditions: list[str] = Field(..., description="Health conditions to check")
    
    class Config:
        json_schema_extra = {
            "example": {
                "herb_or_product": "Ashwagandha",
                "conditions": ["pregnancy", "thyroid_condition"]
            }
        }


class EvaluateRequest(BaseModel):
    """Request model for response evaluation."""
    query: str = Field(..., description="Original query")
    response: str = Field(..., description="Generated response to evaluate")
    context: list[str] = Field(default=[], description="Retrieved context chunks")
    ground_truth: Optional[str] = Field(default=None, description="Ground truth answer")


class GenerateResponse(BaseModel):
    """Response model for content generation."""
    content: str
    hallucination_score: float
    brand_alignment_score: float
    citations: list[dict]
    revision_count: int
    workflow_messages: list[str]
    processing_time_seconds: float


class QueryResponse(BaseModel):
    """Response model for simple Q&A."""
    answer: str
    sources: list[dict]
    crag_status: str
    confidence: float
    processing_time_seconds: float


class SafetyResponse(BaseModel):
    """Response model for safety check."""
    safe: bool
    severity: str
    reason: str
    recommendations: list[str]


class EvaluationResponse(BaseModel):
    """Response model for evaluation."""
    overall_score: float
    pass_threshold: bool
    faithfulness: float
    answer_relevancy: float
    hallucination_score: float
    brand_alignment: float
    unsupported_claims: list[str]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    indexed_documents: int
    indexed_chunks: int
    knowledge_graph_nodes: int


# ============ Application State ============

class AppState:
    """Holds initialized components."""
    
    def __init__(self):
        self.config: Optional[Config] = None
        self.retriever: Optional[HybridRetriever] = None
        self.crag: Optional[CorrectiveRAG] = None
        self.kg: Optional[AyurvedaKnowledgeGraph] = None
        self.workflow: Optional[AyurvedaAgentWorkflow] = None
        self.evaluator: Optional[RAGEvaluator] = None
        self.chunks: list = []
        self.documents: list = []
        self.is_initialized: bool = False


state = AppState()



# ============ Helper: Gemini Answer Synthesis ============

def _synthesize_answer(query: str, context: str) -> str:
    """Use Gemini to synthesize a concise, professional answer from retrieved context."""
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold

    try:
        cfg = Config()
        genai.configure(api_key=cfg.google.api_key)
        model = genai.GenerativeModel(cfg.google.model)
        
        prompt = f"""You are a knowledgeable Ayurveda expert for Kerala Ayurveda, a premium wellness brand.
        
Using ONLY the provided context, answer the user's question in a clear, professional, and concise way.
Write in 2-4 short paragraphs using plain prose. Do not repeat the question or use document headers.
Always include a brief note on consulting a healthcare professional when relevant.

USER QUESTION:
{query}

RETRIEVED CONTEXT:
{context[:4000]}

ANSWER (plain prose, no document headers, no "=== CONTENT ===" markers):"""

        safety = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        response = model.generate_content(
            prompt,
            safety_settings=safety,
            generation_config=genai.types.GenerationConfig(temperature=0.3, max_output_tokens=600)
        )
        return response.text.strip()
    except Exception as e:
        logger.warning(f"Answer synthesis failed, falling back to raw text: {e}")
        return context[:1500]


# ============ FastAPI App ============

app = FastAPI(
    title="Kerala Ayurveda RAG API",
    description="Production-ready RAG system for Ayurveda content generation with CRAG, Knowledge Graphs, and Multi-Agent workflows.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============ Startup/Shutdown ============

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    logger.info("Initializing Kerala Ayurveda RAG system...")
    
    try:
        # Initialize config
        state.config = Config()
        
        # Initialize components (no config arg — they use global config internally)
        state.retriever = HybridRetriever()
        state.crag = CorrectiveRAG(state.retriever)
        state.kg = AyurvedaKnowledgeGraph()
        state.evaluator = RAGEvaluator(state.config)
        
        # Load and index documents
        loader = DocumentLoader()
        chunker = AyurvedaChunker()
        
        state.documents = loader.load_all_documents()
        
        for doc in state.documents:
            chunks = chunker.chunk_document(doc)
            state.chunks.extend(chunks)
        
        # Build indexes
        state.retriever.index_chunks(state.chunks)
        
        # Initialize workflow
        state.workflow = AyurvedaAgentWorkflow(
            state.config, state.retriever, state.crag, state.kg
        )
        
        state.is_initialized = True
        logger.info(f"System initialized: {len(state.documents)} docs, {len(state.chunks)} chunks")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        state.is_initialized = False


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Kerala Ayurveda RAG system...")


# ============ Endpoints ============

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health and return stats."""
    if not state.is_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return HealthResponse(
        status="healthy",
        indexed_documents=len(state.documents),
        indexed_chunks=len(state.chunks),
        knowledge_graph_nodes=state.kg.graph.number_of_nodes()
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate_content(request: GenerateRequest):
    """
    Generate content using the multi-agent workflow.
    
    This is the main content generation endpoint that:
    1. Understands the brief
    2. Creates an outline with KG enhancement
    3. Writes content with CRAG retrieval
    4. Verifies facts and reduces hallucinations
    5. Applies brand voice
    """
    if not state.is_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    start_time = time.time()
    
    try:
        if request.use_agents and state.workflow:
            # Use full agent workflow
            result = state.workflow.run(
                query=request.brief,
                content_type=request.content_type
            )
            
            return GenerateResponse(
                content=result.get("final_output", result.get("draft", "")),
                hallucination_score=result.get("hallucination_score", 0),
                brand_alignment_score=result.get("brand_alignment_score", 0),
                citations=result.get("citations", []),
                revision_count=result.get("revision_count", 0),
                workflow_messages=result.get("workflow_messages", []),
                processing_time_seconds=round(time.time() - start_time, 2)
            )
        else:
            # Simple CRAG-based generation
            crag_result = state.crag.retrieve_with_correction(request.brief, k=5)
            
            # Format simple response
            context = "\n\n".join([doc.chunk.text for doc in crag_result.documents])
            
            return GenerateResponse(
                content=f"Based on retrieved information:\n\n{context}",
                hallucination_score=0.0,
                brand_alignment_score=0.0,
                citations=[{"source": doc.chunk.metadata.get("source", "Unknown")} for doc in crag_result.documents],
                revision_count=0,
                workflow_messages=[f"CRAG status: {crag_result.status}"],
                processing_time_seconds=round(time.time() - start_time, 2)
            )
            
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Simple Q&A query against the knowledge base.
    
    Uses CRAG for self-healing retrieval and returns
    relevant chunks with confidence scores.
    """
    if not state.is_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    start_time = time.time()
    
    try:
        if request.use_crag:
            result = state.crag.retrieve_with_correction(request.query, k=request.top_k)
            
            sources = [
                {
                    "text": doc.chunk.text[:500],
                    "source": doc.chunk.metadata.get("source", "Unknown"),
                    "score": round(doc.score, 3),
                    "doc_type": doc.chunk.metadata.get("doc_type", "Unknown")
                }
                for doc in result.documents
            ]
            
            context = "\n\n---\n\n".join([doc.chunk.text for doc in result.documents[:5]])
            answer = _synthesize_answer(request.query, context)
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                crag_status=result.status,
                confidence=result.confidence,
                processing_time_seconds=round(time.time() - start_time, 2)
            )
        else:
            # Direct retrieval without CRAG
            results = state.retriever.search(request.query, k=request.top_k)
            
            sources = [
                {
                    "text": r.chunk.text[:500],
                    "source": r.chunk.metadata.get("source", "Unknown"),
                    "score": round(r.score, 3)
                }
                for r in results
            ]
            
            context = "\n\n---\n\n".join([r.chunk.text for r in results[:5]])
            answer = _synthesize_answer(request.query, context)
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                crag_status="DIRECT_RETRIEVAL",
                confidence=1.0,
                processing_time_seconds=round(time.time() - start_time, 2)
            )
            
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/safety-check", response_model=SafetyResponse)
async def check_safety(request: SafetyCheckRequest):
    """
    Check contraindications for an herb or product.
    
    Uses the Knowledge Graph to identify safety concerns
    based on specified health conditions.
    """
    if not state.is_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        result = state.kg.check_contraindication(
            request.herb_or_product,
            request.conditions
        )
        
        recommendations = []
        if not result.get("safe", True):
            recommendations.append("Consult with a healthcare provider before use")
            recommendations.append("Review product contraindications carefully")
            if "pregnancy" in [c.lower() for c in request.conditions]:
                recommendations.append("Seek advice from your OB/GYN")
        
        return SafetyResponse(
            safe=result.get("safe", True),
            severity=result.get("severity", "LOW"),
            reason=result.get("reason", "No known contraindications found"),
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Safety check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_response(request: EvaluateRequest):
    """
    Evaluate a generated response for quality.
    
    Returns RAGAS-style metrics including faithfulness,
    relevancy, hallucination score, and brand alignment.
    """
    if not state.is_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        result = state.evaluator.evaluate(
            query=request.query,
            response=request.response,
            context=request.context,
            ground_truth=request.ground_truth
        )
        
        return EvaluationResponse(
            overall_score=result.overall_score,
            pass_threshold=result.pass_threshold,
            faithfulness=result.faithfulness,
            answer_relevancy=result.answer_relevancy,
            hallucination_score=result.hallucination_score,
            brand_alignment=result.brand_alignment,
            unsupported_claims=result.unsupported_claims
        )
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get detailed system statistics."""
    if not state.is_initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    # Document type distribution
    doc_types = {}
    for doc in state.documents:
        dtype = doc.doc_type.value if hasattr(doc.doc_type, 'value') else str(doc.doc_type)
        doc_types[dtype] = doc_types.get(dtype, 0) + 1
    
    # Chunk type distribution
    chunk_types = {}
    for chunk in state.chunks:
        ctype = chunk.metadata.get("section_type", "unknown")
        chunk_types[ctype] = chunk_types.get(ctype, 0) + 1
    
    return {
        "documents": {
            "total": len(state.documents),
            "by_type": doc_types
        },
        "chunks": {
            "total": len(state.chunks),
            "by_type": chunk_types
        },
        "knowledge_graph": {
            "nodes": state.kg.graph.number_of_nodes(),
            "edges": state.kg.graph.number_of_edges()
        },
        "config": {
            "model": state.config.google.model,
            "embedding_model": state.config.google.embedding_model,
            "crag_thresholds": {
                "high": state.config.crag.high_confidence_threshold,
                "medium": state.config.crag.medium_confidence_threshold
            }
        }
    }


# ============ Factory Function ============

def create_api_app() -> FastAPI:
    """Create and return the FastAPI app."""
    return app


# ============ Run Directly ============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
