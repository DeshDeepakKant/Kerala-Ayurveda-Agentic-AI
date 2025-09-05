"""
Data models for Kerala Ayurveda RAG System
"""
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


class DocumentType(str, Enum):
    """Document type enumeration"""
    FAQ = "faq"
    PRODUCT = "product"
    TREATMENT = "treatment"
    FOUNDATION = "foundation"
    GUIDE = "guide"
    CATALOG = "catalog"


class Chunk(BaseModel):
    """Chunk model"""
    id: str
    text: str
    doc_id: str
    doc_title: str
    doc_type: DocumentType
    section_type: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None


class Document(BaseModel):
    """Document model"""
    id: str
    title: str
    content: str
    doc_type: DocumentType
    file_path: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Retrieval result model"""
    chunk: Chunk
    score: float
    rank: int
    retrieval_method: str  # "bm25", "vector", "hybrid"


class CRAGResult(BaseModel):
    """CRAG retrieval result"""
    status: str  # "CORRECT", "AMBIGUOUS_CORRECTED", "INCORRECT_RECOVERED", "REQUIRES_HUMAN_REVIEW", "OUT_OF_SCOPE"
    documents: List[RetrievalResult]
    confidence: float
    action_taken: str
    reason: Optional[str] = None


class Citation(BaseModel):
    """Citation model"""
    citation_id: str
    document: str
    section: Optional[str] = None
    chunk_id: str
    confidence: float = 1.0


class VerificationReport(BaseModel):
    """Verification report from fact-checker agent"""
    hallucination_score: float
    citation_coverage: float
    brand_alignment_score: float
    ayurveda_term_accuracy: float
    safety_violations: int
    safety_issues: List[str] = Field(default_factory=list)
    brand_violations: List[str] = Field(default_factory=list)
    issues: List[str] = Field(default_factory=list)


class AgentState(BaseModel):
    """State object for LangGraph agents"""
    # Input
    user_brief: str
    
    # Intermediate states
    intent: Optional[Dict[str, Any]] = None
    outline: Optional[Dict[str, Any]] = None
    retrieved_docs: List[RetrievalResult] = Field(default_factory=list)
    draft: str = ""
    citations: List[Citation] = Field(default_factory=list)
    
    # Verification
    verification_report: Optional[VerificationReport] = None
    reflection_feedback: str = ""
    
    # Output
    final_content: str = ""
    
    # Metadata
    iteration: int = 0
    agent_log: List[str] = Field(default_factory=list)
    
    class Config:
        arbitrary_types_allowed = True


class QueryStrategy(str, Enum):
    """Query transformation strategies"""
    REWRITE = "rewrite"
    DECOMPOSE = "decompose"
    STEP_BACK = "step_back"
    HYDE = "hyde"
    AUTO = "auto"


class EvaluationMetrics(BaseModel):
    """Evaluation metrics"""
    # Retrieval metrics
    context_precision: float = 0.0
    context_recall: float = 0.0
    mrr: float = 0.0
    
    # Generation metrics
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    citation_accuracy: Dict[str, float] = Field(default_factory=dict)
    
    # Domain-specific metrics
    ayurveda_terminology_accuracy: float = 0.0
    contraindication_coverage: float = 0.0
    brand_voice_alignment: float = 0.0
    
    # Hallucination detection
    hallucination_scores: Dict[str, float] = Field(default_factory=dict)
    ensemble_hallucination_score: float = 0.0
    is_hallucination: bool = False
