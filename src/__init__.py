# Kerala Ayurveda RAG System
# Main source package

__version__ = "1.0.0"
__author__ = "Kerala Ayurveda RAG Team"

from .config import Config, config
from .models import (
    Document,
    DocumentType,
    Chunk,
    RetrievalResult,
    CRAGResult,
    AgentState,
    VerificationReport,
    EvaluationMetrics,
    QueryStrategy
)

__all__ = [
    "Config",
    "config",
    "Document",
    "DocumentType", 
    "Chunk",
    "RetrievalResult",
    "CRAGResult",
    "AgentState",
    "VerificationReport",
    "EvaluationMetrics",
    "QueryStrategy"
]
