"""
Configuration management for Kerala Ayurveda RAG System
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
INDEXES_DIR = PROJECT_ROOT / "data" / "indexes"


class OpenAIConfig(BaseModel):
    """OpenAI API configuration"""
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = Field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"))
    embedding_model: str = Field(default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"))
    temperature: float = 0.1
    max_tokens: int = 2000


class GoogleConfig(BaseModel):
    """Google Gemini API configuration"""
    api_key: str = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    model: str = Field(default_factory=lambda: os.getenv("GOOGLE_MODEL", "gemini-1.5-flash"))
    embedding_model: str = Field(default_factory=lambda: os.getenv("GOOGLE_EMBEDDING_MODEL", "models/text-embedding-004"))
    temperature: float = 0.1
    max_tokens: int = 2000


class AnthropicConfig(BaseModel):
    """Anthropic API configuration"""
    api_key: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    model: str = "claude-3-5-sonnet-20241022"
    temperature: float = 0.1
    max_tokens: int = 2000


class VectorStoreConfig(BaseModel):
    """Vector store configuration"""
    persist_directory: Path = Field(default_factory=lambda: Path(os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/indexes/chroma")))
    collection_name: str = "kerala_ayurveda"
    dimension: int = Field(default_factory=lambda: int(os.getenv("VECTOR_DIMENSION", "3072")))


class BM25Config(BaseModel):
    """BM25 configuration"""
    k1: float = Field(default_factory=lambda: float(os.getenv("BM25_K1", "1.5")))
    b: float = Field(default_factory=lambda: float(os.getenv("BM25_B", "0.75")))


class RetrievalConfig(BaseModel):
    """Retrieval configuration"""
    top_k: int = Field(default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "10")))
    final_top_k: int = Field(default_factory=lambda: int(os.getenv("FINAL_TOP_K", "5")))
    rrf_constant: int = Field(default_factory=lambda: int(os.getenv("RRF_CONSTANT", "60")))


class CRAGConfig(BaseModel):
    """CRAG configuration"""
    high_confidence_threshold: float = Field(default_factory=lambda: float(os.getenv("CRAG_HIGH_CONFIDENCE_THRESHOLD", "0.7")))
    low_confidence_threshold: float = Field(default_factory=lambda: float(os.getenv("CRAG_LOW_CONFIDENCE_THRESHOLD", "0.3")))
    max_query_rewrites: int = Field(default_factory=lambda: int(os.getenv("MAX_QUERY_REWRITES", "3")))


class AgentConfig(BaseModel):
    """Agent configuration"""
    max_reflection_iterations: int = Field(default_factory=lambda: int(os.getenv("MAX_REFLECTION_ITERATIONS", "3")))
    hallucination_threshold: float = Field(default_factory=lambda: float(os.getenv("HALLUCINATION_THRESHOLD", "0.10")))
    citation_coverage_threshold: float = Field(default_factory=lambda: float(os.getenv("CITATION_COVERAGE_THRESHOLD", "0.85")))
    brand_alignment_threshold: float = Field(default_factory=lambda: float(os.getenv("BRAND_ALIGNMENT_THRESHOLD", "0.80")))


class Config(BaseModel):
    """Main configuration class"""
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    google: GoogleConfig = Field(default_factory=GoogleConfig)
    anthropic: AnthropicConfig = Field(default_factory=AnthropicConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    bm25: BM25Config = Field(default_factory=BM25Config)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    crag: CRAGConfig = Field(default_factory=CRAGConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    
    # Paths
    data_dir: Path = DATA_DIR
    processed_data_dir: Path = PROCESSED_DATA_DIR
    indexes_dir: Path = INDEXES_DIR


# Global config instance
config = Config()
