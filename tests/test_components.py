"""
Basic tests for Kerala Ayurveda RAG System.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfig:
    """Test configuration loading."""
    
    def test_config_loads(self):
        from src.config import Config
        config = Config()
        assert config is not None
        assert hasattr(config, 'openai')
        assert hasattr(config, 'crag')
    
    def test_config_defaults(self):
        from src.config import Config
        config = Config()
        assert config.crag.high_confidence_threshold == 0.7
        assert config.crag.medium_confidence_threshold == 0.3


class TestModels:
    """Test data models."""
    
    def test_document_model(self):
        from src.models import Document, DocumentType
        doc = Document(
            content="Test content",
            doc_type=DocumentType.FAQ,
            source="test.md"
        )
        assert doc.content == "Test content"
        assert doc.doc_type == DocumentType.FAQ
    
    def test_chunk_model(self):
        from src.models import Chunk
        chunk = Chunk(
            text="Test chunk",
            doc_title="Test Doc",
            metadata={"source": "test.md"}
        )
        assert chunk.text == "Test chunk"
        assert chunk.doc_title == "Test Doc"


class TestKnowledgeGraph:
    """Test Knowledge Graph functionality."""
    
    def test_kg_initialization(self):
        from src.knowledge_graph.ayurveda_kg import AyurvedaKnowledgeGraph
        kg = AyurvedaKnowledgeGraph()
        assert kg.graph.number_of_nodes() > 0
        assert kg.graph.number_of_edges() > 0
    
    def test_contraindication_check(self):
        from src.knowledge_graph.ayurveda_kg import AyurvedaKnowledgeGraph
        kg = AyurvedaKnowledgeGraph()
        
        # Test pregnancy contraindication
        result = kg.check_contraindication("Ashwagandha", ["pregnancy"])
        assert "safe" in result
        assert "severity" in result
        assert "reason" in result
    
    def test_related_entities(self):
        from src.knowledge_graph.ayurveda_kg import AyurvedaKnowledgeGraph
        kg = AyurvedaKnowledgeGraph()
        
        related = kg.get_related_entities("Ashwagandha", radius=2)
        assert isinstance(related, list)


class TestDocumentLoader:
    """Test document loading."""
    
    def test_loader_initialization(self):
        from src.data_processing.document_loader import DocumentLoader
        from src.config import Config
        
        config = Config()
        loader = DocumentLoader(config)
        assert loader is not None
    
    def test_load_documents(self):
        from src.data_processing.document_loader import DocumentLoader
        from src.config import Config
        
        config = Config()
        loader = DocumentLoader(config)
        
        # This may fail if kerala-data folder doesn't exist
        try:
            docs = loader.load_all_documents()
            assert isinstance(docs, list)
        except FileNotFoundError:
            pytest.skip("kerala-data folder not found")


class TestChunker:
    """Test chunking functionality."""
    
    def test_chunker_initialization(self):
        from src.data_processing.chunker import AyurvedaChunker
        from src.config import Config
        
        config = Config()
        chunker = AyurvedaChunker(config)
        assert chunker is not None
    
    def test_chunk_document(self):
        from src.data_processing.chunker import AyurvedaChunker
        from src.models import Document, DocumentType
        from src.config import Config
        
        config = Config()
        chunker = AyurvedaChunker(config)
        
        doc = Document(
            content="# Test\n\nThis is test content.\n\n## Section\n\nMore content here.",
            doc_type=DocumentType.FOUNDATION,
            source="test.md",
            title="Test Document"
        )
        
        chunks = chunker.chunk_document(doc)
        assert isinstance(chunks, list)
        assert len(chunks) > 0


class TestHybridRetriever:
    """Test hybrid retrieval."""
    
    def test_retriever_initialization(self):
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.config import Config
        
        config = Config()
        retriever = HybridRetriever(config)
        assert retriever is not None


class TestQueryTransformer:
    """Test query transformation."""
    
    def test_transformer_initialization(self):
        from src.retrieval.query_transformer import AyurvedaQueryTransformer
        from src.config import Config
        
        config = Config()
        transformer = AyurvedaQueryTransformer(config)
        assert transformer is not None


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
