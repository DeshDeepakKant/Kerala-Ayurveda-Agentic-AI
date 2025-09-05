"""
Kerala Ayurveda RAG System - Main Pipeline
Complete end-to-end system for document processing, indexing, and retrieval
"""
from pathlib import Path
from typing import List
from loguru import logger
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import config
from src.models import Chunk
from src.data_processing.document_loader import DocumentLoader
from src.data_processing.chunker import AyurvedaChunker
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.corrective_rag import CorrectiveRAG
from src.knowledge_graph.ayurveda_kg import AyurvedaKnowledgeGraph


class KeralaAyurvedaRAG:
    """
    Main RAG System for Kerala Ayurveda
    """
    
    def __init__(self):
        logger.info("=" * 60)
        logger.info("Initializing Kerala Ayurveda RAG System")
        logger.info("=" * 60)
        
        # Initialize components
        self.document_loader = DocumentLoader()
        self.chunker = AyurvedaChunker()
        self.retriever = HybridRetriever()
        self.crag = CorrectiveRAG(self.retriever)
        self.knowledge_graph = AyurvedaKnowledgeGraph()
        
        self.documents = []
        self.chunks = []
        
        logger.info("✓ System initialized")
    
    def build_index(self, force_reindex: bool = False):
        """
        Complete indexing pipeline:
        1. Load documents
        2. Chunk documents
        3. Build hybrid indexes (BM25 + Vector)
        """
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: Loading Documents")
        logger.info("=" * 60)
        
        self.documents = self.document_loader.load_all_documents()
        logger.info(f"✓ Loaded {len(self.documents)} documents")
        
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: Chunking Documents")
        logger.info("=" * 60)
        
        all_chunks = []
        for doc in self.documents:
            chunks = self.chunker.chunk_document(doc)
            all_chunks.extend(chunks)
        
        self.chunks = all_chunks
        logger.info(f"✓ Created {len(self.chunks)} chunks")
        
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: Building Hybrid Indexes")
        logger.info("=" * 60)
        
        self.retriever.index_chunks(self.chunks, force_reindex=force_reindex)
        logger.info("✓ Indexing complete")
        
        # Print summary
        self._print_index_summary()
    
    def query(self, query: str, use_crag: bool = True):
        """
        Query the system
        
        Args:
            query: User query
            use_crag: Whether to use CRAG (default: True)
        """
        logger.info("\n" + "=" * 60)
        logger.info(f"QUERY: {query}")
        logger.info("=" * 60)
        
        if use_crag:
            # Use CRAG for self-healing retrieval
            result = self.crag.retrieve_with_correction(query, k=5)
            
            logger.info(f"Status: {result.status}")
            logger.info(f"Confidence: {result.confidence:.3f}")
            logger.info(f"Action: {result.action_taken}")
            
            if result.reason:
                logger.info(f"Reason: {result.reason}")
            
            logger.info(f"\nRetrieved {len(result.documents)} documents:")
            
            for i, doc in enumerate(result.documents[:3], 1):
                logger.info(f"\n[{i}] {doc.chunk.doc_title} ({doc.chunk.section_type})")
                logger.info(f"    Score: {doc.score:.4f}")
                logger.info(f"    Preview: {doc.chunk.text[:200]}...")
            
            return result
        else:
            # Direct retrieval
            results = self.retriever.search(query, k=5)
            
            logger.info(f"Retrieved {len(results)} documents:")
            
            for i, result in enumerate(results[:3], 1):
                logger.info(f"\n[{i}] {result.chunk.doc_title} ({result.chunk.section_type})")
                logger.info(f"    Score: {result.score:.4f}")
                logger.info(f"    Preview: {result.chunk.text[:200]}...")
            
            return results
    
    def check_safety(self, herb_or_treatment: str, conditions: List[str]):
        """
        Check contraindications using Knowledge Graph
        """
        logger.info("\n" + "=" * 60)
        logger.info(f"SAFETY CHECK: {herb_or_treatment}")
        logger.info(f"Conditions: {', '.join(conditions)}")
        logger.info("=" * 60)
        
        result = self.knowledge_graph.check_contraindication(
            herb_or_treatment, conditions
        )
        
        logger.info(f"Safe: {result['safe']}")
        logger.info(f"Severity: {result['severity']}")
        logger.info(f"Reason: {result['reason']}")
        
        if result['contraindications_found']:
            logger.warning(f"⚠️  CONTRAINDICATIONS: {', '.join(result['contraindications_found'])}")
        
        if result['cautions_found']:
            logger.warning(f"⚠️  CAUTIONS: {', '.join(result['cautions_found'])}")
        
        return result
    
    def _print_index_summary(self):
        """Print summary of indexed data"""
        logger.info("\n" + "=" * 60)
        logger.info("INDEX SUMMARY")
        logger.info("=" * 60)
        
        # Documents by type
        from collections import Counter
        doc_types = Counter(doc.doc_type.value for doc in self.documents)
        
        logger.info("\nDocuments by type:")
        for doc_type, count in doc_types.items():
            logger.info(f"  {doc_type}: {count}")
        
        # Chunks by section type
        chunk_types = Counter(chunk.section_type for chunk in self.chunks if chunk.section_type)
        
        logger.info("\nChunks by section type:")
        for chunk_type, count in list(chunk_types.most_common(10)):
            logger.info(f"  {chunk_type}: {count}")
        
        logger.info(f"\nTotal documents: {len(self.documents)}")
        logger.info(f"Total chunks: {len(self.chunks)}")


def demo():
    """
    Demo the Kerala Ayurveda RAG System
    """
    # Initialize system
    rag = KeralaAyurvedaRAG()
    
    # Build index
    rag.build_index(force_reindex=False)
    
    print("\n" + "=" * 60)
    print("DEMO: Testing Kerala Ayurveda RAG System")
    print("=" * 60)
    
    # Test queries
    test_queries = [
        "What are the benefits of Triphala for digestion?",
        "How does Ashwagandha help with stress and sleep?",
        "Compare Ashwagandha and Brahmi for stress support",
        "Can pregnant women use Ashwagandha?",
    ]
    
    for query in test_queries:
        rag.query(query, use_crag=True)
        input("\nPress Enter to continue...")
    
    # Test safety checks
    print("\n" + "=" * 60)
    print("DEMO: Testing Safety Checks")
    print("=" * 60)
    
    rag.check_safety("Ashwagandha", ["pregnancy"])
    rag.check_safety("Ashwagandha", ["thyroid_condition"])
    rag.check_safety("Triphala", ["pregnancy"])


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )
    
    # Run demo
    demo()
