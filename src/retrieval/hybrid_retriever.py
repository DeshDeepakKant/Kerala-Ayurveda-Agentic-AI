"""
Hybrid Retrieval System: BM25 + Vector Search with Reciprocal Rank Fusion
"""
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi
import chromadb
from chromadb.config import Settings
import google.generativeai as genai
from loguru import logger

from src.models import Chunk, RetrievalResult
from src.config import config


class HybridRetriever:
    """
    Hybrid retrieval combining BM25 (keyword) and dense embeddings (semantic)
    using Reciprocal Rank Fusion (RRF)
    """
    
    def __init__(self):
        genai.configure(api_key=config.google.api_key)
        self.embedding_model = config.google.embedding_model
        
        # BM25 index
        self.bm25: Optional[BM25Okapi] = None
        self.bm25_corpus: List[List[str]] = []
        self.chunk_id_to_chunk: Dict[str, Chunk] = {}
        self.chunk_ids_ordered: List[str] = []
        
        # Vector store (ChromaDB)
        self.vector_store = self._initialize_vector_store()
        self.collection = None
        
        logger.info("Initialized HybridRetriever")
    
    def _initialize_vector_store(self):
        """Initialize ChromaDB vector store"""
        persist_dir = config.vector_store.persist_directory
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        return client
    
    def index_chunks(self, chunks: List[Chunk], force_reindex: bool = False):
        """
        Index chunks in both BM25 and vector store
        """
        logger.info(f"Indexing {len(chunks)} chunks...")
        
        # Check if already indexed
        if not force_reindex:
            try:
                self.collection = self.vector_store.get_collection(
                    name=config.vector_store.collection_name
                )
                if self.collection.count() == len(chunks):
                    logger.info("Collection already indexed, loading...")
                    self._load_bm25_index()
                    self._rebuild_chunk_map(chunks)
                    return
            except Exception:
                pass
        
        # Create/reset collection
        try:
            self.vector_store.delete_collection(config.vector_store.collection_name)
        except:
            pass
        
        self.collection = self.vector_store.create_collection(
            name=config.vector_store.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Build BM25 index
        self._build_bm25_index(chunks)
        
        # Build vector index
        self._build_vector_index(chunks)
        
        # Save BM25 index
        self._save_bm25_index()
        
        logger.info("✓ Indexing complete")
    
    def _build_bm25_index(self, chunks: List[Chunk]):
        """Build BM25 index"""
        logger.info("Building BM25 index...")
        
        self.chunk_id_to_chunk = {chunk.id: chunk for chunk in chunks}
        self.chunk_ids_ordered = [chunk.id for chunk in chunks]
        
        # Tokenize corpus (simple whitespace tokenization)
        self.bm25_corpus = [
            chunk.text.lower().split() for chunk in chunks
        ]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(
            self.bm25_corpus,
            k1=config.bm25.k1,
            b=config.bm25.b
        )
        
        logger.info(f"✓ BM25 index built with {len(chunks)} documents")
    
    def _build_vector_index(self, chunks: List[Chunk]):
        """Build vector index with embeddings"""
        logger.info("Building vector index...")
        
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Generate embeddings
            texts = [chunk.text for chunk in batch]
            embeddings = self._batch_embed(texts)
            
            # Add to ChromaDB
            self.collection.add(
                ids=[chunk.id for chunk in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=[{
                    "doc_id": chunk.doc_id,
                    "doc_title": chunk.doc_title,
                    "doc_type": chunk.doc_type.value,
                    "section_type": chunk.section_type or ""
                } for chunk in batch]
            )
            
            logger.info(f"Indexed {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")
        
        logger.info(f"✓ Vector index built with {len(chunks)} embeddings")
    
    def _batch_embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Google Generative AI"""
        try:
            result = genai.embed_content(
                model=self.embedding_model,
                content=texts,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Return zero vectors as fallback
            return [[0.0] * config.vector_store.dimension for _ in texts]
    
    def search(self, query: str, k: int = 5) -> List[RetrievalResult]:
        """
        Hybrid search: BM25 + Vector search with Reciprocal Rank Fusion
        """
        if not self.bm25 or not self.collection:
            logger.error("Indexes not initialized. Call index_chunks() first.")
            return []
        
        # BM25 search
        bm25_results = self._bm25_search(query, k=config.retrieval.top_k)
        
        # Vector search
        vector_results = self._vector_search(query, k=config.retrieval.top_k)
        
        # Reciprocal Rank Fusion
        merged_results = self._reciprocal_rank_fusion(
            bm25_results, vector_results, k=k
        )
        
        return merged_results
    
    def _bm25_search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """BM25 keyword search"""
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k results
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        results = [
            (self.chunk_ids_ordered[idx], scores[idx])
            for idx in top_indices
        ]
        
        return results
    
    def _vector_search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Dense vector search"""
        # Generate query embedding
        query_embedding = self._batch_embed([query])[0]
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        # Format results (ChromaDB returns distances, convert to similarity)
        chunk_ids = results['ids'][0]
        distances = results['distances'][0]
        
        # Convert distance to similarity score (cosine similarity = 1 - distance)
        similarities = [1 - dist for dist in distances]
        
        return list(zip(chunk_ids, similarities))
    
    def _reciprocal_rank_fusion(
        self,
        bm25_results: List[Tuple[str, float]],
        vector_results: List[Tuple[str, float]],
        k: int = 5
    ) -> List[RetrievalResult]:
        """
        Reciprocal Rank Fusion: score(d) = Σ 1/(c + rank_i(d))
        """
        c = config.retrieval.rrf_constant
        scores = defaultdict(float)
        
        # Score from BM25 (using rank)
        for rank, (chunk_id, _) in enumerate(bm25_results, start=1):
            scores[chunk_id] += 1.0 / (c + rank)
        
        # Score from vector search (using rank)
        for rank, (chunk_id, _) in enumerate(vector_results, start=1):
            scores[chunk_id] += 1.0 / (c + rank)
        
        # Sort by combined score
        ranked_chunks = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        # Convert to RetrievalResult objects
        results = []
        for rank, (chunk_id, score) in enumerate(ranked_chunks, start=1):
            chunk = self.chunk_id_to_chunk.get(chunk_id)
            if chunk:
                result = RetrievalResult(
                    chunk=chunk,
                    score=score,
                    rank=rank,
                    retrieval_method="hybrid_rrf"
                )
                results.append(result)
        
        return results
    
    def _save_bm25_index(self):
        """Save BM25 index to disk"""
        index_path = config.indexes_dir / "bm25_index.pkl"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(index_path, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'chunk_ids_ordered': self.chunk_ids_ordered,
                'bm25_corpus': self.bm25_corpus
            }, f)
        
        logger.info(f"Saved BM25 index to {index_path}")
    
    def _load_bm25_index(self):
        """Load BM25 index from disk"""
        index_path = config.indexes_dir / "bm25_index.pkl"
        
        if not index_path.exists():
            logger.warning("BM25 index not found")
            return
        
        with open(index_path, 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
            self.chunk_ids_ordered = data['chunk_ids_ordered']
            self.bm25_corpus = data['bm25_corpus']
        
        logger.info(f"Loaded BM25 index from {index_path}")
    
    def _rebuild_chunk_map(self, chunks: List[Chunk]):
        """Rebuild chunk ID to chunk mapping"""
        self.chunk_id_to_chunk = {chunk.id: chunk for chunk in chunks}


from typing import Optional
