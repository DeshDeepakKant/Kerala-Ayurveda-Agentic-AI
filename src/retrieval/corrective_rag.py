"""
Corrective RAG (CRAG): Self-healing retrieval with confidence scoring
"""
from typing import List, Dict
import numpy as np
from loguru import logger
import google.generativeai as genai

from src.models import CRAGResult, RetrievalResult, QueryStrategy
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.query_transformer import AyurvedaQueryTransformer
from src.config import config


class CorrectiveRAG:
    """
    CRAG: Evaluates retrieval quality and triggers adaptive actions
    
    Three outcomes based on confidence:
    - HIGH (>0.7): Use retrieved docs directly
    - MEDIUM (0.3-0.7): Augment with query rewriting
    - LOW (<0.3): Flag for human review OR decompose query
    """
    
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.query_transformer = AyurvedaQueryTransformer()
        genai.configure(api_key=config.google.api_key)
        self.model_name = config.google.model
        self.model = genai.GenerativeModel(self.model_name)
        
        self.high_threshold = config.crag.high_confidence_threshold
        self.low_threshold = config.crag.low_confidence_threshold
        
        logger.info("Initialized CorrectiveRAG")
    
    def retrieve_with_correction(
        self,
        query: str,
        k: int = 5
    ) -> CRAGResult:
        """
        Main CRAG method: retrieve with confidence-based correction
        """
        # Step 1: Initial retrieval
        initial_results = self.retriever.search(query, k=10)
        
        if not initial_results:
            return CRAGResult(
                status="REQUIRES_HUMAN_REVIEW",
                documents=[],
                confidence=0.0,
                action_taken="no_results_found",
                reason="No documents retrieved"
            )
        
        # Step 2: Evaluate retrieval quality
        confidence = self._evaluate_retrieval_quality(query, initial_results)
        
        logger.info(f"Initial retrieval confidence: {confidence:.3f}")
        
        # Step 3: Adaptive action based on confidence
        if confidence > self.high_threshold:
            # CORRECT: High confidence, use directly
            return CRAGResult(
                status="CORRECT",
                documents=initial_results[:k],
                confidence=confidence,
                action_taken="direct_retrieval"
            )
        
        elif confidence > self.low_threshold:
            # AMBIGUOUS: Augment with refined retrieval
            return self._handle_ambiguous(query, initial_results, k)
        
        else:
            # INCORRECT: Low confidence, try recovery
            return self._handle_incorrect(query, initial_results, k)
    
    def _evaluate_retrieval_quality(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> float:
        """
        Evaluate retrieval quality using a reliable hybrid scoring approach:
          1. Normalized retrieval score  (how strong is the best match?)
          2. Query-document keyword overlap
          3. Score consistency           (are multiple results good?)
        Returns confidence score 0.0–1.0.
        """
        if not results:
            return 0.0

        scores = [r.score for r in results[:5]]
        max_score = max(scores)

        # --- Component 1: normalised retrieval score ---
        # BM25/hybrid scores for this corpus peak ~0.03–0.08 for good matches.
        # Map 0.05 → 1.0 so "reasonable" scores register as confident.
        SCORE_SCALE = 0.05
        norm_score = min(max_score / SCORE_SCALE, 1.0)

        # --- Component 2: keyword overlap ---
        stopwords = {'what', 'are', 'the', 'is', 'how', 'does', 'can', 'do',
                     'a', 'an', 'of', 'for', 'in', 'to', 'and', 'or', 'i',
                     'me', 'my', 'you', 'it', 'this', 'that', 'with', 'on'}
        query_words = set(query.lower().split()) - stopwords

        if query_words:
            top_text = ' '.join(r.chunk.text.lower() for r in results[:3])
            matched = sum(1 for w in query_words if w in top_text)
            overlap = min(matched / len(query_words), 1.0)
        else:
            overlap = 0.5

        # --- Component 3: score consistency ---
        good_results = sum(1 for s in scores if s > SCORE_SCALE * 0.4)
        consistency = min(good_results / 3, 1.0)

        confidence = 0.50 * norm_score + 0.35 * overlap + 0.15 * consistency
        confidence = round(max(0.0, min(1.0, confidence)), 3)

        logger.debug(
            f"CRAG confidence — norm={norm_score:.3f} overlap={overlap:.3f} "
            f"consistency={consistency:.3f} → {confidence:.3f}"
        )
        return confidence
    
    def _handle_ambiguous(
        self,
        query: str,
        initial_results: List[RetrievalResult],
        k: int
    ) -> CRAGResult:
        """
        Handle AMBIGUOUS case: Augment with query rewriting
        """
        logger.info("Handling AMBIGUOUS case - augmenting retrieval")
        
        # Try multiple query transformation strategies
        strategies = [QueryStrategy.REWRITE, QueryStrategy.STEP_BACK]
        
        augmented_results = list(initial_results)
        
        for strategy in strategies:
            try:
                transformed_query = self.query_transformer.transform_query(query, strategy)
                
                if isinstance(transformed_query, str):
                    additional_results = self.retriever.search(transformed_query, k=5)
                    augmented_results.extend(additional_results)
                
            except Exception as e:
                logger.error(f"Error with {strategy.value} strategy: {e}")
        
        # Deduplicate and rerank
        merged = self._merge_and_rerank(augmented_results, k)
        
        return CRAGResult(
            status="AMBIGUOUS_CORRECTED",
            documents=merged,
            confidence=0.5,  # Medium confidence
            action_taken=f"query_augmentation_{len(strategies)}_strategies"
        )
    
    def _handle_incorrect(
        self,
        query: str,
        initial_results: List[RetrievalResult],
        k: int
    ) -> CRAGResult:
        """
        Handle INCORRECT case: Try recovery through decomposition
        """
        logger.info("Handling INCORRECT case - attempting recovery")
        
        # Check if query is within Ayurveda domain
        if not self._is_ayurveda_query(query):
            return CRAGResult(
                status="OUT_OF_SCOPE",
                documents=[],
                confidence=0.0,
                action_taken="domain_check_failed",
                reason="Query appears outside Kerala Ayurveda domain"
            )
        
        # Try query decomposition
        try:
            decomposed_queries = self.query_transformer.transform_query(
                query,
                QueryStrategy.DECOMPOSE
            )
            
            if not isinstance(decomposed_queries, list):
                decomposed_queries = [decomposed_queries]
            
            # Retrieve for each sub-query
            recovered_results = []
            for sub_query in decomposed_queries:
                sub_results = self.retriever.search(sub_query, k=3)
                recovered_results.extend(sub_results)
            
            if recovered_results:
                merged = self._merge_and_rerank(recovered_results, k)
                
                return CRAGResult(
                    status="INCORRECT_RECOVERED",
                    documents=merged,
                    confidence=0.3,  # Low but recovered
                    action_taken=f"query_decomposition_{len(decomposed_queries)}_subqueries"
                )
        
        except Exception as e:
            logger.error(f"Error in recovery: {e}")
        
        # Failed to recover
        return CRAGResult(
            status="REQUIRES_HUMAN_REVIEW",
            documents=initial_results[:k] if initial_results else [],
            confidence=0.0,
            action_taken="recovery_failed",
            reason="No confident matches in Kerala Ayurveda corpus"
        )
    
    def _is_ayurveda_query(self, query: str) -> bool:
        """
        Check if query is within Ayurveda domain
        """
        ayurveda_keywords = [
            "ayurveda", "dosha", "vata", "pitta", "kapha",
            "herb", "treatment", "wellness", "digestion",
            "stress", "sleep", "health", "natural", "traditional",
            "ashwagandha", "brahmi", "triphala", "panchakarma"
        ]
        
        query_lower = query.lower()
        
        # Check for Ayurveda keywords
        keyword_match = any(keyword in query_lower for keyword in ayurveda_keywords)
        
        # If obvious match, return True
        if keyword_match:
            return True
        
        # Otherwise, use LLM to check
        try:
            prompt = f"""Is this question related to Ayurveda, herbal medicine, or wellness?

Question: {query}

Answer with ONLY "yes" or "no"."""
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=5
                )
            )
            
            answer = response.text.strip().lower()
            return "yes" in answer
            
        except Exception as e:
            logger.error(f"Error in domain check: {e}")
            return True  # Default to True to avoid false negatives
    
    def _merge_and_rerank(
        self,
        results: List[RetrievalResult],
        k: int
    ) -> List[RetrievalResult]:
        """
        Merge and deduplicate results, rerank by score
        """
        # Deduplicate by chunk ID
        seen_ids = set()
        unique_results = []
        
        for result in results:
            if result.chunk.id not in seen_ids:
                seen_ids.add(result.chunk.id)
                unique_results.append(result)
        
        # Sort by score
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(unique_results[:k], 1):
            result.rank = i
        
        return unique_results[:k]
