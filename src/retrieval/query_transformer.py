"""
Advanced Query Transformation: 4 Strategies
Rewrite, Decompose, Step-Back, HyDE
"""
import json
import re
from typing import List, Union
from loguru import logger
import google.generativeai as genai

from src.models import QueryStrategy
from src.config import config


class AyurvedaQueryTransformer:
    """
    Multi-strategy query transformation (RQ-RAG inspired)
    
    Strategies:
    1. REWRITE - Ayurvedic terminology alignment
    2. DECOMPOSE - Multi-hop query splitting
    3. STEP_BACK - Broader context generation
    4. HYDE - Hypothetical document generation
    """
    
    def __init__(self):
        genai.configure(api_key=config.google.api_key)
        self.model_name = config.google.model
        self.model = genai.GenerativeModel(self.model_name)
        logger.info(f"Initialized AyurvedaQueryTransformer with {self.model_name}")
    
    def transform_query(
        self,
        query: str,
        strategy: QueryStrategy = QueryStrategy.AUTO
    ) -> Union[str, List[str]]:
        """
        Intelligently select and apply query transformation strategy
        """
        if strategy == QueryStrategy.AUTO:
            strategy = self._detect_best_strategy(query)
            logger.info(f"Auto-selected strategy: {strategy.value}")
        
        if strategy == QueryStrategy.REWRITE:
            return self._rewrite_for_corpus(query)
        elif strategy == QueryStrategy.DECOMPOSE:
            return self._decompose_multi_hop(query)
        elif strategy == QueryStrategy.STEP_BACK:
            return self._generate_step_back(query)
        elif strategy == QueryStrategy.HYDE:
            return self._hypothetical_document(query)
        
        return query
    
    def _detect_best_strategy(self, query: str) -> QueryStrategy:
        """
        Use heuristics to detect best strategy
        """
        query_lower = query.lower()
        
        # Multi-hop indicators: compare, vs, and, both
        multi_hop_indicators = ["compare", " vs ", " vs. ", " versus ", " and ", "both", "difference between"]
        if any(indicator in query_lower for indicator in multi_hop_indicators):
            return QueryStrategy.DECOMPOSE
        
        # Vague or general queries benefit from step-back
        if len(query.split()) < 6 and any(word in query_lower for word in ["what", "how", "why"]):
            return QueryStrategy.STEP_BACK
        
        # Specific herb/treatment queries benefit from rewriting
        ayurveda_terms = [
            "ashwagandha", "brahmi", "triphala", "vata", "pitta", "kapha",
            "dosha", "panchakarma", "abhyanga", "shirodhara"
        ]
        if any(term in query_lower for term in ayurveda_terms):
            return QueryStrategy.REWRITE
        
        # Default: hypothetical document for abstract queries
        return QueryStrategy.HYDE
    
    def _rewrite_for_corpus(self, query: str) -> str:
        """
        Strategy 1: REWRITE with Ayurvedic terminology
        """
        prompt = f"""Rewrite this question using specific Ayurvedic terminology from Kerala Ayurveda corpus:

Original: {query}

Guidelines:
- Use Sanskrit terms when appropriate (Vata, Pitta, Kapha, Dosha, Prakriti)
- Mention specific treatments (Panchakarma, Abhyanga, Shirodhara)
- Reference body systems (digestive, circulatory, nervous)
- Use Kerala Ayurveda's warm, grounded language

Rewritten question (single sentence):"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=200
                )
            )
            
            rewritten = response.text.strip()
            logger.debug(f"Rewrite: '{query}' → '{rewritten}'")
            return rewritten
            
        except Exception as e:
            logger.error(f"Error in rewrite strategy: {e}")
            return query
    
    def _decompose_multi_hop(self, query: str) -> List[str]:
        """
        Strategy 2: DECOMPOSE complex multi-hop queries
        """
        prompt = f"""Break down this complex Ayurveda question into simple sub-questions:

Complex query: {query}

Instructions:
- Create 2-4 atomic sub-questions
- Each sub-question should be answerable independently
- Cover all aspects: definitions, benefits, contraindications, patient profiles

Return as JSON array of strings.

Sub-questions:"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=300,
                    response_mime_type="application/json"
                )
            )
            
            content = response.text.strip()
            
            # Try to parse JSON
            try:
                # Extract JSON array
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    sub_queries = json.loads(json_match.group(0))
                    logger.debug(f"Decomposed into {len(sub_queries)} sub-queries")
                    return sub_queries
                else:
                    # Fallback: split by numbered lines
                    lines = content.split('\n')
                    sub_queries = []
                    for line in lines:
                        line = line.strip()
                        # Match patterns like "1. ", "1) ", "- "
                        match = re.match(r'^[\d\-\*]+[\.\)]\s*(.+)$', line)
                        if match:
                            sub_queries.append(match.group(1).strip())
                    
                    if sub_queries:
                        return sub_queries
            except json.JSONDecodeError:
                pass
            
            # Final fallback: return original query
            logger.warning("Could not parse decomposed queries, returning original")
            return [query]
            
        except Exception as e:
            logger.error(f"Error in decompose strategy: {e}")
            return [query]
    
    def _generate_step_back(self, query: str) -> str:
        """
        Strategy 3: STEP-BACK to broader conceptual question
        """
        prompt = f"""Given this specific question, generate a broader, more general question that provides helpful background context:

Specific: {query}

Guidelines:
- Move from specific (herb name) to general (condition category)
- Move from treatment to underlying Ayurvedic principles
- Keep it focused on Ayurveda domain

Broader question (single sentence):"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=150
                )
            )
            
            broader_query = response.text.strip()
            logger.debug(f"Step-back: '{query}' → '{broader_query}'")
            return broader_query
            
        except Exception as e:
            logger.error(f"Error in step-back strategy: {e}")
            return query
    
    def _hypothetical_document(self, query: str) -> str:
        """
        Strategy 4: HYDE (Hypothetical Document Embeddings)
        Generate a hypothetical "perfect answer", then search for it
        """
        prompt = f"""Generate a detailed, factual answer to this Ayurveda question as if you had access to the perfect Kerala Ayurveda source document:

Question: {query}

Requirements:
- Write 2-3 sentences
- Use Ayurvedic terminology
- Include specific herb names, doshas, or treatments
- Sound like Kerala Ayurveda's warm, grounded voice

Hypothetical answer:"""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=250
                )
            )
            
            hypothetical_answer = response.text.strip()
            logger.debug(f"HyDE generated hypothetical answer ({len(hypothetical_answer)} chars)")
            return hypothetical_answer
            
        except Exception as e:
            logger.error(f"Error in HyDE strategy: {e}")
            return query
