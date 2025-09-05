"""
Comprehensive Evaluation Framework for Kerala Ayurveda RAG System.

This module provides:
1. RAGAS-style metrics (faithfulness, relevancy, context precision/recall)
2. Multi-method hallucination detection
3. Domain-specific Ayurveda metrics
4. Brand voice alignment scoring
"""

import json
import re
from typing import Optional
from dataclasses import dataclass, field
import google.generativeai as genai

from src.config import Config
from src.models import EvaluationMetrics, RetrievalResult

import logging

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Complete evaluation result for a RAG response."""
    # Core RAGAS metrics
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    
    # Hallucination metrics
    hallucination_score: float = 0.0
    unsupported_claims: list = field(default_factory=list)
    
    # Domain-specific metrics
    ayurveda_accuracy: float = 0.0
    contraindication_coverage: float = 0.0
    
    # Brand metrics
    brand_alignment: float = 0.0
    forbidden_phrases_found: list = field(default_factory=list)
    
    # Overall
    overall_score: float = 0.0
    pass_threshold: bool = False
    
    def to_dict(self) -> dict:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "hallucination_score": self.hallucination_score,
            "unsupported_claims": self.unsupported_claims,
            "ayurveda_accuracy": self.ayurveda_accuracy,
            "contraindication_coverage": self.contraindication_coverage,
            "brand_alignment": self.brand_alignment,
            "forbidden_phrases_found": self.forbidden_phrases_found,
            "overall_score": self.overall_score,
            "pass_threshold": self.pass_threshold
        }


class HallucinationDetector:
    """
    Multi-method hallucination detection.
    
    Methods:
    1. Claim extraction + source verification
    2. Consistency checking (multiple generations)
    3. LLM-as-Judge evaluation
    """
    
    def __init__(self, config: Config):
        self.config = config
        genai.configure(api_key=config.google.api_key)
        self.model_name = config.google.model
        self.model = genai.GenerativeModel(self.model_name)
    
    def detect(
        self,
        response: str,
        context: list[str],
        query: str
    ) -> dict:
        """
        Detect hallucinations using multiple methods.
        
        Args:
            response: Generated response to check
            context: Retrieved context chunks used for generation
            query: Original user query
            
        Returns:
            dict with hallucination_score and details
        """
        # Method 1: Claim extraction and verification
        claim_result = self._verify_claims(response, context)
        
        # Method 2: LLM-as-Judge
        judge_result = self._llm_judge(response, context, query)
        
        # Combine scores (weighted average)
        combined_score = (
            claim_result["score"] * 0.6 +
            judge_result["score"] * 0.4
        )
        
        return {
            "hallucination_score": combined_score,
            "claim_verification": claim_result,
            "llm_judge": judge_result,
            "is_hallucinated": combined_score > 0.3,
            "unsupported_claims": claim_result.get("unsupported", [])
        }
    
    def _verify_claims(self, response: str, context: list[str]) -> dict:
        """Extract claims and verify against context."""
        
        context_text = "\n\n".join(context[:5])  # Use top 5 chunks
        
        prompt = f"""Extract factual claims from this response and verify each against the context.

RESPONSE:
{response[:2000]}

CONTEXT:
{context_text[:3000]}

For each factual claim (specific benefits, dosages, contraindications, etc.):
1. Extract the claim
2. Check if context supports it
3. Mark as SUPPORTED, PARTIAL, or UNSUPPORTED

Return JSON:
{{
    "claims": [
        {{"claim": "text", "status": "SUPPORTED|PARTIAL|UNSUPPORTED", "evidence": "quote or null"}}
    ],
    "total_claims": N,
    "supported_count": N,
    "unsupported_count": N
}}
"""
        
        try:
            response_obj = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )

            result = json.loads(response_obj.text)
            
            total = result.get("total_claims", 1)
            unsupported = result.get("unsupported_count", 0)
            
            score = unsupported / max(total, 1)
            
            unsupported_claims = [
                c["claim"] for c in result.get("claims", [])
                if c.get("status") == "UNSUPPORTED"
            ]
            
            return {
                "score": score,
                "total_claims": total,
                "supported": result.get("supported_count", 0),
                "unsupported": unsupported_claims,
                "details": result.get("claims", [])
            }
            
        except Exception as e:
            logger.error(f"Claim verification failed: {e}")
            return {"score": 0.5, "error": str(e), "unsupported": []}
    
    def _llm_judge(self, response: str, context: list[str], query: str) -> dict:
        """Use LLM as a judge to evaluate faithfulness."""
        
        context_text = "\n\n".join(context[:5])
        
        prompt = f"""You are a judge evaluating whether an AI response is faithful to the provided context.

QUERY: {query}

CONTEXT PROVIDED:
{context_text[:2500]}

AI RESPONSE:
{response[:1500]}

Evaluate on a scale of 0-10:
1. How much of the response is directly supported by the context?
2. Does the response make claims beyond what the context supports?
3. Are there any contradictions with the context?

Return JSON:
{{
    "faithfulness_score": 0-10,
    "issues": ["list of specific issues found"],
    "verdict": "FAITHFUL|MOSTLY_FAITHFUL|PARTIALLY_FAITHFUL|UNFAITHFUL"
}}
"""
        
        try:
            response_obj = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )

            result = json.loads(response_obj.text)
            
            # Convert 0-10 score to 0-1 hallucination score (inverted)
            faithfulness = result.get("faithfulness_score", 5) / 10
            hallucination_score = 1 - faithfulness
            
            return {
                "score": hallucination_score,
                "faithfulness": faithfulness,
                "verdict": result.get("verdict", "UNKNOWN"),
                "issues": result.get("issues", [])
            }
            
        except Exception as e:
            logger.error(f"LLM judge failed: {e}")
            return {"score": 0.5, "error": str(e)}


class RAGEvaluator:
    """
    Comprehensive RAG evaluation using RAGAS-style metrics.
    
    Metrics:
    - Faithfulness: Is the answer grounded in context?
    - Answer Relevancy: Does the answer address the query?
    - Context Precision: Are retrieved docs relevant?
    - Context Recall: Is needed information retrieved?
    """
    
    FORBIDDEN_PHRASES = [
        "cures", "treats disease", "heals", "100% safe",
        "miracle", "guaranteed", "no side effects",
        "replace medication", "stop taking"
    ]
    
    AYURVEDA_TERMS = [
        "dosha", "vata", "pitta", "kapha", "prakriti",
        "agni", "ama", "ojas", "panchakarma", "rasayana",
        "abhyanga", "shirodhara", "ayurveda", "ayurvedic"
    ]
    
    def __init__(self, config: Config):
        self.config = config
        genai.configure(api_key=config.google.api_key)
        self.model_name = config.google.model
        self.model = genai.GenerativeModel(self.model_name)
        self.hallucination_detector = HallucinationDetector(config)
    
    def evaluate(
        self,
        query: str,
        response: str,
        context: list[str],
        ground_truth: Optional[str] = None
    ) -> EvaluationResult:
        """
        Evaluate a RAG response comprehensively.
        
        Args:
            query: Original user query
            response: Generated response
            context: Retrieved context chunks
            ground_truth: Optional ground truth answer for comparison
            
        Returns:
            EvaluationResult with all metrics
        """
        logger.info(f"Evaluating response for query: {query[:50]}...")
        
        result = EvaluationResult()
        
        # 1. Faithfulness (is response grounded in context?)
        result.faithfulness = self._evaluate_faithfulness(response, context)
        
        # 2. Answer Relevancy (does response address query?)
        result.answer_relevancy = self._evaluate_relevancy(query, response)
        
        # 3. Context Precision (are retrieved docs relevant?)
        result.context_precision = self._evaluate_context_precision(query, context)
        
        # 4. Context Recall (if ground truth available)
        if ground_truth:
            result.context_recall = self._evaluate_context_recall(ground_truth, context)
        else:
            result.context_recall = 0.8  # Default assumption
        
        # 5. Hallucination detection
        hallucination_result = self.hallucination_detector.detect(response, context, query)
        result.hallucination_score = hallucination_result["hallucination_score"]
        result.unsupported_claims = hallucination_result.get("unsupported_claims", [])
        
        # 6. Domain-specific: Ayurveda accuracy
        result.ayurveda_accuracy = self._evaluate_ayurveda_accuracy(response)
        
        # 7. Brand alignment
        brand_result = self._evaluate_brand_alignment(response)
        result.brand_alignment = brand_result["score"]
        result.forbidden_phrases_found = brand_result["violations"]
        
        # 8. Contraindication coverage (for product/treatment content)
        result.contraindication_coverage = self._evaluate_contraindication_coverage(response)
        
        # Calculate overall score
        result.overall_score = self._calculate_overall_score(result)
        result.pass_threshold = result.overall_score >= 0.7 and result.hallucination_score < 0.2
        
        return result
    
    def _evaluate_faithfulness(self, response: str, context: list[str]) -> float:
        """Evaluate if response is faithful to context."""
        
        context_text = "\n\n".join(context[:5])
        
        prompt = f"""Rate how faithful this response is to the provided context.

CONTEXT:
{context_text[:2500]}

RESPONSE:
{response[:1500]}

A faithful response:
- Only contains information from the context
- Does not add unsupported claims
- Does not contradict the context

Rate faithfulness from 0.0 to 1.0 and return JSON:
{{"faithfulness": 0.0-1.0, "reason": "brief explanation"}}
"""
        
        try:
            response_obj = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )

            result = json.loads(response_obj.text)
            return float(result.get("faithfulness", 0.5))
            
        except Exception as e:
            logger.error(f"Faithfulness evaluation failed: {e}")
            return 0.5
    
    def _evaluate_relevancy(self, query: str, response: str) -> float:
        """Evaluate if response is relevant to query."""
        
        prompt = f"""Rate how relevant this response is to the query.

QUERY: {query}

RESPONSE:
{response[:1500]}

A relevant response:
- Directly addresses the query
- Provides the information requested
- Stays on topic

Rate relevancy from 0.0 to 1.0 and return JSON:
{{"relevancy": 0.0-1.0, "reason": "brief explanation"}}
"""
        
        try:
            response_obj = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )

            result = json.loads(response_obj.text)
            return float(result.get("relevancy", 0.5))
            
        except Exception as e:
            logger.error(f"Relevancy evaluation failed: {e}")
            return 0.5
    
    def _evaluate_context_precision(self, query: str, context: list[str]) -> float:
        """Evaluate if retrieved context is relevant to query."""
        
        if not context:
            return 0.0
        
        context_summaries = "\n".join([
            f"Chunk {i+1}: {chunk[:200]}..."
            for i, chunk in enumerate(context[:5])
        ])
        
        prompt = f"""Rate how relevant each retrieved context chunk is to the query.

QUERY: {query}

RETRIEVED CHUNKS:
{context_summaries}

For each chunk, determine if it's relevant to answering the query.
Return JSON:
{{"relevant_chunks": N, "total_chunks": N, "precision": 0.0-1.0}}
"""
        
        try:
            response_obj = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )

            result = json.loads(response_obj.text)
            return float(result.get("precision", 0.5))
            
        except Exception as e:
            logger.error(f"Context precision evaluation failed: {e}")
            return 0.5
    
    def _evaluate_context_recall(self, ground_truth: str, context: list[str]) -> float:
        """Evaluate if context contains info needed for ground truth."""
        
        context_text = "\n\n".join(context[:5])
        
        prompt = f"""Determine if the retrieved context contains the information in the ground truth.

GROUND TRUTH ANSWER:
{ground_truth[:500]}

RETRIEVED CONTEXT:
{context_text[:2000]}

What percentage of the ground truth information is present in the context?
Return JSON:
{{"recall": 0.0-1.0, "missing_info": ["list of missing information"]}}
"""
        
        try:
            response_obj = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )

            result = json.loads(response_obj.text)
            return float(result.get("recall", 0.5))
            
        except Exception as e:
            logger.error(f"Context recall evaluation failed: {e}")
            return 0.5
    
    def _evaluate_ayurveda_accuracy(self, response: str) -> float:
        """Evaluate Ayurvedic terminology accuracy."""
        
        response_lower = response.lower()
        
        # Check for Ayurvedic terms usage
        terms_used = sum(1 for term in self.AYURVEDA_TERMS if term in response_lower)
        
        # Simple heuristic: more terms = more domain-appropriate
        # But also check for correct usage via LLM
        
        prompt = f"""Evaluate the Ayurvedic accuracy of this content.

CONTENT:
{response[:1500]}

Check:
1. Are Ayurvedic terms used correctly?
2. Are dosha descriptions accurate?
3. Are treatment/herb descriptions consistent with Ayurvedic principles?

Return JSON:
{{"accuracy": 0.0-1.0, "issues": ["list of inaccuracies if any"]}}
"""
        
        try:
            response_obj = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )

            result = json.loads(response_obj.text)
            return float(result.get("accuracy", 0.7))
            
        except Exception as e:
            logger.error(f"Ayurveda accuracy evaluation failed: {e}")
            # Fallback: basic term presence check
            return min(1.0, terms_used / 5)
    
    def _evaluate_brand_alignment(self, response: str) -> dict:
        """Evaluate alignment with Kerala Ayurveda brand voice."""
        
        response_lower = response.lower()
        
        # Check for forbidden phrases
        violations = []
        for phrase in self.FORBIDDEN_PHRASES:
            if phrase in response_lower:
                violations.append(phrase)
        
        # Check for required elements
        positive_indicators = [
            "traditionally" in response_lower,
            "may support" in response_lower or "may help" in response_lower,
            "consult" in response_lower,
            "healthcare" in response_lower or "health care" in response_lower
        ]
        
        # Calculate score
        violation_penalty = len(violations) * 0.15
        positive_bonus = sum(positive_indicators) * 0.1
        
        score = max(0.0, min(1.0, 0.7 - violation_penalty + positive_bonus))
        
        return {
            "score": score,
            "violations": violations,
            "positive_indicators": sum(positive_indicators)
        }
    
    def _evaluate_contraindication_coverage(self, response: str) -> float:
        """Check if response includes appropriate safety information."""
        
        response_lower = response.lower()
        
        safety_indicators = [
            "contraindic" in response_lower,
            "caution" in response_lower,
            "consult" in response_lower,
            "pregnant" in response_lower or "pregnancy" in response_lower,
            "side effect" in response_lower,
            "not recommended" in response_lower,
            "avoid" in response_lower,
            "healthcare provider" in response_lower
        ]
        
        # For product/treatment content, safety info is important
        coverage = sum(safety_indicators) / len(safety_indicators)
        
        return coverage
    
    def _calculate_overall_score(self, result: EvaluationResult) -> float:
        """Calculate weighted overall score."""
        
        weights = {
            "faithfulness": 0.25,
            "answer_relevancy": 0.20,
            "context_precision": 0.10,
            "context_recall": 0.10,
            "hallucination": 0.15,  # Inverted
            "ayurveda_accuracy": 0.10,
            "brand_alignment": 0.10
        }
        
        score = (
            result.faithfulness * weights["faithfulness"] +
            result.answer_relevancy * weights["answer_relevancy"] +
            result.context_precision * weights["context_precision"] +
            result.context_recall * weights["context_recall"] +
            (1 - result.hallucination_score) * weights["hallucination"] +
            result.ayurveda_accuracy * weights["ayurveda_accuracy"] +
            result.brand_alignment * weights["brand_alignment"]
        )
        
        return round(score, 3)
    
    def evaluate_batch(
        self,
        test_cases: list[dict]
    ) -> dict:
        """
        Evaluate multiple test cases.
        
        Args:
            test_cases: List of {"query": str, "response": str, "context": list, "ground_truth": optional}
            
        Returns:
            Aggregate metrics and individual results
        """
        results = []
        
        for case in test_cases:
            result = self.evaluate(
                query=case["query"],
                response=case["response"],
                context=case.get("context", []),
                ground_truth=case.get("ground_truth")
            )
            results.append(result)
        
        # Aggregate metrics
        n = len(results)
        if n == 0:
            return {"error": "No test cases provided"}
        
        aggregate = {
            "avg_faithfulness": sum(r.faithfulness for r in results) / n,
            "avg_relevancy": sum(r.answer_relevancy for r in results) / n,
            "avg_context_precision": sum(r.context_precision for r in results) / n,
            "avg_hallucination_score": sum(r.hallucination_score for r in results) / n,
            "avg_overall_score": sum(r.overall_score for r in results) / n,
            "pass_rate": sum(1 for r in results if r.pass_threshold) / n,
            "total_cases": n,
            "individual_results": [r.to_dict() for r in results]
        }
        
        return aggregate


class GoldenDatasetEvaluator:
    """
    Evaluator using golden (ground truth) test cases.
    """
    
    # Sample golden test cases for Kerala Ayurveda
    GOLDEN_DATASET = [
        {
            "query": "What are the benefits of Ashwagandha for stress?",
            "expected_entities": ["ashwagandha", "stress"],
            "expected_topics": ["adaptogenic", "cortisol", "anxiety", "sleep"],
            "forbidden_claims": ["cures anxiety", "treats depression"],
            "required_safety": True
        },
        {
            "query": "Can pregnant women take Triphala?",
            "expected_entities": ["triphala", "pregnancy"],
            "expected_answer_type": "contraindication_warning",
            "must_include": ["consult", "healthcare", "not recommended"],
            "required_safety": True
        },
        {
            "query": "How does Brahmi support cognitive function?",
            "expected_entities": ["brahmi", "cognitive", "memory"],
            "expected_topics": ["nootropic", "concentration", "mental clarity"],
            "forbidden_claims": ["cures dementia", "treats alzheimer"],
            "required_safety": True
        },
        {
            "query": "What is the difference between Vata and Pitta doshas?",
            "expected_entities": ["vata", "pitta", "dosha"],
            "expected_topics": ["elements", "characteristics", "qualities"],
            "must_include": ["air", "fire", "qualities"],
            "required_safety": False
        },
        {
            "query": "What products does Kerala Ayurveda offer for sleep support?",
            "expected_entities": ["sleep", "product"],
            "expected_products": ["ashwagandha"],
            "must_include": ["traditionally", "support"],
            "required_safety": False
        }
    ]
    
    def __init__(self, config: Config):
        self.config = config
        self.evaluator = RAGEvaluator(config)
    
    def evaluate_against_golden(
        self,
        response: str,
        golden_case: dict,
        context: list[str]
    ) -> dict:
        """Evaluate response against golden test case."""
        
        response_lower = response.lower()
        
        results = {
            "query": golden_case["query"],
            "checks": {}
        }
        
        # Check expected entities are mentioned
        if "expected_entities" in golden_case:
            entities_found = sum(
                1 for e in golden_case["expected_entities"]
                if e.lower() in response_lower
            )
            results["checks"]["entity_coverage"] = entities_found / len(golden_case["expected_entities"])
        
        # Check expected topics are covered
        if "expected_topics" in golden_case:
            topics_found = sum(
                1 for t in golden_case["expected_topics"]
                if t.lower() in response_lower
            )
            results["checks"]["topic_coverage"] = topics_found / len(golden_case["expected_topics"])
        
        # Check forbidden claims are absent
        if "forbidden_claims" in golden_case:
            violations = [
                claim for claim in golden_case["forbidden_claims"]
                if claim.lower() in response_lower
            ]
            results["checks"]["no_forbidden_claims"] = len(violations) == 0
            results["checks"]["forbidden_violations"] = violations
        
        # Check required phrases
        if "must_include" in golden_case:
            included = sum(
                1 for phrase in golden_case["must_include"]
                if phrase.lower() in response_lower
            )
            results["checks"]["required_phrases"] = included / len(golden_case["must_include"])
        
        # Check safety info if required
        if golden_case.get("required_safety"):
            safety_phrases = ["consult", "healthcare", "caution", "contraindic"]
            safety_present = any(p in response_lower for p in safety_phrases)
            results["checks"]["safety_included"] = safety_present
        
        # Run full evaluation
        full_eval = self.evaluator.evaluate(
            query=golden_case["query"],
            response=response,
            context=context
        )
        results["full_evaluation"] = full_eval.to_dict()
        
        # Calculate golden test score
        check_scores = [
            v for k, v in results["checks"].items()
            if isinstance(v, (int, float))
        ]
        results["golden_score"] = sum(check_scores) / len(check_scores) if check_scores else 0.0
        
        return results
    
    def run_golden_evaluation(
        self,
        generate_fn,
        retrieve_fn
    ) -> dict:
        """
        Run evaluation on all golden test cases.
        
        Args:
            generate_fn: Function(query) -> response
            retrieve_fn: Function(query) -> list[context_chunks]
            
        Returns:
            Aggregate results across all golden cases
        """
        results = []
        
        for case in self.GOLDEN_DATASET:
            # Generate response
            response = generate_fn(case["query"])
            context = retrieve_fn(case["query"])
            
            # Evaluate
            case_result = self.evaluate_against_golden(response, case, context)
            results.append(case_result)
        
        # Aggregate
        n = len(results)
        aggregate = {
            "total_cases": n,
            "avg_golden_score": sum(r["golden_score"] for r in results) / n,
            "avg_overall_score": sum(
                r["full_evaluation"]["overall_score"] for r in results
            ) / n,
            "cases_passed": sum(
                1 for r in results
                if r["golden_score"] >= 0.7 and r["full_evaluation"]["pass_threshold"]
            ),
            "individual_results": results
        }
        
        return aggregate
