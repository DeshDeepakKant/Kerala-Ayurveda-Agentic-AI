"""
LangGraph Multi-Agent Workflow for Kerala Ayurveda RAG System.

This module implements a 6-agent pipeline with reflection loops:
1. Query Understanding Agent - Extract intent, entities, constraints
2. Outline Generator Agent - Create structure with KG enhancement
3. Writer Agent - Generate content with CRAG retrieval
4. Fact-Checker Agent - Multi-method hallucination detection
5. Reflection Agent - Self-healing feedback loops
6. Style Editor Agent - Brand voice alignment
"""

import json
from typing import TypedDict, Annotated, Sequence, Literal, Optional
from dataclasses import dataclass, field
from enum import Enum
import operator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import google.generativeai as genai

from src.config import Config
from src.models import AgentState, VerificationReport, DocumentType
from src.retrieval.corrective_rag import CorrectiveRAG
from src.retrieval.hybrid_retriever import HybridRetriever
from src.knowledge_graph.ayurveda_kg import AyurvedaKnowledgeGraph

import logging

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict):
    """State passed between agents in the workflow."""
    # Input
    query: str
    content_type: str  # article, faq, product_description
    
    # Query Understanding output
    intent: str
    entities: list[str]
    constraints: list[str]
    dosha_context: Optional[str]
    
    # Outline output
    outline: list[dict]
    kg_enhancements: list[str]
    
    # Writer output
    draft: str
    citations: list[dict]
    retrieved_chunks: list[dict]
    
    # Fact-Checker output
    verification_report: Optional[dict]
    hallucination_score: float
    flagged_claims: list[dict]
    
    # Reflection output
    feedback: list[str]
    revision_count: int
    
    # Style Editor output
    final_output: str
    brand_alignment_score: float
    
    # Control flow
    should_revise: bool
    error: Optional[str]
    messages: Annotated[Sequence[str], operator.add]


class QueryUnderstandingAgent:
    """
    Agent 1: Analyzes the user query to extract intent, entities, and constraints.
    
    Responsibilities:
    - Classify query intent (informational, product inquiry, treatment advice)
    - Extract Ayurvedic entities (herbs, doshas, treatments)
    - Identify constraints (contraindications, patient profile)
    - Determine dosha context if relevant
    """
    
    def __init__(self, config: Config):
        self.config = config
        genai.configure(api_key=config.google.api_key)
        self.model_name = config.google.model
        self.model = genai.GenerativeModel(self.model_name)
    
    def __call__(self, state: WorkflowState) -> WorkflowState:
        """Process the query and extract understanding."""
        logger.info(f"QueryUnderstandingAgent processing: {state['query'][:50]}...")
        
        prompt = f"""Analyze this Ayurveda-related query and extract structured information.

Query: {state['query']}
Content Type Requested: {state['content_type']}

Provide a JSON response with:
{{
    "intent": "informational|product_inquiry|treatment_advice|comparison|safety_check",
    "entities": ["list of Ayurvedic entities mentioned (herbs, doshas, treatments, products)"],
    "constraints": ["any constraints or conditions mentioned (pregnancy, allergies, etc.)"],
    "dosha_context": "vata|pitta|kapha|null if not specified",
    "complexity": "simple|moderate|complex"
}}

Be thorough in entity extraction. Include:
- Herbs: Ashwagandha, Brahmi, Triphala, Tulsi, etc.
- Doshas: Vata, Pitta, Kapha
- Treatments: Panchakarma, Abhyanga, Shirodhara
- Conditions: stress, digestion, sleep, immunity
"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )
            
            result = json.loads(response.text)
            
            return {
                **state,
                "intent": result.get("intent", "informational"),
                "entities": result.get("entities", []),
                "constraints": result.get("constraints", []),
                "dosha_context": result.get("dosha_context"),
                "messages": [f"✓ Query analyzed: intent={result.get('intent')}, {len(result.get('entities', []))} entities found"]
            }
            
        except Exception as e:
            logger.error(f"Query understanding failed: {e}")
            return {
                **state,
                "intent": "informational",
                "entities": [],
                "constraints": [],
                "dosha_context": None,
                "error": str(e),
                "messages": [f"⚠ Query understanding error: {str(e)}"]
            }


class OutlineGeneratorAgent:
    """
    Agent 2: Creates content outline enhanced with Knowledge Graph context.
    
    Responsibilities:
    - Generate logical content structure
    - Enhance with KG relationships (herb→dosha→product)
    - Include safety sections for relevant queries
    - Add citation placeholders
    """
    
    def __init__(self, config: Config, knowledge_graph: AyurvedaKnowledgeGraph):
        self.config = config
        genai.configure(api_key=config.google.api_key)
        self.model_name = config.google.model
        self.model = genai.GenerativeModel(self.model_name)
        self.kg = knowledge_graph
    
    def __call__(self, state: WorkflowState) -> WorkflowState:
        """Generate content outline with KG enhancement."""
        logger.info(f"OutlineGeneratorAgent creating outline for: {state['intent']}")
        
        # Get KG enhancements for entities
        kg_context = []
        for entity in state.get("entities", []):
            related = self.kg.get_related_entities(entity, radius=2)
            if related and related.get("connections"):
                connection_names = [c["node"] for c in related["connections"][:5]]
                kg_context.append(f"{entity}: related to {', '.join(connection_names)}")
        
        # Check contraindications if constraints exist
        safety_notes = []
        for entity in state.get("entities", []):
            for constraint in state.get("constraints", []):
                check = self.kg.check_contraindication(entity, [constraint])
                if not check.get("safe", True):
                    safety_notes.append(f"⚠️ {entity} + {constraint}: {check.get('reason', 'potential concern')}")
        
        prompt = f"""Create a content outline for this Ayurveda content request.

Query: {state['query']}
Content Type: {state['content_type']}
Intent: {state['intent']}
Entities: {', '.join(state.get('entities', []))}
Dosha Context: {state.get('dosha_context', 'Not specified')}

Knowledge Graph Context:
{chr(10).join(kg_context) if kg_context else 'No additional context'}

Safety Notes:
{chr(10).join(safety_notes) if safety_notes else 'No safety concerns identified'}

Create a JSON outline:
{{
    "title": "Suggested title",
    "sections": [
        {{
            "heading": "Section heading",
            "key_points": ["point 1", "point 2"],
            "requires_citation": true/false,
            "safety_critical": true/false
        }}
    ],
    "safety_section_required": true/false,
    "recommended_length": "short|medium|long"
}}

Guidelines:
- Always include safety/contraindications section for product/treatment content
- Mark sections requiring citations (factual claims, benefits, usage instructions)
- Structure logically: Introduction → Main Content → Safety → Conclusion
"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    response_mime_type="application/json"
                )
            )
            
            result = json.loads(response.text)
            
            outline = result.get("sections", [])
            
            return {
                **state,
                "outline": outline,
                "kg_enhancements": kg_context + safety_notes,
                "messages": [f"✓ Outline created: {len(outline)} sections, safety_required={result.get('safety_section_required', False)}"]
            }
            
        except Exception as e:
            logger.error(f"Outline generation failed: {e}")
            # Fallback outline
            return {
                **state,
                "outline": [
                    {"heading": "Introduction", "key_points": ["Overview of topic"], "requires_citation": False},
                    {"heading": "Main Content", "key_points": ["Key information"], "requires_citation": True},
                    {"heading": "Conclusion", "key_points": ["Summary"], "requires_citation": False}
                ],
                "kg_enhancements": kg_context,
                "error": str(e),
                "messages": [f"⚠ Outline generation error, using fallback: {str(e)}"]
            }


class WriterAgent:
    """
    Agent 3: Generates content using CRAG retrieval and outline.
    
    Responsibilities:
    - Retrieve relevant chunks using CRAG
    - Generate content following outline structure
    - Include inline citations
    - Use appropriate Ayurvedic terminology
    """
    
    def __init__(self, config: Config, crag: CorrectiveRAG):
        self.config = config
        genai.configure(api_key=config.google.api_key)
        self.model_name = config.google.model
        self.model = genai.GenerativeModel(self.model_name)
        self.crag = crag
    
    def __call__(self, state: WorkflowState) -> WorkflowState:
        """Generate content with CRAG-retrieved context."""
        logger.info(f"WriterAgent generating content for {len(state.get('outline', []))} sections")
        
        # Retrieve relevant content using CRAG
        crag_result = self.crag.retrieve_with_correction(state["query"], k=8)
        
        # Format retrieved chunks for context
        retrieved_context = []
        citations = []
        for i, doc in enumerate(crag_result.documents):
            chunk_info = {
                "id": i + 1,
                "text": doc.chunk.text[:500],  # Truncate for prompt
                "source": doc.chunk.metadata.get("source", "Unknown"),
                "type": doc.chunk.metadata.get("doc_type", "Unknown"),
                "score": doc.score
            }
            retrieved_context.append(chunk_info)
            citations.append({
                "id": i + 1,
                "source": chunk_info["source"],
                "text_snippet": doc.chunk.text[:100]
            })
        
        context_text = "\n\n".join([
            f"[Source {c['id']}] ({c['source']}, {c['type']}):\n{c['text']}"
            for c in retrieved_context
        ])
        
        outline_text = "\n".join([
            f"## {section.get('heading', 'Section')}\n- " + "\n- ".join(section.get('key_points', []))
            for section in state.get("outline", [])
        ])
        
        prompt = f"""Write content for this Ayurveda article based on the outline and retrieved sources.

QUERY: {state['query']}
CONTENT TYPE: {state['content_type']}
INTENT: {state['intent']}

OUTLINE:
{outline_text}

RETRIEVED SOURCES (use [Source N] for citations):
{context_text}

KNOWLEDGE GRAPH CONTEXT:
{chr(10).join(state.get('kg_enhancements', []))}

WRITING GUIDELINES:
1. Follow the outline structure exactly
2. Include citations using [Source N] format for factual claims
3. Use Kerala Ayurveda brand voice:
   - "Traditionally used to support..." (not "cures" or "treats")
   - "May help maintain..." (not definitive claims)
   - Invitational, warm, educational tone
4. Include safety information where relevant
5. Be accurate - only include information from the sources
6. If information is not in sources, say "According to traditional Ayurvedic texts..."

Write the complete content now:
"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.4,
                    max_output_tokens=2000
                )
            )
            
            draft = response.text
            
            return {
                **state,
                "draft": draft,
                "citations": citations,
                "retrieved_chunks": [{"text": c["text"], "source": c["source"]} for c in retrieved_context],
                "messages": [f"✓ Draft generated: {len(draft)} chars, {len(citations)} sources cited, CRAG status={crag_result.status}"]
            }
            
        except Exception as e:
            logger.error(f"Writer agent failed: {e}")
            return {
                **state,
                "draft": "",
                "citations": [],
                "retrieved_chunks": [],
                "error": str(e),
                "messages": [f"⚠ Writer error: {str(e)}"]
            }


class FactCheckerAgent:
    """
    Agent 4: Multi-method hallucination detection and fact verification.
    
    Responsibilities:
    - Extract factual claims from draft
    - Verify each claim against retrieved sources
    - Calculate hallucination score
    - Flag unsupported claims
    """
    
    def __init__(self, config: Config):
        self.config = config
        genai.configure(api_key=config.google.api_key)
        self.model_name = config.google.model
        self.model = genai.GenerativeModel(self.model_name)
        self.hallucination_threshold = config.agent.hallucination_threshold
    
    def __call__(self, state: WorkflowState) -> WorkflowState:
        """Verify facts in the draft against sources."""
        logger.info("FactCheckerAgent verifying claims...")
        
        if not state.get("draft"):
            return {
                **state,
                "verification_report": {"error": "No draft to verify"},
                "hallucination_score": 1.0,
                "flagged_claims": [],
                "should_revise": True,
                "messages": ["⚠ No draft to verify"]
            }
        
        # Format sources for verification
        sources_text = "\n\n".join([
            f"Source {i+1}: {chunk['text'][:400]}"
            for i, chunk in enumerate(state.get("retrieved_chunks", []))
        ])
        
        prompt = f"""You are a fact-checker for Ayurveda content. Verify the claims in this draft against the provided sources.

DRAFT TO VERIFY:
{state['draft'][:3000]}

AVAILABLE SOURCES:
{sources_text}

For each factual claim in the draft, determine if it is:
1. SUPPORTED - Directly supported by the sources
2. PARTIALLY_SUPPORTED - Related information exists but claim is extrapolated
3. UNSUPPORTED - No supporting evidence in sources
4. CONTRADICTED - Sources contradict the claim

Return JSON:
{{
    "claims": [
        {{
            "claim": "The factual claim text",
            "verdict": "SUPPORTED|PARTIALLY_SUPPORTED|UNSUPPORTED|CONTRADICTED",
            "evidence": "Quote or reference from source if supported",
            "severity": "low|medium|high"
        }}
    ],
    "overall_assessment": {{
        "supported_count": N,
        "unsupported_count": N,
        "hallucination_score": 0.0-1.0,
        "safety_concerns": ["any safety-related issues"]
    }}
}}

Focus especially on:
- Health benefit claims
- Dosage or usage instructions
- Contraindication statements
- Dosha-related claims
"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )
            
            result = json.loads(response.text)
            
            claims = result.get("claims", [])
            assessment = result.get("overall_assessment", {})
            hallucination_score = assessment.get("hallucination_score", 0.5)
            
            # Flag problematic claims
            flagged = [
                claim for claim in claims
                if claim.get("verdict") in ["UNSUPPORTED", "CONTRADICTED"]
                or claim.get("severity") == "high"
            ]
            
            should_revise = (
                hallucination_score > self.hallucination_threshold
                or len(flagged) > 2
                or any(c.get("severity") == "high" for c in flagged)
            )
            
            return {
                **state,
                "verification_report": result,
                "hallucination_score": hallucination_score,
                "flagged_claims": flagged,
                "should_revise": should_revise and state.get("revision_count", 0) < 3,
                "messages": [f"✓ Fact check complete: hallucination_score={hallucination_score:.2f}, {len(flagged)} claims flagged, revise={should_revise}"]
            }
            
        except Exception as e:
            logger.error(f"Fact checker failed: {e}")
            return {
                **state,
                "verification_report": {"error": str(e)},
                "hallucination_score": 0.5,
                "flagged_claims": [],
                "should_revise": False,
                "messages": [f"⚠ Fact check error: {str(e)}"]
            }


class ReflectionAgent:
    """
    Agent 5: Self-healing feedback loops for content improvement.
    
    Responsibilities:
    - Analyze verification results
    - Generate specific revision instructions
    - Track revision iterations
    - Decide when to stop revising
    """
    
    def __init__(self, config: Config):
        self.config = config
        genai.configure(api_key=config.google.api_key)
        self.model_name = config.google.model
        self.model = genai.GenerativeModel(self.model_name)
        self.max_iterations = config.agent.max_reflection_iterations
    
    def __call__(self, state: WorkflowState) -> WorkflowState:
        """Generate revision feedback based on verification."""
        logger.info(f"ReflectionAgent analyzing (iteration {state.get('revision_count', 0) + 1})")
        
        if not state.get("should_revise"):
            return {
                **state,
                "feedback": [],
                "messages": ["✓ No revision needed, proceeding to style editing"]
            }
        
        # Check iteration limit
        current_revision = state.get("revision_count", 0)
        if current_revision >= self.max_iterations:
            return {
                **state,
                "should_revise": False,
                "feedback": ["Max revision iterations reached"],
                "messages": [f"⚠ Max revisions ({self.max_iterations}) reached, proceeding with current draft"]
            }
        
        flagged_claims = state.get("flagged_claims", [])
        
        prompt = f"""Analyze these fact-checking results and provide specific revision instructions.

FLAGGED CLAIMS:
{json.dumps(flagged_claims, indent=2)}

HALLUCINATION SCORE: {state.get('hallucination_score', 0)}

AVAILABLE SOURCES:
{chr(10).join([f"- {chunk['source']}: {chunk['text'][:200]}..." for chunk in state.get('retrieved_chunks', [])[:5]])}

Provide specific, actionable feedback for revising the content:
1. Which claims should be removed or modified?
2. What information from sources should be used instead?
3. Are there any safety concerns that need addressing?

Return JSON:
{{
    "feedback": [
        "Specific instruction 1",
        "Specific instruction 2"
    ],
    "priority_fixes": ["most critical fixes"],
    "suggested_additions": ["information from sources to add"]
}}
"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    response_mime_type="application/json"
                )
            )
            
            result = json.loads(response.text)
            feedback = result.get("feedback", []) + result.get("priority_fixes", [])
            
            return {
                **state,
                "feedback": feedback,
                "revision_count": current_revision + 1,
                "messages": [f"✓ Reflection complete: {len(feedback)} revision items, iteration {current_revision + 1}"]
            }
            
        except Exception as e:
            logger.error(f"Reflection agent failed: {e}")
            return {
                **state,
                "feedback": [],
                "should_revise": False,
                "messages": [f"⚠ Reflection error: {str(e)}"]
            }


class StyleEditorAgent:
    """
    Agent 6: Brand voice alignment and final polish.
    
    Responsibilities:
    - Ensure Kerala Ayurveda brand voice
    - Check for forbidden phrases
    - Add appropriate disclaimers
    - Final formatting and polish
    """
    
    FORBIDDEN_PHRASES = [
        "cures", "treats", "heals disease", "100% safe",
        "miracle", "guaranteed results", "no side effects",
        "replace medication", "stop taking medicine"
    ]
    
    REQUIRED_ELEMENTS = [
        "traditionally", "may support", "consult healthcare provider"
    ]
    
    def __init__(self, config: Config):
        self.config = config
        genai.configure(api_key=config.google.api_key)
        self.model_name = config.google.model
        self.model = genai.GenerativeModel(self.model_name)
    
    def __call__(self, state: WorkflowState) -> WorkflowState:
        """Apply brand voice and final editing."""
        logger.info("StyleEditorAgent applying brand voice...")
        
        draft = state.get("draft", "")
        
        # Check for forbidden phrases
        violations = []
        for phrase in self.FORBIDDEN_PHRASES:
            if phrase.lower() in draft.lower():
                violations.append(f"Contains forbidden phrase: '{phrase}'")
        
        prompt = f"""Edit this Ayurveda content to align with Kerala Ayurveda brand voice.

CURRENT DRAFT:
{draft[:3000]}

BRAND VOICE GUIDELINES:
1. Use "traditionally used to support..." instead of "treats" or "cures"
2. Use "may help maintain..." instead of definitive claims
3. Use warm, invitational tone: "You may find..." "Consider exploring..."
4. Include safety reminder: "Consult with a healthcare provider..."
5. Never dismiss modern medicine
6. Be educational and empowering, not prescriptive

VIOLATIONS FOUND:
{chr(10).join(violations) if violations else 'None'}

REVISION FEEDBACK TO ADDRESS:
{chr(10).join(state.get('feedback', [])) if state.get('feedback') else 'None'}

Edit the content to:
1. Fix any brand voice violations
2. Address the revision feedback
3. Ensure proper disclaimers are included
4. Maintain accuracy of information
5. Polish for readability

Return the edited content only, no explanations.
"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=2500
                )
            )
            
            final_output = response.text
            
            # Calculate brand alignment score
            alignment_score = 1.0
            alignment_score -= len(violations) * 0.1
            
            # Check for required elements
            for element in self.REQUIRED_ELEMENTS:
                if element.lower() in final_output.lower():
                    alignment_score += 0.05
            
            alignment_score = max(0.0, min(1.0, alignment_score))
            
            return {
                **state,
                "final_output": final_output,
                "brand_alignment_score": alignment_score,
                "should_revise": False,  # Style editing is final
                "messages": [f"✓ Style editing complete: brand_alignment={alignment_score:.2f}, {len(violations)} violations fixed"]
            }
            
        except Exception as e:
            logger.error(f"Style editor failed: {e}")
            return {
                **state,
                "final_output": draft,  # Use draft as fallback
                "brand_alignment_score": 0.5,
                "messages": [f"⚠ Style editor error, using draft: {str(e)}"]
            }


class RevisionWriterAgent:
    """
    Sub-agent for revising content based on reflection feedback.
    """
    
    def __init__(self, config: Config, crag: CorrectiveRAG):
        self.config = config
        genai.configure(api_key=config.google.api_key)
        self.model_name = config.google.model
        self.model = genai.GenerativeModel(self.model_name)
        self.crag = crag
    
    def __call__(self, state: WorkflowState) -> WorkflowState:
        """Revise draft based on feedback."""
        logger.info(f"RevisionWriterAgent revising draft (iteration {state.get('revision_count', 0)})")
        
        feedback = state.get("feedback", [])
        if not feedback:
            return state
        
        sources_text = "\n".join([
            f"Source: {chunk['source']}\n{chunk['text'][:300]}"
            for chunk in state.get("retrieved_chunks", [])[:5]
        ])
        
        prompt = f"""Revise this Ayurveda content based on the feedback provided.

CURRENT DRAFT:
{state.get('draft', '')[:2500]}

REVISION FEEDBACK:
{chr(10).join(f"- {f}" for f in feedback)}

FLAGGED CLAIMS TO FIX:
{json.dumps(state.get('flagged_claims', []), indent=2)}

AVAILABLE SOURCES FOR ACCURATE INFORMATION:
{sources_text}

Revise the content to:
1. Address all feedback points
2. Remove or fix flagged claims
3. Use information from sources only
4. Maintain the overall structure and flow
5. Keep citations where appropriate

Return the revised content only.
"""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=2000
                )
            )
            
            revised_draft = response.text
            
            return {
                **state,
                "draft": revised_draft,
                "messages": [f"✓ Draft revised based on {len(feedback)} feedback items"]
            }
            
        except Exception as e:
            logger.error(f"Revision failed: {e}")
            return {
                **state,
                "messages": [f"⚠ Revision error: {str(e)}"]
            }


class AyurvedaAgentWorkflow:
    """
    Main workflow orchestrator using LangGraph.
    
    Workflow:
    Query → Understanding → Outline → Write → FactCheck → Reflect ↺ → StyleEdit → Output
                                              ↑_______revision loop_______↓
    """
    
    def __init__(
        self,
        config: Config,
        retriever: HybridRetriever,
        crag: CorrectiveRAG,
        knowledge_graph: AyurvedaKnowledgeGraph
    ):
        self.config = config
        self.retriever = retriever
        self.crag = crag
        self.kg = knowledge_graph
        
        # Initialize agents
        self.query_agent = QueryUnderstandingAgent(config)
        self.outline_agent = OutlineGeneratorAgent(config, knowledge_graph)
        self.writer_agent = WriterAgent(config, crag)
        self.fact_checker = FactCheckerAgent(config)
        self.reflection_agent = ReflectionAgent(config)
        self.style_editor = StyleEditorAgent(config)
        self.revision_writer = RevisionWriterAgent(config, crag)
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        # Create the graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("query_understanding", self.query_agent)
        workflow.add_node("outline_generator", self.outline_agent)
        workflow.add_node("writer", self.writer_agent)
        workflow.add_node("fact_checker", self.fact_checker)
        workflow.add_node("reflection", self.reflection_agent)
        workflow.add_node("revision_writer", self.revision_writer)
        workflow.add_node("style_editor", self.style_editor)
        
        # Add edges
        workflow.set_entry_point("query_understanding")
        workflow.add_edge("query_understanding", "outline_generator")
        workflow.add_edge("outline_generator", "writer")
        workflow.add_edge("writer", "fact_checker")
        workflow.add_edge("fact_checker", "reflection")
        
        # Conditional edge: revision loop or proceed to style editing
        workflow.add_conditional_edges(
            "reflection",
            self._should_revise,
            {
                "revise": "revision_writer",
                "proceed": "style_editor"
            }
        )
        
        workflow.add_edge("revision_writer", "fact_checker")  # Loop back
        workflow.add_edge("style_editor", END)
        
        # Compile with memory for checkpointing
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    
    def _should_revise(self, state: WorkflowState) -> Literal["revise", "proceed"]:
        """Determine if revision is needed."""
        if state.get("should_revise", False) and state.get("revision_count", 0) < 3:
            return "revise"
        return "proceed"
    
    def run(
        self,
        query: str,
        content_type: str = "article",
        thread_id: str = "default"
    ) -> dict:
        """
        Run the complete agent workflow.
        
        Args:
            query: User query/brief for content generation
            content_type: Type of content (article, faq, product_description)
            thread_id: Unique ID for conversation tracking
            
        Returns:
            dict with final_output, verification_report, and workflow_messages
        """
        logger.info(f"Starting workflow for: {query[:50]}...")
        
        # Initial state
        initial_state: WorkflowState = {
            "query": query,
            "content_type": content_type,
            "intent": "",
            "entities": [],
            "constraints": [],
            "dosha_context": None,
            "outline": [],
            "kg_enhancements": [],
            "draft": "",
            "citations": [],
            "retrieved_chunks": [],
            "verification_report": None,
            "hallucination_score": 0.0,
            "flagged_claims": [],
            "feedback": [],
            "revision_count": 0,
            "final_output": "",
            "brand_alignment_score": 0.0,
            "should_revise": False,
            "error": None,
            "messages": []
        }
        
        # Run the workflow
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            final_state = None
            for state in self.graph.stream(initial_state, config):
                # Get the latest state from the stream
                for node_name, node_state in state.items():
                    final_state = node_state
                    logger.debug(f"Completed node: {node_name}")
            
            if final_state is None:
                raise ValueError("Workflow produced no output")
            
            return {
                "final_output": final_state.get("final_output", ""),
                "draft": final_state.get("draft", ""),
                "verification_report": final_state.get("verification_report"),
                "hallucination_score": final_state.get("hallucination_score", 0),
                "brand_alignment_score": final_state.get("brand_alignment_score", 0),
                "citations": final_state.get("citations", []),
                "revision_count": final_state.get("revision_count", 0),
                "workflow_messages": final_state.get("messages", []),
                "entities": final_state.get("entities", []),
                "intent": final_state.get("intent", "")
            }
            
        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            return {
                "final_output": "",
                "error": str(e),
                "workflow_messages": [f"Workflow error: {str(e)}"]
            }
    
    def run_simple(self, query: str) -> str:
        """Simple interface - returns just the final content."""
        result = self.run(query)
        return result.get("final_output", result.get("draft", "Error generating content"))


# Convenience function for quick testing
def create_workflow(config: Config = None) -> AyurvedaAgentWorkflow:
    """Create a workflow with default configuration."""
    if config is None:
        config = Config()
    
    # These would normally be initialized with indexed data
    retriever = HybridRetriever(config)
    crag = CorrectiveRAG(config, retriever)
    kg = AyurvedaKnowledgeGraph()
    
    return AyurvedaAgentWorkflow(config, retriever, crag, kg)
