"""
Gradio UI for Kerala Ayurveda RAG System.

A user-friendly interface for:
- Content generation with agent workflow
- Simple Q&A queries
- Safety/contraindication checks
- Response evaluation
"""

import gradio as gr
import time
from typing import Optional
import logging

from src.config import Config
from src.data_processing.document_loader import DocumentLoader
from src.data_processing.chunker import AyurvedaChunker
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.corrective_rag import CorrectiveRAG
from src.knowledge_graph.ayurveda_kg import AyurvedaKnowledgeGraph
from src.agents.langgraph_workflow import AyurvedaAgentWorkflow
from src.evaluation.metrics import RAGEvaluator

logger = logging.getLogger(__name__)


class RAGInterface:
    """Main interface class managing all RAG components."""
    
    def __init__(self):
        self.config = None
        self.retriever = None
        self.crag = None
        self.kg = None
        self.workflow = None
        self.evaluator = None
        self.chunks = []
        self.is_initialized = False
    
    def initialize(self, progress=gr.Progress()):
        """Initialize all components."""
        if self.is_initialized:
            return "✅ System already initialized!"
        
        try:
            progress(0.1, desc="Loading configuration...")
            self.config = Config()
            
            progress(0.2, desc="Initializing retriever...")
            self.retriever = HybridRetriever(self.config)
            
            progress(0.3, desc="Initializing CRAG...")
            self.crag = CorrectiveRAG(self.config, self.retriever)
            
            progress(0.4, desc="Building Knowledge Graph...")
            self.kg = AyurvedaKnowledgeGraph()
            
            progress(0.5, desc="Loading documents...")
            loader = DocumentLoader(self.config)
            chunker = AyurvedaChunker(self.config)
            
            documents = loader.load_all_documents()
            
            progress(0.6, desc="Chunking documents...")
            for doc in documents:
                chunks = chunker.chunk_document(doc)
                self.chunks.extend(chunks)
            
            progress(0.7, desc="Building indexes...")
            self.retriever.index_chunks(self.chunks)
            
            progress(0.8, desc="Initializing agent workflow...")
            self.workflow = AyurvedaAgentWorkflow(
                self.config, self.retriever, self.crag, self.kg
            )
            
            progress(0.9, desc="Initializing evaluator...")
            self.evaluator = RAGEvaluator(self.config)
            
            self.is_initialized = True
            
            progress(1.0, desc="Done!")
            return f"""✅ System initialized successfully!
            
📊 Stats:
- Documents loaded: {len(documents)}
- Chunks indexed: {len(self.chunks)}
- Knowledge Graph nodes: {self.kg.graph.number_of_nodes()}
- Knowledge Graph edges: {self.kg.graph.number_of_edges()}
"""
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return f"❌ Initialization failed: {str(e)}"
    
    def generate_content(
        self,
        brief: str,
        content_type: str,
        use_agents: bool,
        progress=gr.Progress()
    ):
        """Generate content from a brief."""
        if not self.is_initialized:
            return "❌ System not initialized. Click 'Initialize System' first.", "", ""
        
        if not brief.strip():
            return "❌ Please enter a content brief.", "", ""
        
        start_time = time.time()
        
        try:
            if use_agents and self.workflow:
                progress(0.3, desc="Running agent workflow...")
                result = self.workflow.run(
                    query=brief,
                    content_type=content_type
                )
                
                content = result.get("final_output", result.get("draft", ""))
                
                # Format metadata
                metadata = f"""📊 Generation Metrics:
- Hallucination Score: {result.get('hallucination_score', 0):.2f}
- Brand Alignment: {result.get('brand_alignment_score', 0):.2f}
- Revisions Made: {result.get('revision_count', 0)}
- Entities Found: {', '.join(result.get('entities', []))}
- Intent: {result.get('intent', 'N/A')}
- Processing Time: {time.time() - start_time:.2f}s
"""
                
                # Format workflow log
                workflow_log = "\n".join(result.get("workflow_messages", []))
                
                progress(1.0, desc="Done!")
                return content, metadata, workflow_log
                
            else:
                progress(0.3, desc="Running CRAG retrieval...")
                crag_result = self.crag.retrieve_with_correction(brief, k=5)
                
                context = "\n\n---\n\n".join([
                    f"**Source:** {doc.chunk.metadata.get('source', 'Unknown')}\n\n{doc.chunk.text}"
                    for doc in crag_result.documents
                ])
                
                metadata = f"""📊 Retrieval Metrics:
- CRAG Status: {crag_result.status}
- Confidence: {crag_result.confidence:.2f}
- Documents Retrieved: {len(crag_result.documents)}
- Processing Time: {time.time() - start_time:.2f}s
"""
                
                progress(1.0, desc="Done!")
                return context, metadata, f"CRAG Action: {crag_result.action_taken}"
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"❌ Generation failed: {str(e)}", "", ""
    
    def query(self, question: str, use_crag: bool, top_k: int):
        """Answer a question using the knowledge base."""
        if not self.is_initialized:
            return "❌ System not initialized.", ""
        
        if not question.strip():
            return "❌ Please enter a question.", ""
        
        start_time = time.time()
        
        try:
            if use_crag:
                result = self.crag.retrieve_with_correction(question, k=top_k)
                
                answer = "\n\n---\n\n".join([
                    f"**[{i+1}] {doc.chunk.metadata.get('source', 'Unknown')}** (Score: {doc.score:.3f})\n\n{doc.chunk.text}"
                    for i, doc in enumerate(result.documents)
                ])
                
                metadata = f"""📊 CRAG Status: {result.status}
Confidence: {result.confidence:.2f}
Action: {result.action_taken}
Reason: {result.reason}
Time: {time.time() - start_time:.2f}s"""
                
            else:
                results = self.retriever.search(question, k=top_k)
                
                answer = "\n\n---\n\n".join([
                    f"**[{i+1}] {r.chunk.metadata.get('source', 'Unknown')}** (Score: {r.score:.3f})\n\n{r.chunk.text}"
                    for i, r in enumerate(results)
                ])
                
                metadata = f"""📊 Direct Retrieval
Documents Found: {len(results)}
Time: {time.time() - start_time:.2f}s"""
            
            return answer, metadata
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return f"❌ Query failed: {str(e)}", ""
    
    def check_safety(self, herb: str, conditions: str):
        """Check contraindications."""
        if not self.is_initialized:
            return "❌ System not initialized."
        
        if not herb.strip():
            return "❌ Please enter an herb or product name."
        
        try:
            # Parse conditions
            condition_list = [c.strip() for c in conditions.split(",") if c.strip()]
            
            if not condition_list:
                condition_list = ["general"]
            
            result = self.kg.check_contraindication(herb.strip(), condition_list)
            
            status = "✅ SAFE" if result.get("safe", True) else "⚠️ CAUTION"
            severity = result.get("severity", "LOW")
            reason = result.get("reason", "No specific contraindications found")
            
            # Get related entities
            related = self.kg.get_related_entities(herb.strip(), radius=2)
            
            return f"""## {status}

**Herb/Product:** {herb}
**Conditions Checked:** {', '.join(condition_list)}
**Severity:** {severity}

### Assessment
{reason}

### Related Entities in Knowledge Graph
{', '.join(related) if related else 'No related entities found'}

### Recommendations
{"⚠️ Consult with a healthcare provider before use." if not result.get("safe", True) else "✅ No known contraindications for the specified conditions. Always consult with a healthcare provider for personalized advice."}
"""
            
        except Exception as e:
            logger.error(f"Safety check failed: {e}")
            return f"❌ Safety check failed: {str(e)}"
    
    def evaluate_response(self, query: str, response: str, context: str):
        """Evaluate a response."""
        if not self.is_initialized:
            return "❌ System not initialized."
        
        if not query.strip() or not response.strip():
            return "❌ Please provide both query and response."
        
        try:
            context_list = [c.strip() for c in context.split("---") if c.strip()]
            
            result = self.evaluator.evaluate(
                query=query,
                response=response,
                context=context_list
            )
            
            status = "✅ PASSED" if result.pass_threshold else "❌ FAILED"
            
            return f"""## Evaluation Result: {status}

### Overall Score: {result.overall_score:.2f}

### Metrics Breakdown

| Metric | Score |
|--------|-------|
| Faithfulness | {result.faithfulness:.2f} |
| Answer Relevancy | {result.answer_relevancy:.2f} |
| Context Precision | {result.context_precision:.2f} |
| Hallucination Score | {result.hallucination_score:.2f} |
| Ayurveda Accuracy | {result.ayurveda_accuracy:.2f} |
| Brand Alignment | {result.brand_alignment:.2f} |

### Issues Found

**Unsupported Claims:** {len(result.unsupported_claims)}
{chr(10).join(['- ' + c for c in result.unsupported_claims]) if result.unsupported_claims else '- None detected'}

**Forbidden Phrases:** {len(result.forbidden_phrases_found)}
{chr(10).join(['- ' + p for p in result.forbidden_phrases_found]) if result.forbidden_phrases_found else '- None detected'}

### Recommendation
{"✅ Response meets quality thresholds." if result.pass_threshold else "⚠️ Response needs revision. Focus on reducing hallucinations and improving source grounding."}
"""
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return f"❌ Evaluation failed: {str(e)}"


# ============ Create Interface ============

def create_interface():
    """Create the Gradio interface."""
    
    rag = RAGInterface()
    
    with gr.Blocks(
        title="Kerala Ayurveda RAG System"
    ) as demo:
        
        gr.Markdown("""
        # 🌿 Kerala Ayurveda RAG System
        
        **Production-ready content generation with CRAG, Knowledge Graphs, and Multi-Agent workflows.**
        
        ---
        """)
        
        # Initialize button
        with gr.Row():
            init_btn = gr.Button("🚀 Initialize System", variant="primary", scale=2)
            init_status = gr.Textbox(label="Status", lines=6, interactive=False)
        
        init_btn.click(rag.initialize, outputs=init_status)
        
        gr.Markdown("---")
        
        # Main tabs
        with gr.Tabs():
            
            # Tab 1: Content Generation
            with gr.TabItem("📝 Generate Content"):
                gr.Markdown("### Generate Ayurveda content with AI agents")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        gen_brief = gr.Textbox(
                            label="Content Brief",
                            placeholder="Write about the benefits of Ashwagandha for stress management...",
                            lines=3
                        )
                        
                        with gr.Row():
                            gen_type = gr.Dropdown(
                                choices=["article", "faq", "product_description"],
                                value="article",
                                label="Content Type"
                            )
                            gen_agents = gr.Checkbox(
                                value=True,
                                label="Use Multi-Agent Workflow"
                            )
                        
                        gen_btn = gr.Button("✨ Generate", variant="primary")
                    
                    with gr.Column(scale=1):
                        gen_metadata = gr.Textbox(label="Metrics", lines=8, interactive=False)
                
                gen_output = gr.Markdown(label="Generated Content")
                gen_log = gr.Textbox(label="Workflow Log", lines=5, interactive=False)
                
                gen_btn.click(
                    rag.generate_content,
                    inputs=[gen_brief, gen_type, gen_agents],
                    outputs=[gen_output, gen_metadata, gen_log]
                )
            
            # Tab 2: Q&A
            with gr.TabItem("❓ Ask Questions"):
                gr.Markdown("### Ask questions about Ayurveda")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        qa_question = gr.Textbox(
                            label="Your Question",
                            placeholder="What are the contraindications for Triphala?",
                            lines=2
                        )
                        
                        with gr.Row():
                            qa_crag = gr.Checkbox(value=True, label="Use CRAG")
                            qa_topk = gr.Slider(1, 10, value=5, step=1, label="Results")
                        
                        qa_btn = gr.Button("🔍 Search", variant="primary")
                    
                    with gr.Column(scale=1):
                        qa_metadata = gr.Textbox(label="Search Info", lines=6, interactive=False)
                
                qa_answer = gr.Markdown(label="Answer")
                
                qa_btn.click(
                    rag.query,
                    inputs=[qa_question, qa_crag, qa_topk],
                    outputs=[qa_answer, qa_metadata]
                )
            
            # Tab 3: Safety Check
            with gr.TabItem("⚠️ Safety Check"):
                gr.Markdown("### Check contraindications for herbs and products")
                
                with gr.Row():
                    safety_herb = gr.Textbox(
                        label="Herb or Product",
                        placeholder="Ashwagandha",
                        scale=1
                    )
                    safety_conditions = gr.Textbox(
                        label="Conditions (comma-separated)",
                        placeholder="pregnancy, thyroid_condition, autoimmune",
                        scale=2
                    )
                
                safety_btn = gr.Button("🔍 Check Safety", variant="primary")
                safety_result = gr.Markdown(label="Safety Assessment")
                
                safety_btn.click(
                    rag.check_safety,
                    inputs=[safety_herb, safety_conditions],
                    outputs=safety_result
                )
            
            # Tab 4: Evaluate
            with gr.TabItem("📊 Evaluate Response"):
                gr.Markdown("### Evaluate generated content quality")
                
                eval_query = gr.Textbox(label="Original Query", lines=2)
                eval_response = gr.Textbox(label="Response to Evaluate", lines=5)
                eval_context = gr.Textbox(
                    label="Context (separate with ---)",
                    lines=3,
                    placeholder="Context chunk 1\n---\nContext chunk 2"
                )
                
                eval_btn = gr.Button("📊 Evaluate", variant="primary")
                eval_result = gr.Markdown(label="Evaluation Results")
                
                eval_btn.click(
                    rag.evaluate_response,
                    inputs=[eval_query, eval_response, eval_context],
                    outputs=eval_result
                )
        
        gr.Markdown("""
        ---
        
        ### 📚 About This System
        
        This RAG system uses:
        - **CRAG (Corrective RAG)**: Self-healing retrieval with confidence-based actions
        - **Hybrid Search**: BM25 + Dense embeddings with Reciprocal Rank Fusion
        - **Knowledge Graph**: 3-tier Ayurveda ontology (Doshas → Herbs → Products)
        - **Multi-Agent Workflow**: 6 specialized agents with reflection loops
        - **Comprehensive Evaluation**: RAGAS metrics + hallucination detection
        
        Built for Kerala Ayurveda | 2025
        """)
    
    return demo


# ============ Main ============

def launch_ui(share: bool = False, port: int = 7860):
    """Launch the Gradio interface."""
    demo = create_interface()
    demo.launch(share=share, server_port=port)


if __name__ == "__main__":
    launch_ui()
