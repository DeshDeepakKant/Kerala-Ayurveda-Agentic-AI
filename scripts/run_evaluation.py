#!/usr/bin/env python3
"""
Run comprehensive evaluation of the Kerala Ayurveda RAG System.

Usage:
    python run_evaluation.py [--golden] [--custom FILE]
"""

import argparse
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import Config
from src.data_processing.document_loader import DocumentLoader
from src.data_processing.chunker import AyurvedaChunker
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.corrective_rag import CorrectiveRAG
from src.knowledge_graph.ayurveda_kg import AyurvedaKnowledgeGraph
from src.agents.langgraph_workflow import AyurvedaAgentWorkflow
from src.evaluation.metrics import RAGEvaluator, GoldenDatasetEvaluator


def initialize_system():
    """Initialize all RAG components."""
    print("🔧 Initializing system...")
    
    config = Config()
    
    # Load and index documents
    loader = DocumentLoader(config)
    chunker = AyurvedaChunker(config)
    
    documents = loader.load_all_documents()
    chunks = []
    for doc in documents:
        chunks.extend(chunker.chunk_document(doc))
    
    # Initialize components
    retriever = HybridRetriever(config)
    retriever.index_chunks(chunks)
    
    crag = CorrectiveRAG(config, retriever)
    kg = AyurvedaKnowledgeGraph()
    workflow = AyurvedaAgentWorkflow(config, retriever, crag, kg)
    evaluator = RAGEvaluator(config)
    golden_evaluator = GoldenDatasetEvaluator(config)
    
    print(f"✅ System initialized: {len(documents)} docs, {len(chunks)} chunks")
    
    return {
        "config": config,
        "retriever": retriever,
        "crag": crag,
        "kg": kg,
        "workflow": workflow,
        "evaluator": evaluator,
        "golden_evaluator": golden_evaluator,
        "chunks": chunks
    }


def run_golden_evaluation(system: dict):
    """Run evaluation on golden test cases."""
    print("\n" + "=" * 60)
    print("📊 Running Golden Dataset Evaluation")
    print("=" * 60)
    
    def generate_fn(query):
        result = system["workflow"].run(query)
        return result.get("final_output", result.get("draft", ""))
    
    def retrieve_fn(query):
        results = system["retriever"].search(query, k=5)
        return [r.chunk.text for r in results]
    
    results = system["golden_evaluator"].run_golden_evaluation(
        generate_fn, retrieve_fn
    )
    
    print(f"\n📈 Results:")
    print(f"  Total Cases: {results['total_cases']}")
    print(f"  Cases Passed: {results['cases_passed']}")
    print(f"  Pass Rate: {results['cases_passed'] / results['total_cases'] * 100:.1f}%")
    print(f"  Avg Golden Score: {results['avg_golden_score']:.3f}")
    print(f"  Avg Overall Score: {results['avg_overall_score']:.3f}")
    
    return results


def run_sample_evaluation(system: dict):
    """Run evaluation on sample queries."""
    print("\n" + "=" * 60)
    print("📊 Running Sample Query Evaluation")
    print("=" * 60)
    
    sample_queries = [
        "What are the benefits of Ashwagandha for stress?",
        "How should I use Triphala for digestion?",
        "Can Brahmi help with memory and concentration?",
        "What is the best time to take Ayurvedic supplements?",
        "Are there any side effects of Ashwagandha?"
    ]
    
    results = []
    
    for query in sample_queries:
        print(f"\n🔍 Query: {query}")
        
        # Generate response
        workflow_result = system["workflow"].run(query)
        response = workflow_result.get("final_output", workflow_result.get("draft", ""))
        
        # Get context
        retrieval_results = system["retriever"].search(query, k=5)
        context = [r.chunk.text for r in retrieval_results]
        
        # Evaluate
        eval_result = system["evaluator"].evaluate(
            query=query,
            response=response,
            context=context
        )
        
        print(f"  ✓ Overall Score: {eval_result.overall_score:.3f}")
        print(f"  ✓ Faithfulness: {eval_result.faithfulness:.3f}")
        print(f"  ✓ Hallucination: {eval_result.hallucination_score:.3f}")
        print(f"  ✓ Pass: {'✅' if eval_result.pass_threshold else '❌'}")
        
        results.append({
            "query": query,
            "evaluation": eval_result.to_dict()
        })
    
    # Aggregate
    avg_overall = sum(r["evaluation"]["overall_score"] for r in results) / len(results)
    avg_faithfulness = sum(r["evaluation"]["faithfulness"] for r in results) / len(results)
    avg_hallucination = sum(r["evaluation"]["hallucination_score"] for r in results) / len(results)
    pass_count = sum(1 for r in results if r["evaluation"]["pass_threshold"])
    
    print(f"\n📈 Aggregate Results:")
    print(f"  Queries Evaluated: {len(results)}")
    print(f"  Pass Rate: {pass_count}/{len(results)} ({pass_count/len(results)*100:.1f}%)")
    print(f"  Avg Overall Score: {avg_overall:.3f}")
    print(f"  Avg Faithfulness: {avg_faithfulness:.3f}")
    print(f"  Avg Hallucination: {avg_hallucination:.3f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run RAG System Evaluation")
    parser.add_argument("--golden", action="store_true", help="Run golden dataset evaluation")
    parser.add_argument("--sample", action="store_true", help="Run sample query evaluation")
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    
    args = parser.parse_args()
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║         🌿 Kerala Ayurveda RAG - Evaluation Suite            ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # Initialize
    system = initialize_system()
    
    all_results = {}
    
    # Run evaluations
    if args.golden or not (args.golden or args.sample):
        all_results["golden"] = run_golden_evaluation(system)
    
    if args.sample or not (args.golden or args.sample):
        all_results["sample"] = run_sample_evaluation(system)
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {output_path}")
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
