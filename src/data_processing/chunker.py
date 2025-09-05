"""
Advanced chunking strategies for Kerala Ayurveda documents
"""
import re
from typing import List, Dict, Optional
from loguru import logger

from src.models import Document, Chunk, DocumentType
from src.config import config


class AyurvedaChunker:
    """
    Document-type aware chunking for Kerala Ayurveda corpus
    
    Strategies:
    - FAQ: Q&A pair splitting
    - Product: Section-based (benefits/contraindications/usage)
    - Treatment: Phase-based (preparation/procedure/post-care)
    - Foundation: Semantic chunking with context enrichment
    """
    
    def __init__(self, chunk_size: int = 300, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        logger.info("Initialized AyurvedaChunker")
    
    def chunk_document(self, doc: Document) -> List[Chunk]:
        """Route to appropriate chunking strategy based on document type"""
        
        if doc.doc_type == DocumentType.FAQ:
            chunks = self._chunk_faq(doc)
        elif doc.doc_type == DocumentType.PRODUCT:
            chunks = self._chunk_product(doc)
        elif doc.doc_type == DocumentType.TREATMENT:
            chunks = self._chunk_treatment(doc)
        elif doc.doc_type == DocumentType.CATALOG:
            chunks = self._chunk_catalog(doc)
        else:
            # Default: semantic chunking with context enrichment
            chunks = self._chunk_semantic(doc)
        
        logger.debug(f"Chunked document {doc.id} into {len(chunks)} chunks")
        return chunks
    
    def _chunk_faq(self, doc: Document) -> List[Chunk]:
        """
        FAQ chunking: Split by Q&A pairs
        Each Q&A pair becomes one chunk
        """
        chunks = []
        
        # Pattern to match Q&A pairs (## followed by question)
        qa_pattern = r'##\s+\d*\.?\s*(.+?)\n+(.*?)(?=##\s+\d*\.?\s+|$)'
        matches = re.findall(qa_pattern, doc.content, re.DOTALL)
        
        if not matches:
            # Fallback: split by ## headings
            sections = re.split(r'\n##\s+', doc.content)
            for i, section in enumerate(sections[1:], 1):  # Skip first (title)
                lines = section.split('\n', 1)
                question = lines[0].strip()
                answer = lines[1].strip() if len(lines) > 1 else ""
                
                if question and answer:
                    matches.append((question, answer))
        
        for idx, (question, answer) in enumerate(matches, 1):
            # Context enrichment
            enriched_text = f"""=== DOCUMENT CONTEXT ===
Document: {doc.title}
Type: FAQ
Category: General Ayurveda Questions

=== Q&A PAIR ===
Question: {question.strip()}

Answer: {answer.strip()}
"""
            
            chunk = Chunk(
                id=f"{doc.id}_qa_{idx}",
                text=enriched_text,
                doc_id=doc.id,
                doc_title=doc.title,
                doc_type=doc.doc_type,
                section_type="qa_pair",
                metadata={
                    "question": question.strip(),
                    "answer": answer.strip(),
                    "qa_index": idx
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_product(self, doc: Document) -> List[Chunk]:
        """
        Product chunking: Split by sections (benefits/contraindications/usage)
        """
        chunks = []
        
        # Define key sections to extract
        sections = {
            "traditional_positioning": r"##\s+Traditional Positioning(.*?)(?=##|\Z)",
            "key_messages": r"##\s+Key Messages for Content(.*?)(?=##|\Z)",
            "benefits": r"##\s+(?:Benefits|Traditional Positioning)(.*?)(?=##|\Z)",
            "contraindications": r"##\s+(?:Safety|Precautions|Contraindications)(.*?)(?=##|\Z)",
            "usage": r"##\s+(?:Usage|Suggested Use|Dosage)(.*?)(?=##|\Z)",
            "audience": r"##\s+(?:Audience|Suitable Use)(.*?)(?=##|\Z)",
        }
        
        # Try to extract structured sections
        extracted_sections = {}
        for section_name, pattern in sections.items():
            match = re.search(pattern, doc.content, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                if content and len(content) > 20:  # Meaningful content
                    extracted_sections[section_name] = content
        
        # If no structured sections found, fall back to semantic chunking
        if not extracted_sections:
            return self._chunk_semantic(doc)
        
        # Create chunks for each section
        for idx, (section_name, content) in enumerate(extracted_sections.items(), 1):
            # Get product name from metadata or title
            product_name = doc.metadata.get('name', doc.title)
            category = doc.metadata.get('category', 'General')
            
            enriched_text = f"""=== DOCUMENT CONTEXT ===
Product: {product_name}
Category: {category}
Type: Product Information
Section: {section_name.replace('_', ' ').title()}

=== CONTENT ===
{content}
"""
            
            chunk = Chunk(
                id=f"{doc.id}_{section_name}",
                text=enriched_text,
                doc_id=doc.id,
                doc_title=doc.title,
                doc_type=doc.doc_type,
                section_type=section_name,
                metadata={
                    "section": section_name,
                    "product_name": product_name,
                    "category": category,
                    "is_safety_critical": section_name == "contraindications"
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_treatment(self, doc: Document) -> List[Chunk]:
        """
        Treatment chunking: Split by phases (preparation/procedure/post-care)
        """
        chunks = []
        
        # Treatment-specific sections
        phases = {
            "overview": r"##\s+Overview(.*?)(?=##|\Z)",
            "components": r"##\s+(?:Core Components|Components)(.*?)(?=##|\Z)",
            "preparation": r"##\s+(?:Preparation|Initial)(.*?)(?=##|\Z)",
            "procedure": r"##\s+(?:Therapy Plan|Procedure|Main)(.*?)(?=##|\Z)",
            "home_routine": r"##\s+(?:Home Routine|Self-Care)(.*?)(?=##|\Z)",
            "content_angles": r"##\s+Content Angles(.*?)(?=##|\Z)",
        }
        
        extracted_phases = {}
        for phase_name, pattern in phases.items():
            match = re.search(pattern, doc.content, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                if content and len(content) > 20:
                    extracted_phases[phase_name] = content
        
        if not extracted_phases:
            return self._chunk_semantic(doc)
        
        # Create chunks for each phase
        for idx, (phase_name, content) in enumerate(extracted_phases.items(), 1):
            enriched_text = f"""=== DOCUMENT CONTEXT ===
Treatment: {doc.title}
Type: Treatment Protocol
Phase: {phase_name.replace('_', ' ').title()}

=== CONTENT ===
{content}
"""
            
            chunk = Chunk(
                id=f"{doc.id}_{phase_name}",
                text=enriched_text,
                doc_id=doc.id,
                doc_title=doc.title,
                doc_type=doc.doc_type,
                section_type=phase_name,
                metadata={
                    "phase": phase_name,
                    "treatment_type": doc.title
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_catalog(self, doc: Document) -> List[Chunk]:
        """
        Catalog chunking: Keep each product as one enriched chunk
        """
        # For catalog items, keep as single chunk with full context
        enriched_text = f"""=== DOCUMENT CONTEXT ===
Source: Product Catalog
Product ID: {doc.metadata.get('product_id', 'N/A')}
Category: {doc.metadata.get('category', 'N/A')}

=== PRODUCT INFORMATION ===
{doc.content}
"""
        
        chunk = Chunk(
            id=f"{doc.id}_full",
            text=enriched_text,
            doc_id=doc.id,
            doc_title=doc.title,
            doc_type=doc.doc_type,
            section_type="full_product",
            metadata=doc.metadata
        )
        
        return [chunk]
    
    def _chunk_semantic(self, doc: Document) -> List[Chunk]:
        """
        Semantic chunking: Split by paragraph/heading boundaries
        with context enrichment (Anthropic's technique)
        """
        chunks = []
        
        # Generate document summary for context
        doc_summary = self._generate_doc_summary(doc)
        
        # Split by double newlines (paragraphs) or headings
        sections = re.split(r'\n#{1,3}\s+|\n\n+', doc.content)
        
        current_chunk_text = []
        current_token_count = 0
        chunk_idx = 1
        
        for section in sections:
            section = section.strip()
            if not section:
                continue
            
            # Estimate tokens (rough: 1 token ≈ 4 characters)
            section_tokens = len(section) // 4
            
            if current_token_count + section_tokens > self.chunk_size and current_chunk_text:
                # Save current chunk
                chunk_text = " ".join(current_chunk_text)
                enriched_text = self._enrich_with_context(
                    doc.title, doc.doc_type.value, doc_summary, chunk_text
                )
                
                chunk = Chunk(
                    id=f"{doc.id}_chunk_{chunk_idx}",
                    text=enriched_text,
                    doc_id=doc.id,
                    doc_title=doc.title,
                    doc_type=doc.doc_type,
                    section_type="semantic",
                    metadata={
                        "chunk_index": chunk_idx,
                        "has_context_enrichment": True
                    }
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                if self.overlap > 0 and current_chunk_text:
                    overlap_text = " ".join(current_chunk_text[-2:])  # Last 2 sections
                    current_chunk_text = [overlap_text, section]
                    current_token_count = len(overlap_text) // 4 + section_tokens
                else:
                    current_chunk_text = [section]
                    current_token_count = section_tokens
                
                chunk_idx += 1
            else:
                current_chunk_text.append(section)
                current_token_count += section_tokens
        
        # Add final chunk
        if current_chunk_text:
            chunk_text = " ".join(current_chunk_text)
            enriched_text = self._enrich_with_context(
                doc.title, doc.doc_type.value, doc_summary, chunk_text
            )
            
            chunk = Chunk(
                id=f"{doc.id}_chunk_{chunk_idx}",
                text=enriched_text,
                doc_id=doc.id,
                doc_title=doc.title,
                doc_type=doc.doc_type,
                section_type="semantic",
                metadata={
                    "chunk_index": chunk_idx,
                    "has_context_enrichment": True
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _generate_doc_summary(self, doc: Document) -> str:
        """Generate a brief summary of the document for context"""
        # Simple heuristic: take first 200 characters + document type
        first_part = doc.content[:200].replace('\n', ' ').strip()
        return f"{doc.doc_type.value.title()} document about {first_part}..."
    
    def _enrich_with_context(self, title: str, doc_type: str, summary: str, chunk_text: str) -> str:
        """Add document-level context to chunk (Anthropic's technique)"""
        return f"""=== DOCUMENT CONTEXT ===
Title: {title}
Type: {doc_type}
Summary: {summary}

=== CHUNK CONTENT ===
{chunk_text}
"""
