"""
Document loader for Kerala Ayurveda corpus
"""
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from loguru import logger

from src.models import Document, DocumentType
from src.config import config


class DocumentLoader:
    """Load documents from Kerala Ayurveda corpus"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or config.data_dir
        logger.info(f"Initialized DocumentLoader with data_dir: {self.data_dir}")
    
    def load_all_documents(self) -> List[Document]:
        """Load all documents from the corpus"""
        documents = []
        
        # Load markdown files
        md_files = list(self.data_dir.glob("*.md"))
        for md_file in md_files:
            doc = self._load_markdown_file(md_file)
            if doc:
                documents.append(doc)
        
        # Load CSV catalog
        csv_files = list(self.data_dir.glob("*.csv"))
        for csv_file in csv_files:
            docs = self._load_csv_file(csv_file)
            documents.extend(docs)
        
        logger.info(f"Loaded {len(documents)} documents from corpus")
        return documents
    
    def _load_markdown_file(self, file_path: Path) -> Optional[Document]:
        """Load a markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Determine document type from filename
            doc_type = self._infer_document_type(file_path.stem)
            
            # Extract title (first heading or filename)
            title = self._extract_title(content, file_path.stem)
            
            doc = Document(
                id=file_path.stem,
                title=title,
                content=content,
                doc_type=doc_type,
                file_path=str(file_path),
                metadata={
                    "source": "markdown",
                    "filename": file_path.name
                }
            )
            
            logger.debug(f"Loaded document: {doc.id} ({doc.doc_type})")
            return doc
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def _load_csv_file(self, file_path: Path) -> List[Document]:
        """Load CSV file (product catalog)"""
        documents = []
        
        try:
            df = pd.read_csv(file_path)
            
            for idx, row in df.iterrows():
                # Create a document for each product
                content = self._format_product_row(row)
                
                doc = Document(
                    id=f"{file_path.stem}_{row.get('product_id', idx)}",
                    title=row.get('name', f"Product {idx}"),
                    content=content,
                    doc_type=DocumentType.CATALOG,
                    file_path=str(file_path),
                    metadata={
                        "source": "csv",
                        "row_index": idx,
                        "product_id": row.get('product_id'),
                        "category": row.get('category'),
                        **row.to_dict()
                    }
                )
                
                documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} products from {file_path.name}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading CSV {file_path}: {e}")
            return []
    
    def _infer_document_type(self, filename: str) -> DocumentType:
        """Infer document type from filename"""
        filename_lower = filename.lower()
        
        if "faq" in filename_lower:
            return DocumentType.FAQ
        elif "product_" in filename_lower:
            return DocumentType.PRODUCT
        elif "treatment_" in filename_lower:
            return DocumentType.TREATMENT
        elif "dosha" in filename_lower or "guide" in filename_lower:
            return DocumentType.GUIDE
        elif "foundation" in filename_lower or "content_style" in filename_lower:
            return DocumentType.FOUNDATION
        else:
            return DocumentType.FOUNDATION
    
    def _extract_title(self, content: str, fallback: str) -> str:
        """Extract title from content (first # heading)"""
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('# '):
                return line[2:].strip()
        
        return fallback.replace('_', ' ').title()
    
    def _format_product_row(self, row: pd.Series) -> str:
        """Format CSV row into readable text"""
        content_parts = []
        
        content_parts.append(f"# Product: {row.get('name', 'Unknown')}\n")
        content_parts.append(f"**Product ID:** {row.get('product_id', 'N/A')}\n")
        content_parts.append(f"**Category:** {row.get('category', 'N/A')}\n")
        content_parts.append(f"**Format:** {row.get('format', 'N/A')}\n\n")
        
        if pd.notna(row.get('target_concerns')):
            content_parts.append(f"**Target Concerns:** {row['target_concerns']}\n\n")
        
        if pd.notna(row.get('key_herbs')):
            content_parts.append(f"**Key Herbs:** {row['key_herbs']}\n\n")
        
        if pd.notna(row.get('contraindications_short')):
            content_parts.append(f"## Safety & Precautions\n")
            content_parts.append(f"**Contraindications:** {row['contraindications_short']}\n\n")
        
        if pd.notna(row.get('internal_tags')):
            content_parts.append(f"**Tags:** {row['internal_tags']}\n")
        
        return "".join(content_parts)


from typing import Optional
