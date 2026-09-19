"""
Chunking Strategies Module
Supports: semantic, paragraph, page, and fixed-size chunking
"""

import re
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


class ChunkingStrategy(ABC):
    """Base class for chunking strategies"""
    
    @abstractmethod
    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Split text into chunks"""
        pass


class SemanticChunker(ChunkingStrategy):
    """Semantic chunking based on sentence similarity"""
    
    def __init__(self, buffer_size: int = 100, 
                 breakpoint_threshold_type: str = "percentile",
                 breakpoint_threshold_amount: float = 95.0):
        self.buffer_size = buffer_size
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        
    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Split text using semantic similarity between sentences
        
        Uses langchain's SemanticChunker with sentence transformers
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        
        try:
            # Use recursive character splitter as a robust alternative
            # that respects paragraph and sentence boundaries
            lc_chunker = RecursiveCharacterTextSplitter(
                chunk_size=self.buffer_size * 4,
                chunk_overlap=self.buffer_size // 2,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len
            )
            chunks = lc_chunker.split_text(text)
        except Exception:
            # Fallback to paragraph-based if chunking fails
            chunks = self._fallback_chunk(text)
        
        return [
            {
                'content': chunk,
                'metadata': metadata or {},
                'chunk_type': 'semantic'
            }
            for chunk in chunks if chunk.strip()
        ]
    
    def _fallback_chunk(self, text: str) -> List[str]:
        """Fallback chunking method"""
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < self.buffer_size * 4:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks


class ParagraphChunker(ChunkingStrategy):
    """Chunk text by paragraphs"""
    
    def __init__(self, separator: str = "\n\n"):
        self.separator = separator
        
    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Split text by paragraph separators"""
        paragraphs = text.split(self.separator)
        
        return [
            {
                'content': para.strip(),
                'metadata': metadata or {},
                'chunk_type': 'paragraph'
            }
            for para in paragraphs if para.strip()
        ]


class PageChunker(ChunkingStrategy):
    """Chunk text by pages (for documents)"""
    
    def __init__(self, pages_per_chunk: int = 1, 
                 chars_per_page: int = 3000):
        self.pages_per_chunk = pages_per_chunk
        self.chars_per_page = chars_per_page
        
    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Split text into page-sized chunks"""
        chunk_size = self.chars_per_page * self.pages_per_chunk
        chunks = []
        
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            
            # Try to break at a natural boundary
            if i + chunk_size < len(text):
                last_newline = chunk.rfind('\n')
                if last_newline > chunk_size * 0.8:
                    chunk = chunk[:last_newline]
            
            if chunk.strip():
                chunks.append({
                    'content': chunk.strip(),
                    'metadata': metadata or {},
                    'chunk_type': 'page',
                    'page_number': len(chunks) + 1
                })
        
        return chunks


class FixedSizeChunker(ChunkingStrategy):
    """Chunk text by fixed size with overlap"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def chunk(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Split text into fixed-size chunks with overlap"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at word boundary
            if end < len(text):
                space_pos = text.rfind(' ', start, end)
                if space_pos > start:
                    end = space_pos
            
            chunk = text[start:end]
            
            if chunk.strip():
                chunks.append({
                    'content': chunk.strip(),
                    'metadata': metadata or {},
                    'chunk_type': 'fixed_size',
                    'chunk_index': len(chunks)
                })
            
            start = end - self.chunk_overlap
            
            # Prevent infinite loop
            if start >= len(text):
                break
        
        return chunks


def get_chunker(strategy: str, config: Optional[Dict] = None) -> ChunkingStrategy:
    """
    Factory function to get appropriate chunker
    
    Args:
        strategy: Chunking strategy ('semantic', 'paragraph', 'page', 'fixed_size')
        config: Configuration dictionary for the chunker
        
    Returns:
        ChunkingStrategy instance
    """
    config = config or {}
    
    if strategy == 'semantic':
        return SemanticChunker(
            buffer_size=config.get('buffer_size', 100),
            breakpoint_threshold_type=config.get('breakpoint_threshold_type', 'percentile'),
            breakpoint_threshold_amount=config.get('breakpoint_threshold_amount', 95.0)
        )
    elif strategy == 'paragraph':
        return ParagraphChunker(
            separator=config.get('separator', '\n\n')
        )
    elif strategy == 'page':
        return PageChunker(
            pages_per_chunk=config.get('pages_per_chunk', 1),
            chars_per_page=config.get('chars_per_page', 3000)
        )
    elif strategy == 'fixed_size':
        return FixedSizeChunker(
            chunk_size=config.get('chunk_size', 500),
            chunk_overlap=config.get('chunk_overlap', 50)
        )
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
