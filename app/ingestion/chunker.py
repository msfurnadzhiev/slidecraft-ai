"""Module for chunking text into smaller pieces with token counting."""

import uuid
from typing import List
import tiktoken

from app.schemas.chunk import ChunkCreate
from app.schemas.document import DocumentContent
from app.utils.singleton import SingletonMeta

class TextChunker(metaclass=SingletonMeta):
    """Chunks text into smaller pieces based on token count."""
    
    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 50):
        """Initialize the chunker.
        
        Args:
            max_tokens: Maximum number of tokens per chunk
            overlap_tokens: Number of tokens to overlap between chunks
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text."""
        return len(self.encoding.encode(text))
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks based on token count."""
        tokens = self.encoding.encode(text)
        chunks = []
        
        start_idx = 0
        while start_idx < len(tokens):
            end_idx = start_idx + self.max_tokens
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text)
            
            if end_idx >= len(tokens):
                break
            
            start_idx = end_idx - self.overlap_tokens
        
        return chunks
    
    def chunk_document(self, document: DocumentContent) -> List[ChunkCreate]:
        """Chunk a document into smaller pieces. Records character offsets so text can be re-extracted from the PDF."""
        all_chunks = []
        global_chunk_index = 0

        for page in document.pages:
            if not page.text.strip():
                continue

            page_text = page.text
            tokens = self.encoding.encode(page_text)
            start_idx = 0

            while start_idx < len(tokens):
                end_idx = min(start_idx + self.max_tokens, len(tokens))
                chunk_tokens = tokens[start_idx:end_idx]
                chunk_text = self.encoding.decode(chunk_tokens)
                token_count = len(chunk_tokens)

                start_char = len(self.encoding.decode(tokens[:start_idx]))
                end_char = len(self.encoding.decode(tokens[:end_idx]))

                chunk_id = str(uuid.uuid4())
                chunk = ChunkCreate(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    page_number=page.page_number,
                    chunk_index=global_chunk_index,
                    token_count=token_count,
                    start_char_offset=start_char,
                    end_char_offset=end_char,
                    text=chunk_text,
                )
                all_chunks.append(chunk)
                global_chunk_index += 1

                if end_idx >= len(tokens):
                    break
                start_idx = end_idx - self.overlap_tokens

        return all_chunks
