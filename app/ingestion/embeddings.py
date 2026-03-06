"""Module for generating embeddings from text chunks and image placeholders."""

from typing import List
from sentence_transformers import SentenceTransformer

from app.schemas.chunk import ChunkCreate
from app.schemas.embedding import EmbeddingCreate
from app.schemas.image import ImageCreate
from app.utils.singleton import SingletonMeta

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"

class EmbeddingGenerator(metaclass=SingletonMeta):
    """Generates embeddings for text chunks using sentence transformers."""
    
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        """Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate an embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]
    
    def generate_chunk_embeddings(
        self, document_id: str, chunks: List[ChunkCreate]
    ) -> List[EmbeddingCreate]:
        """Generate embeddings for a list of chunks (unified EmbeddingCreate)."""
        texts = [chunk.text for chunk in chunks]
        vectors = self.generate_embeddings_batch(texts)
        return [
            EmbeddingCreate(
                document_id=document_id,
                object_id=chunk.chunk_id,
                object_type="chunk",
                vector=vector,
                page_number=chunk.page_number,
            )
            for chunk, vector in zip(chunks, vectors)
        ]

    def generate_image_embeddings(
        self, images: List[ImageCreate]
    ) -> List[EmbeddingCreate]:
        """Build EmbeddingCreate list for images (placeholder vectors; no model encode)."""
        placeholder = [0.0] * self.embedding_dim
        return [
            EmbeddingCreate(
                document_id=img.document_id,
                object_id=img.image_id,
                object_type="image",
                vector=placeholder,
                page_number=img.page_number,
            )
            for img in images
        ]
