"""
Two-model embedding strategy: sentence-transformers for text, CLIP for images.

- Text chunks are embedded with all-MiniLM-L12-v2 (384-dim) for high-quality
  text-to-text retrieval.
- Images are embedded with CLIP ViT-B/32 (512-dim) so that a CLIP text query
  can find visually relevant images.

The two vector spaces are independent — each model has its own database table
and index.
"""

import io
from typing import Dict, List

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from app.schemas.chunk import ChunkCreate
from app.schemas.image import ImageCreate
from app.utils.singleton import SingletonMeta


# Default models for text and image embeddings
_DEFAULT_TEXT_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
_DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"

# Maps image_id (string) to raw image bytes
ImageBytesMap = Dict[str, bytes]

# List of floats representing the embedding
EmbeddingVector: type = List[float]

class TextEmbedder(metaclass=SingletonMeta):
    """Sentence-transformer embeddings for text chunks (384-dim)."""

    def __init__(self, model_name: str = _DEFAULT_TEXT_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim: int = self.model.get_sentence_embedding_dimension()

    def generate_embedding(self, text: str) -> EmbeddingVector:
        """Encode a single text string into a vector."""
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def generate_embeddings_batch(self, texts: List[str]) -> List[EmbeddingVector]:
        """Encode a batch of texts into vectors."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]

    def embed_chunks(self, chunks: List[ChunkCreate]) -> List[ChunkCreate]:
        """
        Generate embeddings for text chunks and update them in-place.

        Args:
            chunks: List of ChunkCreate objects containing text.

        Returns:
            The same list of ChunkCreate objects with 'vector' updated.
        """
        texts = [chunk.text for chunk in chunks]
        vectors = self.generate_embeddings_batch(texts)
        for chunk, vector in zip(chunks, vectors):
            chunk.vector = vector
        return chunks


class ImageEmbedder(metaclass=SingletonMeta):
    """CLIP embeddings for images (512-dim) and CLIP text queries."""

    def __init__(self, model_name: str = _DEFAULT_CLIP_MODEL):
        self.model_name = model_name
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        self.embedding_dim: int = self.model.config.projection_dim  # 512

    @torch.no_grad()
    def generate_text_query_embedding(self, text: str) -> EmbeddingVector:
        """Encode a text query via CLIP's text encoder for cross-modal search."""
        inputs = self.processor(
            text=[text], return_tensors="pt", padding=True, truncation=True
        )
        features = self.model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features[0].tolist()

    @torch.no_grad()
    def generate_image_embeddings_batch(self, images_bytes: List[bytes]) -> List[EmbeddingVector]:
        """Encode a batch of images (raw bytes) into CLIP image vectors."""
        pil_images = [Image.open(io.BytesIO(b)).convert("RGB") for b in images_bytes]
        inputs = self.processor(images=pil_images, return_tensors="pt")
        features = self.model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.tolist()

    def embed_images(
        self,
        images: List[ImageCreate],
        image_bytes_map: ImageBytesMap,
    ) -> List[ImageCreate]:
        """
        Generate CLIP image embeddings and update images in-place.

        Args:
            images: List of ImageCreate objects to embed.
            image_bytes_map: Dict mapping image_id -> raw image bytes.

        Returns:
            The same list of ImageCreate objects with 'vector' updated.
        """
        ordered_bytes = [image_bytes_map[img.image_id] for img in images]
        vectors = self.generate_image_embeddings_batch(ordered_bytes)
        for img, vector in zip(images, vectors):
            img.vector = vector
        return images