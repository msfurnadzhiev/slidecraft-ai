"""Two-model embedding strategy: sentence-transformers for text, CLIP for images.

Text chunks are embedded with all-MiniLM-L12-v2 (384-dim) for high-quality
text-to-text retrieval.  Images are embedded with CLIP ViT-B/32 (512-dim) so
that a CLIP text query can find visually relevant images.  The two vector
spaces are independent — each model has its own DB table and index.
"""

import io
from typing import Dict, List

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from app.schemas.chunk import ChunkCreate
from app.schemas.embedding import ChunkEmbeddingCreate, ImageEmbeddingCreate
from app.schemas.image import ImageCreate
from app.utils.singleton import SingletonMeta

_DEFAULT_TEXT_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
_DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


class TextEmbedder(metaclass=SingletonMeta):
    """Sentence-transformer embeddings for text chunks (384-dim)."""

    def __init__(self, model_name: str = _DEFAULT_TEXT_MODEL):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_dim: int = self.model.get_sentence_embedding_dimension()

    def generate_embedding(self, text: str) -> List[float]:
        """Encode a single text string."""
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode a batch of texts."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [emb.tolist() for emb in embeddings]

    def generate_chunk_embeddings(
        self, document_id: str, chunks: List[ChunkCreate],
    ) -> List[ChunkEmbeddingCreate]:
        """Generate sentence-transformer embeddings for a list of chunks."""
        texts = [chunk.text for chunk in chunks]
        vectors = self.generate_embeddings_batch(texts)
        return [
            ChunkEmbeddingCreate(
                document_id=document_id,
                chunk_id=chunk.chunk_id,
                vector=vector,
                page_number=chunk.page_number,
            )
            for chunk, vector in zip(chunks, vectors)
        ]


class ImageEmbedder(metaclass=SingletonMeta):
    """CLIP embeddings for images (512-dim) and CLIP text queries."""

    def __init__(self, model_name: str = _DEFAULT_CLIP_MODEL):
        self.model_name = model_name
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        self.embedding_dim: int = self.model.config.projection_dim  # 512

    @torch.no_grad()
    def generate_text_query_embedding(self, text: str) -> List[float]:
        """Encode a text query via CLIP's text encoder for cross-modal search."""
        inputs = self.processor(
            text=[text], return_tensors="pt", padding=True, truncation=True,
        )
        features = self.model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features[0].tolist()

    @torch.no_grad()
    def generate_image_embeddings_batch(
        self, images_bytes: List[bytes],
    ) -> List[List[float]]:
        """Encode a batch of images (raw bytes each) into CLIP image vectors."""
        pil_images = [
            Image.open(io.BytesIO(b)).convert("RGB") for b in images_bytes
        ]
        inputs = self.processor(images=pil_images, return_tensors="pt")
        features = self.model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.tolist()

    def generate_image_embeddings(
        self,
        images: List[ImageCreate],
        image_bytes_map: Dict[str, bytes],
    ) -> List[ImageEmbeddingCreate]:
        """Generate CLIP image embeddings for extracted images."""
        ordered_bytes = [image_bytes_map[img.image_id] for img in images]
        vectors = self.generate_image_embeddings_batch(ordered_bytes)
        return [
            ImageEmbeddingCreate(
                document_id=img.document_id,
                image_id=img.image_id,
                vector=vector,
                page_number=img.page_number,
            )
            for img, vector in zip(images, vectors)
        ]
