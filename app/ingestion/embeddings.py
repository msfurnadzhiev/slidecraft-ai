"""Unified CLIP-based embeddings for text chunks and images.

Uses openai/clip-vit-base-patch32 so that text and image vectors live in
the same 512-dim space — chunks and images with similar content will have
close cosine similarity.
"""

import io
from typing import Dict, List

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from app.schemas.chunk import ChunkCreate
from app.schemas.embedding import EmbeddingCreate
from app.schemas.image import ImageCreate
from app.utils.singleton import SingletonMeta

DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"


class EmbeddingGenerator(metaclass=SingletonMeta):
    """Generates L2-normalised CLIP embeddings for text and images."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        self.embedding_dim: int = self.model.config.projection_dim  # 512

    @torch.no_grad()
    def generate_embedding(self, text: str) -> List[float]:
        """Encode a single text string into a CLIP text vector."""
        inputs = self.processor(
            text=[text], return_tensors="pt", padding=True, truncation=True,
        )
        features = self.model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features[0].tolist()

    @torch.no_grad()
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode a batch of texts into CLIP text vectors."""
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True,
        )
        features = self.model.get_text_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.tolist()

    @torch.no_grad()
    def generate_image_embedding(self, image_bytes: bytes) -> List[float]:
        """Encode a single image (raw bytes) into a CLIP image vector."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        features = self.model.get_image_features(**inputs)
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

    def generate_chunk_embeddings(
        self, document_id: str, chunks: List[ChunkCreate],
    ) -> List[EmbeddingCreate]:
        """Generate CLIP text embeddings for a list of chunks."""
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
        self,
        images: List[ImageCreate],
        image_bytes_map: Dict[str, bytes],
    ) -> List[EmbeddingCreate]:
        """Generate CLIP image embeddings for extracted images.

        ``image_bytes_map`` maps each ``image_id`` to its raw file bytes so
        the CLIP image encoder can produce a real vector (not a placeholder).
        """
        ordered_bytes = [image_bytes_map[img.image_id] for img in images]
        vectors = self.generate_image_embeddings_batch(ordered_bytes)
        return [
            EmbeddingCreate(
                document_id=img.document_id,
                object_id=img.image_id,
                object_type="image",
                vector=vector,
                page_number=img.page_number,
            )
            for img, vector in zip(images, vectors)
        ]
