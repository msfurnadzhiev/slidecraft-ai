"""Prompts for presentation structure generation."""

import json
from uuid import UUID

from src.schemas.presentation.presentation import PresentationStructure
from src.schemas.presentation.slide import SlideStructure


def build_structure_prompt(document_id: UUID, user_request: str) -> str:
    """Build the prompt for the presentation structure task."""
    presentation_structure_json = json.dumps(PresentationStructure.model_json_schema(), indent=2)
    slide_structure_json = json.dumps(SlideStructure.model_json_schema(), indent=2)

    return (
        "You are an expert presentation designer.\n\n"
        "Your task is to design a presentation outline based on the user's request.\n\n"
        "Do not include any information that is not present in the user's request.\n\n"
        f"User request:\n {user_request}\n\n"
        f"Document ID: {document_id}\n\n"
        "Return a PresentationStructure object strictly following this JSON schema:\n"
        f"{presentation_structure_json}\n\n"
        "The slides list must contain SlideStructure objects strictly following this JSON schema:\n"
        f"{slide_structure_json}\n\n"
        "Here is a brief explanation of the fields:\n"
        "PresentationStructure:\n"
        "- document_id: the ID of the document.\n"
        "- slides: list of slides.\n"
        "SlideStructure:\n"
        "- slide_number: position of the slide in the presentation.\n"
        "- slide_type: one of title, content, image, data, closing.\n"
        "- title: the slide title."
        "--> This should be a concise title for the slide.\n"
        "- description: brief explanation of slide content."
        "--> This should be a concise description of the slide content.\n"
        "Return only a valid PresentationStructure object."
        "Do not include explanations, extra fields, or wrapping objects."
    )
