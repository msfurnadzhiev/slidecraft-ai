"""LangGraph node callables for the presentation workflow."""

import logging
from typing import Any, Dict

from src.agents.content_generator_agent import ContentGeneratorAgent
from src.agents.presentation_structure_agent import PresentationStructureAgent
from src.agents.quality_validator_agent import QualityValidatorAgent
from src.agents.slide_builder_agent import SlideBuilderAgent
from src.pipeline.presentation.state import MAX_QUALITY_ITERATIONS, WorkflowState
from src.services.presentation.builder import PresentationBuilderService

log = logging.getLogger(__name__)

class WorkflowNodes:
    """Holds injected agents/services and implements each graph node."""

    def __init__(
        self,
        structure_agent: PresentationStructureAgent,
        content_agent: ContentGeneratorAgent,
        builder_agent: SlideBuilderAgent,
        validator_agent: QualityValidatorAgent,
        builder_service: PresentationBuilderService,
    ) -> None:
        self.structure_agent = structure_agent
        self.content_agent = content_agent
        self.builder_agent = builder_agent
        self.validator_agent = validator_agent
        self.builder_service = builder_service


    def _current_slide_content(self, state: WorkflowState):
        return state["slide_contents"][state["build_slide_index"]]

    def _current_structure(self, state: WorkflowState):
        return state["structure"].slides[state["content_slide_index"]]


    def generate_structure(self, state: WorkflowState) -> Dict[str, Any]:
        """Generate presentation structure."""
        log.info("generate_structure — document_id=%s", state["document_id"])

        structure = self.structure_agent.suggest_structure(
            state["document_id"], state["user_request"]
        )

        log.info("generate_structure done — %d slides", len(structure.slides))
        return {"structure": structure}

    def generate_slide_content(self, state: WorkflowState) -> Dict[str, Any]:
        """Generate content for the current slide."""
        idx = state["content_slide_index"]
        structure = state["structure"]
        slide_structure = structure.slides[idx]

        log.info(
            "generate_slide_content %d/%d — '%s' [%s]",
            idx + 1,
            len(structure.slides),
            slide_structure.title,
            slide_structure.slide_type,
        )

        content = self.content_agent.generate_structure(
            state["document_id"], slide_structure
        )

        return {
            "slide_contents": [content],
            "content_slide_index": idx + 1,
        }

    def build_slide(self, state: WorkflowState) -> Dict[str, Any]:
        """Assign layout + placeholders for current slide."""
        slide_content = self._current_slide_content(state)
        attempt = state["quality_attempts"] + 1

        log.info(
            "build_slide — slide #%d '%s' (attempt %d/%d)",
            slide_content.slide_number,
            slide_content.title,
            attempt,
            MAX_QUALITY_ITERATIONS,
        )

        assignment = self.builder_agent.assign(
            slide=slide_content,
            layouts=state["template_layouts"],
            used_layout_indices=set(state["used_layout_indices"]),
            revision_feedback=state["revision_feedback"],
        )

        return {"current_assignment": assignment}

    def validate_slide(self, state: WorkflowState) -> Dict[str, Any]:
        """Validate slide quality and update retry state."""
        slide_content = self._current_slide_content(state)

        validation = self.validator_agent.validate(
            slide_content=slide_content,
            slide_assignment=state["current_assignment"],
        )

        return self._handle_validation_result(state, slide_content, validation)

    def accept_slide(self, state: WorkflowState) -> Dict[str, Any]:
        """Commit assignment and reset quality state."""
        assignment = state["current_assignment"]
        slide_content = self._current_slide_content(state)

        log.info(
            "accept_slide — slide #%d committed (layout_index=%d)",
            slide_content.slide_number,
            assignment.layout_index,
        )

        return {
            "assignments": [assignment],
            "used_layout_indices": [assignment.layout_index],
            "build_slide_index": state["build_slide_index"] + 1,
            "quality_attempts": 0,
            "revision_feedback": None,
        }

    def render_presentation(self, state: WorkflowState) -> Dict[str, Any]:
        """Render final PPTX file."""
        log.info(
            "render_presentation — %d slide(s) (document_id=%s)",
            len(state["slide_contents"]),
            state["document_id"],
        )

        prs = self._initialize_presentation(state)

        for slide_content, assignment in zip(
            state["slide_contents"], state["assignments"]
        ):
            self._render_single_slide(prs, slide_content, assignment)

        storage_path = self.builder_service.save_presentation(
            prs, state["presentation_name"]
        )

        log.info("render_presentation done — saved to '%s'", storage_path)
        return {"storage_path": storage_path}

    def _handle_validation_result(
        self, state: WorkflowState, slide_content, validation
    ) -> Dict[str, Any]:
        """Handle validation success/failure."""
        updates: Dict[str, Any] = {"last_validation": validation}

        if not validation.passed:
            attempts = state["quality_attempts"] + 1

            log.info(
                "validate_slide — slide #%d FAILED (attempt %d/%d, score=%.2f) | %s",
                slide_content.slide_number,
                attempts,
                MAX_QUALITY_ITERATIONS,
                validation.score,
                "; ".join(validation.issues) or validation.feedback[:120],
            )

            updates.update(
                {
                    "quality_attempts": attempts,
                    "revision_feedback": validation.feedback,
                    "total_revisions": state["total_revisions"] + 1,
                }
            )
        else:
            log.info(
                "validate_slide — slide #%d PASSED (score=%.2f)",
                slide_content.slide_number,
                validation.score,
            )

        return updates

    def _initialize_presentation(self, state: WorkflowState):
        """Create presentation object from template."""
        template_abs = self.builder_service.template_storage.get_absolute_path(
            state["template_file_path"]
        )
        return self.builder_service.create_blank_presentation(template_abs)

    def _render_single_slide(self, prs, slide_content, assignment) -> None:
        """Render a single slide."""
        slide = self.builder_service.add_slide(prs, assignment.layout_index)

        self.builder_service.set_title(slide, slide_content.title)
        self._fill_placeholders(slide, slide_content, assignment)
        self._add_images(slide, slide_content)

    def _fill_placeholders(self, slide, slide_content, assignment) -> None:
        """Fill slide placeholders."""
        for fill in assignment.placeholder_fills:
            if fill.placeholder_idx == 0:
                log.warning(
                    "Slide #%d: skipping title placeholder (idx=0)",
                    slide_content.slide_number,
                )
                continue

            if fill.text:
                self.builder_service.fill_placeholder(
                    slide, fill.placeholder_idx, fill.text
                )

    def _add_images(self, slide, slide_content) -> None:
        """Attach images to slide."""
        for img in slide_content.images or []:
            if self.builder_service.add_image_from_storage_path(
                slide, img.image_url
            ):
                break