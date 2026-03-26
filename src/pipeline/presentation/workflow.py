"""End-to-end presentation generation workflow — LangGraph orchestration façade."""

import logging
from typing import List, Optional
from uuid import UUID

from src.agents.content_generator_agent import ContentGeneratorAgent
from src.agents.presentation_structure_agent import PresentationStructureAgent
from src.agents.quality_validator_agent import QualityValidatorAgent
from src.agents.slide_builder_agent import SlideBuilderAgent
from src.pipeline.presentation.graph import compile_graph
from src.pipeline.presentation.nodes import WorkflowNodes
from src.pipeline.presentation.state import WorkflowState, make_initial_state
from src.schemas.presentation.presentation import PresentationWorkflowResponse
from src.schemas.template import SlideLayoutResponse
from src.services.presentation.builder import PresentationBuilderService
from src.utils.profiling import trace_runtime

log = logging.getLogger(__name__)


class PresentationWorkflow:
    """Orchestrates the full agent pipeline as a compiled LangGraph graph."""

    def __init__(
        self,
        structure_agent: PresentationStructureAgent,
        content_agent: ContentGeneratorAgent,
        builder_agent: SlideBuilderAgent,
        validator_agent: QualityValidatorAgent,
        builder_service: PresentationBuilderService,
    ) -> None:
        """Initialize the PresentationWorkflow with the required agents and services."""
        self._nodes = WorkflowNodes(
            structure_agent=structure_agent,
            content_agent=content_agent,
            builder_agent=builder_agent,
            validator_agent=validator_agent,
            builder_service=builder_service,
        )
        self._graph = compile_graph(self._nodes)

    @trace_runtime
    def run(
        self,
        document_id: UUID,
        user_request: str,
        template_id: UUID,
        template_file_path: str,
        template_layouts: List[SlideLayoutResponse],
        presentation_name: Optional[str] = None,
    ) -> PresentationWorkflowResponse:
        """Execute the compiled graph and return a workflow response."""
        log.info(
            "Graph workflow started — document_id=%s, template_id=%s, request='%s'",
            document_id,
            template_id,
            user_request[:120],
        )

        initial_state: WorkflowState = make_initial_state(
            document_id=document_id,
            user_request=user_request,
            template_id=template_id,
            template_file_path=template_file_path,
            template_layouts=template_layouts,
            presentation_name=presentation_name,
        )

        final_state: WorkflowState = self._graph.invoke(initial_state)

        log.info(
            "Graph workflow complete — %d slides, %d revision(s) | path='%s'",
            len(final_state["slide_contents"]),
            final_state["total_revisions"],
            final_state["storage_path"],
        )

        return PresentationWorkflowResponse(
            storage_path=final_state["storage_path"],
            total_slides=len(final_state["slide_contents"]),
            quality_revisions=final_state["total_revisions"],
        )
