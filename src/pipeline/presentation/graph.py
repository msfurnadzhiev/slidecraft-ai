"""Assemble and compile the LangGraph StateGraph for the presentation workflow."""

from typing import Callable, Dict

from langgraph.graph import END, START, StateGraph

from src.pipeline.presentation.nodes import WorkflowNodes
from src.pipeline.presentation.routes import route_accept, route_content, route_validation
from src.pipeline.presentation.state import WorkflowState


def compile_graph(nodes: WorkflowNodes):
    """Wire nodes and edges, then return the compiled runnable."""
    
    graph = StateGraph(WorkflowState)

    _register_nodes(graph, nodes)
    _add_linear_edges(graph)
    _add_conditional_edges(graph)

    return graph.compile()

def _register_nodes(graph: StateGraph, nodes: WorkflowNodes) -> None:
    """Register all workflow nodes."""
    node_map: Dict[str, Callable] = {
        "generate_structure": nodes.generate_structure,
        "generate_slide_content": nodes.generate_slide_content,
        "build_slide": nodes.build_slide,
        "validate_slide": nodes.validate_slide,
        "accept_slide": nodes.accept_slide,
        "render_presentation": nodes.render_presentation,
    }

    for name, handler in node_map.items():
        graph.add_node(name, handler)


def _add_linear_edges(graph: StateGraph) -> None:
    """Add simple linear edges."""
    graph.add_edge(START, "generate_structure")
    graph.add_edge("generate_structure", "generate_slide_content")
    graph.add_edge("build_slide", "validate_slide")
    graph.add_edge("render_presentation", END)


def _add_conditional_edges(graph: StateGraph) -> None:
    """Add all conditional routing logic."""
    
    graph.add_conditional_edges(
        "generate_slide_content",
        route_content,
        {
            "more_content": "generate_slide_content",
            "content_done": "build_slide",
        },
    )

    graph.add_conditional_edges(
        "validate_slide",
        route_validation,
        {
            "retry": "build_slide",
            "accepted": "accept_slide",
        },
    )

    graph.add_conditional_edges(
        "accept_slide",
        route_accept,
        {
            "more_slides": "build_slide",
            "all_done": "render_presentation",
        },
    )
