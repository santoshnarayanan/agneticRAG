# This file hols all our graph state

from typing import List, TypedDict


class GraphState(TypedDict):
    """Reprsents the state of the graph

    Args:
        question:question
        generation: LLM generation
        web_search: wether to add search
        documents: list of documents
    """
    question: str
    generation: str
    web_search: bool
    documents: List[str]
    