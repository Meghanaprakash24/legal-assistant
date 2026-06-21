"""
src/orchestrator.py
---------------------
Production-grade LangGraph orchestrator for the Indian Legal RAG system.

Responsibilities
----------------
* Define the agent workflow as a LangGraph ``StateGraph``:
  Classifier → LegalRAGPipeline (retrieval + reranking) → Quote Selector
  → Section Mapper → Remedy Advisor → Citation Validator → Synthesizer.
* Initialise ``LegalRAGPipeline`` exactly once and reuse it across every
  workflow run.
* Compile the graph exactly once and reuse the compiled graph.
* Run each agent node defensively: a failing agent is logged and recorded
  in ``state["errors"]`` without crashing the workflow, so downstream
  nodes still run with whatever upstream output is available (or a safe
  empty default).

This module ONLY controls execution flow. It does NOT perform retrieval,
reranking, legal reasoning, LLM prompting, or section mapping itself —
those all belong to ``LegalRAGPipeline`` and the individual agents in
``agents/``, which this module calls but never reimplements.

Python 3.11+  |  PEP 8  |  Google-style docstrings
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Project-root path resolution
# Mirrors src/pipeline.py's approach so `import config` and sibling-module
# imports succeed regardless of invocation style:
#   python src/orchestrator.py
#   python -m src.orchestrator
#   import src.orchestrator  (from project root)
# ---------------------------------------------------------------------------

import sys
from pathlib import Path as _Path


def _ensure_project_root_on_path() -> None:
    """Add both the project root and src/ to sys.path.

    Guarantees regardless of invocation style:

    * ``import config``             works — config.py is at the project root
    * ``from pipeline import …``    works — pipeline.py is inside src/
    * ``from agents.x import …``    works — agents/ is at the project root

    Safe to call multiple times — paths are only inserted when absent.
    """
    src_dir = _Path(__file__).resolve().parent     # …/backend/src
    root_dir = src_dir.parent.parent               # …/LAW-RAG (project root, contains config.py)

    for path_str in (str(root_dir), str(src_dir)):
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_ensure_project_root_on_path()

import json
import time
import traceback
from typing import Any, Final, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

import config
from pipeline import LegalRAGPipeline

# Agent class imports — each class is instantiated once in __init__.
from agents.classifier import FactExtractionAgent
from agents.quote_selector import QuoteSelector
from agents.section_mapper import SectionMapper
from agents.remedy_advisor import RemedyAdvisor
from agents.validator import CitationValidator
from agents.synthesizer import ResponseSynthesizer

# ---------------------------------------------------------------------------
# Node name constants — used consistently for graph wiring and logging
# ---------------------------------------------------------------------------

NODE_CLASSIFY: Final[str] = "classify"
NODE_RETRIEVE: Final[str] = "retrieve"
NODE_QUOTE_SELECTOR: Final[str] = "quote_selector"
NODE_SECTION_MAPPER: Final[str] = "section_mapper"
NODE_REMEDY: Final[str] = "remedy"
NODE_VALIDATOR: Final[str] = "validator"
NODE_SYNTHESIZER: Final[str] = "synthesizer"


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


class OrchestratorState(TypedDict):
    """LangGraph state threaded through every node in the workflow.

    Every node receives the full state, updates only the keys it owns,
    and returns the full state — never a partial update — so behaviour
    stays consistent regardless of LangGraph's internal state-merging
    rules for a given node's return value.

    Attributes:
        query: The raw, original user incident description. Never
            mutated after the workflow starts.
        classification: Output of ``FactExtractionAgent.extract``.
        retrieval: Output of ``LegalRAGPipeline.search().to_dict()``.
        quotes: Output of ``QuoteSelector.select_quotes``.
        sections: Output of ``SectionMapper.map_sections``.
        remedy: Output of ``RemedyAdvisor.recommend``.
        validation: Output of ``CitationValidator.validate``.
        response: Output of ``ResponseSynthesizer.generate`` — the
            final result of the workflow.
        errors: Accumulated ``{"node": ..., "error": ...}`` entries for
            every node that failed. Empty when every node succeeded.
        node_timings: Per-node wall-clock execution time in seconds,
            keyed by node name, populated as the workflow progresses.
    """

    query: str
    classification: dict[str, Any]
    retrieval: dict[str, Any]
    quotes: list[dict[str, Any]]
    sections: list[dict[str, Any]]
    remedy: dict[str, Any]
    validation: dict[str, Any]
    response: dict[str, Any]
    errors: list[dict[str, str]]
    node_timings: dict[str, float]


def _initial_state(query: str) -> OrchestratorState:
    """Build a fresh, fully-populated initial state for one workflow run.

    Args:
        query: The raw user incident description.

    Returns:
        An ``OrchestratorState`` with every key present (empty defaults
        for everything not yet computed), so downstream nodes never hit
        a ``KeyError`` regardless of which earlier nodes failed.
    """
    return OrchestratorState(
        query=query,
        classification={},
        retrieval={},
        quotes=[],
        sections=[],
        remedy={},
        validation={},
        response={},
        errors=[],
        node_timings={},
    )


def _record_error(state: OrchestratorState, node: str, exc: Exception) -> None:
    """Append a structured error entry to ``state["errors"]`` in place.

    Args:
        state: The workflow state being updated.
        node: Name of the node where the failure occurred.
        exc: The caught exception.
    """
    logger.error("Node '{}' failed: {}", node, exc)
    logger.debug("Traceback for '{}':\n{}", node, traceback.format_exc())
    state["errors"].append({"node": node, "error": str(exc)})


def _time_node(state: OrchestratorState, node: str, start: float) -> None:
    """Record elapsed time for a node into ``state["node_timings"]``.

    Args:
        state: The workflow state being updated.
        node: Name of the node that just finished.
        start: The ``time.perf_counter()`` value captured at node start.
    """
    elapsed = time.perf_counter() - start
    state["node_timings"][node] = round(elapsed, 4)
    logger.info("Node '{}' completed in {:.3f}s.", node, elapsed)


# ---------------------------------------------------------------------------
# Main orchestrator class
# ---------------------------------------------------------------------------


class LegalRAGOrchestrator:
    """Coordinates the multi-agent legal workflow via a compiled LangGraph.

    The ``LegalRAGPipeline`` instance and the compiled ``StateGraph`` are
    each created exactly once, on first use, and reused for every
    subsequent :meth:`run` call. This class controls execution flow
    only — all retrieval, reranking, and legal-reasoning logic lives in
    ``LegalRAGPipeline`` and the ``agents.*`` modules it calls.

    Usage
    -----
    >>> orchestrator = LegalRAGOrchestrator()
    >>> final_state = orchestrator.run("Someone broke into my house and stole my phone.")
    >>> print(final_state["response"])
    """

    def __init__(self, pipeline: LegalRAGPipeline | None = None) -> None:
        """
        Initialise the orchestrator.

        Args:
            pipeline:
                Existing LegalRAGPipeline instance created by FastAPI.
                If None, a pipeline will be created lazily on first use,
                avoiding duplicate BM25 rebuilds, embedding-model reloads,
                CrossEncoder reloads, and Qdrant reconnects.
        """
        self._pipeline = pipeline
        self._graph = None

        # Instantiate every agent exactly once and reuse across all workflow runs.
        self.classifier = FactExtractionAgent()
        self.quote_selector = QuoteSelector()
        self.section_mapper = SectionMapper()
        self.remedy_advisor = RemedyAdvisor()
        self.validator = CitationValidator()
        self.synthesizer = ResponseSynthesizer()

        self._configure_logging()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @staticmethod
    def _configure_logging() -> None:
        """Configure Loguru with a structured, colourised console sink."""
        logger.remove()
        logger.add(
            sys.stderr,
            level="DEBUG",
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>"
            ),
            colorize=True,
        )
        config.LOG_DIR.mkdir(exist_ok=True)
        logger.add(
            config.LOG_DIR / "orchestrator.log",
            level="DEBUG",
            rotation=config.LOG_ROTATION,
            retention=config.LOG_RETENTION,
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Lazy initialisation — pipeline and graph each built exactly once
    # ------------------------------------------------------------------

    def _ensure_pipeline(self) -> LegalRAGPipeline:
        """Create and warm the shared ``LegalRAGPipeline`` exactly once.

        Returns:
            The cached ``LegalRAGPipeline`` instance.
        """
        if self._pipeline is None:
            logger.info("Initialising LegalRAGPipeline for orchestrator…")
            self._pipeline = LegalRAGPipeline()
            self._pipeline.initialize()
            logger.info("LegalRAGPipeline ready.")
        return self._pipeline

    def _ensure_graph(self) -> CompiledStateGraph:
        """Build and compile the LangGraph workflow exactly once.

        Returns:
            The cached, compiled ``StateGraph``.
        """
        if self._graph is None:
            logger.info("Building and compiling LangGraph workflow…")
            self._graph = self._build_graph()
            logger.info("LangGraph workflow compiled.")
        return self._graph

    def _build_graph(self) -> CompiledStateGraph:
        """Construct, wire, and compile the linear agent workflow graph.

        Workflow (strictly linear, no conditional branching):

            START
              → classify
              → retrieve            (LegalRAGPipeline.search)
              → quote_selector
              → section_mapper
              → remedy
              → validator
              → synthesizer
              → END

        Returns:
            A compiled ``StateGraph`` ready for ``.invoke()``.
        """
        graph = StateGraph(OrchestratorState)

        graph.add_node(NODE_CLASSIFY, self.classify_node)
        graph.add_node(NODE_RETRIEVE, self.retrieve_node)
        graph.add_node(NODE_QUOTE_SELECTOR, self.quote_selector_node)
        graph.add_node(NODE_SECTION_MAPPER, self.section_mapper_node)
        graph.add_node(NODE_REMEDY, self.remedy_node)
        graph.add_node(NODE_VALIDATOR, self.validator_node)
        graph.add_node(NODE_SYNTHESIZER, self.synthesizer_node)

        graph.add_edge(START, NODE_CLASSIFY)
        graph.add_edge(NODE_CLASSIFY, NODE_RETRIEVE)
        graph.add_edge(NODE_RETRIEVE, NODE_QUOTE_SELECTOR)
        graph.add_edge(NODE_QUOTE_SELECTOR, NODE_SECTION_MAPPER)
        graph.add_edge(NODE_SECTION_MAPPER, NODE_REMEDY)
        graph.add_edge(NODE_REMEDY, NODE_VALIDATOR)
        graph.add_edge(NODE_VALIDATOR, NODE_SYNTHESIZER)
        graph.add_edge(NODE_SYNTHESIZER, END)

        return graph.compile()

    # ------------------------------------------------------------------
    # Node functions
    # ------------------------------------------------------------------
    # Each node: receives the full state, updates only the keys it owns,
    # and returns the full state. Failures are caught, logged, and
    # recorded in state["errors"] -- the workflow always continues with
    # a safe empty default for the failed node's output, never raising
    # out of a node and aborting the graph.

    def classify_node(self, state: OrchestratorState) -> OrchestratorState:
        """Classify the user's incident description.

        Calls ``FactExtractionAgent.extract``. On failure, leaves
        ``state["classification"]`` as ``{}`` and records the error.

        Args:
            state: Current workflow state.

        Returns:
            The updated state.
        """
        logger.info("Running node: '{}'", NODE_CLASSIFY)
        start = time.perf_counter()
        try:
            state["classification"] = self.classifier.extract(state["query"])
        except Exception as exc:  # noqa: BLE001
            _record_error(state, NODE_CLASSIFY, exc)
            state["classification"] = {}
        _time_node(state, NODE_CLASSIFY, start)
        return state

    def retrieve_node(self, state: OrchestratorState) -> OrchestratorState:
        """Run hybrid retrieval + reranking via the shared pipeline.

        Calls ``LegalRAGPipeline.search()`` — never the retriever or
        reranker directly, since the pipeline already owns query
        analysis, metadata filtering, hybrid retrieval, and reranking.
        On failure, leaves ``state["retrieval"]`` as ``{}`` and records
        the error.

        Args:
            state: Current workflow state.

        Returns:
            The updated state.
        """
        logger.info("Running node: '{}'", NODE_RETRIEVE)
        start = time.perf_counter()
        try:
            pipeline = self._ensure_pipeline()
            output = pipeline.search(state["query"])
            state["retrieval"] = output.to_dict()
        except Exception as exc:  # noqa: BLE001
            _record_error(state, NODE_RETRIEVE, exc)
            state["retrieval"] = {}
        _time_node(state, NODE_RETRIEVE, start)
        return state

    def quote_selector_node(self, state: OrchestratorState) -> OrchestratorState:
        """Select supporting quotes from the retrieval results.

        Calls ``QuoteSelector.select_quotes``. On failure, leaves
        ``state["quotes"]`` as ``[]`` and records the error.

        Args:
            state: Current workflow state.

        Returns:
            The updated state.
        """
        logger.info("Running node: '{}'", NODE_QUOTE_SELECTOR)
        start = time.perf_counter()
        try:
            retrieval_results = state["retrieval"].get("results", [])
            state["quotes"] = self.quote_selector.select_quotes(
                retrieval_results=retrieval_results,
                user_query=state["query"],
            )
        except Exception as exc:  # noqa: BLE001
            _record_error(state, NODE_QUOTE_SELECTOR, exc)
            state["quotes"] = []
        _time_node(state, NODE_QUOTE_SELECTOR, start)
        return state

    def section_mapper_node(self, state: OrchestratorState) -> OrchestratorState:
        """Map selected quotes to applicable statutory sections/articles.

        Calls ``SectionMapper.map_sections``. On failure, leaves
        ``state["sections"]`` as ``[]`` and records the error.

        Args:
            state: Current workflow state.

        Returns:
            The updated state.
        """
        logger.info("Running node: '{}'", NODE_SECTION_MAPPER)
        start = time.perf_counter()
        try:
            result = self.section_mapper.map_sections(
                facts=state["classification"],
                selected_quotes=state["quotes"],
                retrieval_results=state["retrieval"].get("results", []),
            )
            state["sections"] = result
        except Exception as exc:  # noqa: BLE001
            _record_error(state, NODE_SECTION_MAPPER, exc)
            state["sections"] = []
        _time_node(state, NODE_SECTION_MAPPER, start)
        return state

    def remedy_node(self, state: OrchestratorState) -> OrchestratorState:
        """Advise on applicable legal remedies/procedures.

        Calls ``RemedyAdvisor.recommend``. On failure, leaves
        ``state["remedy"]`` as ``{}`` and records the error.

        Args:
            state: Current workflow state.

        Returns:
            The updated state.
        """
        logger.info("Running node: '{}'", NODE_REMEDY)
        start = time.perf_counter()
        try:
            # Build the input dict that RemedyAdvisor.recommend expects:
            # {"incident_type": list[str], "applicable_sections": list[dict]}.
            sections_output = state["sections"]
            applicable_sections = (
                sections_output.get("applicable_sections", [])
                if isinstance(sections_output, dict)
                else []
            )
            remedy_input: dict[str, Any] = {
                "incident_type": state["classification"].get("incident_type", []),
                "applicable_sections": applicable_sections,
            }
            state["remedy"] = self.remedy_advisor.recommend(remedy_input)
        except Exception as exc:  # noqa: BLE001
            _record_error(state, NODE_REMEDY, exc)
            state["remedy"] = {}
        _time_node(state, NODE_REMEDY, start)
        return state

    def validator_node(self, state: OrchestratorState) -> OrchestratorState:
        """Validate that cited quotes/sections are faithfully grounded.

        Calls ``CitationValidator.validate``. On failure, leaves
        ``state["validation"]`` as ``{}`` and records the error.

        Args:
            state: Current workflow state.

        Returns:
            The updated state.
        """
        logger.info("Running node: '{}'", NODE_VALIDATOR)
        start = time.perf_counter()
        try:
            retrieval_results = state["retrieval"].get("results", [])
            sections_output = state["sections"]
            applicable_sections = (
                sections_output.get("applicable_sections", [])
                if isinstance(sections_output, dict)
                else []
            )

            # Flatten per-chunk selected quotes into the tagged dict form
            # that CitationValidator.validate expects.
            selected_quotes: list[dict[str, str]] = []
            for chunk in state["quotes"]:
                if not isinstance(chunk, dict):
                    continue
                doc = chunk.get("document", "")
                section = chunk.get("section", "")
                chunk_id = chunk.get("chunk_id", "")
                for qt in chunk.get("selected_quotes", []):
                    if isinstance(qt, str) and qt.strip():
                        selected_quotes.append({
                            "quote": qt,
                            "document": doc,
                            "section": section,
                            "chunk_id": chunk_id,
                        })

            validator_input: dict[str, Any] = {
                "retrieval_results": retrieval_results,
                "selected_quotes": selected_quotes,
                "applicable_sections": applicable_sections,
            }
            state["validation"] = self.validator.validate(validator_input)
        except Exception as exc:  # noqa: BLE001
            _record_error(state, NODE_VALIDATOR, exc)
            state["validation"] = {}
        _time_node(state, NODE_VALIDATOR, start)
        return state

    def synthesizer_node(self, state: OrchestratorState) -> OrchestratorState:
        """Synthesize all upstream outputs into the final response.

        Calls ``ResponseSynthesizer.generate``. On failure, leaves
        ``state["response"]`` as ``{}`` and records the error — this is
        the last node, so a failure here means the workflow completes
        with no usable response, but the full state (including
        ``errors``) is still returned to the caller.

        Args:
            state: Current workflow state.

        Returns:
            The updated state.
        """
        logger.info("Running node: '{}'", NODE_SYNTHESIZER)
        start = time.perf_counter()
        try:
            sections_output = state["sections"]
            applicable_sections = (
                sections_output.get("applicable_sections", [])
                if isinstance(sections_output, dict)
                else []
            )

            # Flatten per-chunk selected quotes into tagged dict form.
            selected_quotes: list[dict[str, str]] = [
                {
                    "quote": qt,
                    "document": chunk.get("document", ""),
                    "section": chunk.get("section", ""),
                    "chunk_id": chunk.get("chunk_id", ""),
                }
                for chunk in state["quotes"]
                if isinstance(chunk, dict)
                for qt in chunk.get("selected_quotes", [])
                if isinstance(qt, str) and qt.strip()
            ]

            # Assemble the pipeline_output dict expected by
            # ResponseSynthesizer.generate.
            synthesizer_input: dict[str, Any] = {
                "query": state["query"],
                "classification": state["classification"],
                "applicable_sections": applicable_sections,
                "remedy": state["remedy"],
                "validation": state["validation"],
                "retrieval_results": state["retrieval"].get("results", []),
                "selected_quotes": selected_quotes,
            }
            state["response"] = self.synthesizer.generate(synthesizer_input)
        except Exception as exc:  # noqa: BLE001
            _record_error(state, NODE_SYNTHESIZER, exc)
            state["response"] = {}
        _time_node(state, NODE_SYNTHESIZER, start)
        return state

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, query: str) -> OrchestratorState:
        """Run the complete agent workflow for a single user query.

        Lazily initialises the shared ``LegalRAGPipeline`` and compiles
        the LangGraph workflow on first call; both are reused on every
        subsequent call. Designed to be called directly from a FastAPI
        request handler — ``orchestrator = LegalRAGOrchestrator()`` once
        at application start-up, then ``orchestrator.run(query)`` per
        request.

        Args:
            query: The raw user incident description. Must be non-empty.

        Returns:
            The final ``OrchestratorState`` after every node has run
            (or failed and been recorded in ``state["errors"]``).

        Raises:
            ValueError: If ``query`` is empty or whitespace-only.
        """
        if not query or not query.strip():
            raise ValueError("query must not be empty.")

        logger.info("Workflow started — query={!r}", query)
        t_start = time.perf_counter()

        graph = self._ensure_graph()
        state = _initial_state(query)

        final_state = graph.invoke(state)

        total_time = time.perf_counter() - t_start
        final_state["node_timings"]["total"] = round(total_time, 4)

        if final_state["errors"]:
            logger.warning(
                "Workflow completed with {} error(s) in {:.3f}s.",
                len(final_state["errors"]),
                total_time,
            )
        else:
            logger.info("Workflow completed successfully in {:.3f}s.", total_time)

        return final_state


# ---------------------------------------------------------------------------
# CLI / test entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the orchestrator from the command line with a sample query.

    Example
    -------
    .. code-block:: bash

        python src/orchestrator.py
        python src/orchestrator.py "Someone broke into my house and stole my phone."
        python -m src.orchestrator "I was assaulted outside my office."
    """
    query = " ".join(sys.argv[1:]) or (
        "Someone broke into my house and stole my phone."
    )

    orchestrator = LegalRAGOrchestrator()
    final_state = orchestrator.run(query)
    print(json.dumps(final_state, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()