"""
src/reranker.py
---------------
Production-grade Cross-Encoder reranker for the Indian Legal RAG system.

Responsibilities
----------------
* Accept the ranked candidate list produced by ``src/retriever.py``.
* Validate every candidate before scoring.
* Run batch inference with ``BAAI/bge-reranker-base`` via
  ``sentence_transformers.CrossEncoder``.
* Return the top-K results sorted by rerank score descending, with the
  original retrieval score preserved for auditability.

Does NOT retrieve documents, embed queries, call Qdrant, call BM25,
call any LLM, or perform legal reasoning. This module ONLY reranks.

Expected ``config.py`` attributes
----------------------------------
    RERANKER_MODEL_NAME : str   e.g. "BAAI/bge-reranker-base"
    RERANKER_TOP_K      : int   final results returned, e.g. 5
    RERANKER_BATCH_SIZE : int   inference batch size, e.g. 32
    RERANKER_DEVICE     : str   "auto" | "cuda" | "mps" | "cpu"
    LOG_DIR             : Path
    LOG_ROTATION        : str
    LOG_RETENTION       : str

Python 3.11+  |  PEP 8  |  Google-style docstrings
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Project-root path resolution
# ---------------------------------------------------------------------------
# When this module is executed as ``python -m src.reranker`` from the project
# root on Windows (or any OS), Python inserts ``src/`` into ``sys.path``
# rather than the project root, so a bare ``import config`` would fail with
# ``ModuleNotFoundError``.  The block below walks up from this file's real
# location until it finds ``config.py`` and prepends that directory to
# ``sys.path``, making the import robust regardless of the invocation method:
#
#   python src/reranker.py          (project root on sys.path  ✓)
#   python -m src.reranker          (src/ on sys.path, root missing ✗ → fixed)
#   import src.reranker             (called from pipeline.py in root  ✓)
# ---------------------------------------------------------------------------

import sys
from pathlib import Path as _Path

def _ensure_project_root_on_path() -> None:
    """Insert the project root (directory containing config.py) into sys.path."""
    current = _Path(__file__).resolve().parent
    for candidate in [current, *current.parents]:
        if (candidate / "config.py").exists():
            root_str = str(candidate)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
            return
    # config.py not found — proceed anyway; import will surface the real error.

_ensure_project_root_on_path()

import sys as _sys  # re-import after path fix (no-op, just for clarity)
import time
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
from loguru import logger

import config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Fields every valid candidate dict must expose.
_REQUIRED_CANDIDATE_FIELDS: Final[frozenset[str]] = frozenset(
    {"chunk_id", "retrieval_text", "payload"}
)

#: Key written into every output dict to carry the cross-encoder score.
_RERANK_SCORE_KEY: Final[str] = "rerank_score"

#: Key written into every output dict to preserve the retriever score.
_ORIGINAL_SCORE_KEY: Final[str] = "original_score"

#: Input score key from the retriever output schema.
_RETRIEVER_SCORE_KEY: Final[str] = "score"


# ---------------------------------------------------------------------------
# Internal statistics dataclass
# ---------------------------------------------------------------------------


@dataclass
class _RerankerStats:
    """Mutable counter bag describing one ``rerank()`` call."""

    query: str = ""
    candidates_in: int = 0
    candidates_valid: int = 0
    candidates_out: int = 0
    device: str = ""
    build_pairs_time: float = 0.0
    inference_time: float = 0.0
    total_time: float = 0.0


# ---------------------------------------------------------------------------
# Validated candidate wrapper
# ---------------------------------------------------------------------------


@dataclass
class _ValidCandidate:
    """A single retriever candidate that has passed validation.

    Attributes:
        chunk_id: Unique chunk identifier.
        retrieval_text: Hierarchy-aware text passed to the cross-encoder.
        original_score: The retriever's fused score, preserved verbatim.
        payload: Full payload dict forwarded to the output.
        document: Document label (may be empty string if absent).
        section: Section identifier (may be empty string if absent).
        article: Article identifier (may be empty string if absent).
        original: The original input dict, kept for output construction.
    """

    chunk_id: str
    retrieval_text: str
    original_score: float
    payload: dict[str, Any]
    document: str
    section: str
    article: str
    original: dict[str, Any]


# ---------------------------------------------------------------------------
# Main reranker class
# ---------------------------------------------------------------------------


class CrossEncoderReranker:
    """Cross-Encoder reranker for the Indian Legal RAG system.

    The model is loaded exactly once per instance, on first call to
    :meth:`rerank` (or explicitly via :meth:`load_model`), and cached
    for every subsequent call.

    Usage
    -----
    >>> reranker = CrossEncoderReranker()
    >>> results = reranker.rerank(query="What does Article 21 guarantee?",
    ...                           candidates=retriever_results)
    """

    def __init__(
        self,
        model_name: str | None = None,
        top_k: int | None = None,
        batch_size: int | None = None,
        device: str | None = None,
    ) -> None:
        """Initialise configuration and logging.

        All parameters fall back to ``config.py`` values when omitted,
        so the class can be constructed with zero arguments in standard
        deployment.

        Args:
            model_name: HuggingFace model identifier or local path.
                Defaults to ``config.RERANKER_MODEL_NAME``.
            top_k: Maximum number of results returned by :meth:`rerank`.
                Defaults to ``config.RERANKER_TOP_K``.
            batch_size: Inference batch size for the cross-encoder.
                Defaults to ``config.RERANKER_BATCH_SIZE``.
            device: Device string ``"auto"``, ``"cuda"``, ``"mps"``, or
                ``"cpu"``.  Defaults to ``config.RERANKER_DEVICE``.
        """
        self._model_name: str = model_name or config.RERANKER_MODEL_NAME
        self._top_k: int = top_k if top_k is not None else config.RERANKER_TOP_K
        self._batch_size: int = (
            batch_size if batch_size is not None else config.RERANKER_BATCH_SIZE
        )
        self._device_cfg: str = device or config.RERANKER_DEVICE

        # Set lazily by load_model().
        self._model: Any = None
        self._resolved_device: str = ""
        self._last_stats: _RerankerStats | None = None

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
            config.LOG_DIR / "reranker.log",
            level="DEBUG",
            rotation=config.LOG_ROTATION,
            retention=config.LOG_RETENTION,
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Device detection
    # ------------------------------------------------------------------

    def detect_device(self) -> str:
        """Detect the best available compute device.

        Resolution order when ``device`` config is ``"auto"``:
        CUDA → MPS → CPU.

        Returns:
            A PyTorch device string: ``"cuda"``, ``"mps"``, or ``"cpu"``.
        """
        cfg = self._device_cfg.strip().lower()
        if cfg != "auto":
            logger.debug("Device forced to '{}' by configuration.", cfg)
            return cfg

        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
            elif (
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ):
                device = "mps"
            else:
                device = "cpu"
        except ImportError:
            logger.warning(
                "torch not importable during device detection — defaulting to 'cpu'."
            )
            device = "cpu"

        logger.debug("Auto-detected device: '{}'.", device)
        return device

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> bool:
        """Load the CrossEncoder model exactly once, caching the result.

        Subsequent calls are no-ops and always return ``True`` if the
        model was previously loaded successfully.

        Returns:
            ``True`` if the model is ready, ``False`` on any failure
            (missing dependency, download error, CUDA OOM, etc.).
            Errors are logged; this method never raises.
        """
        if self._model is not None:
            return True

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            logger.error(
                "sentence-transformers is not installed: {}. "
                "Install it with `pip install sentence-transformers`.",
                exc,
            )
            return False

        device = self.detect_device()

        try:
            logger.info(
                "Loading CrossEncoder '{}' on device '{}'…",
                self._model_name,
                device,
            )
            self._model = CrossEncoder(
                self._model_name,
                device=device,
                max_length=config.RERANKER_MAX_LENGTH,
            )
            self._resolved_device = device
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to load CrossEncoder '{}': {}", self._model_name, exc
            )
            self._model = None
            self._resolved_device = ""
            return False

        logger.info(
            "CrossEncoder '{}' loaded on '{}'.",
            self._model_name,
            self._resolved_device,
        )
        return True

    # ------------------------------------------------------------------
    # Candidate validation
    # ------------------------------------------------------------------

    def validate_candidates(
        self,
        candidates: list[dict[str, Any]],
    ) -> list[_ValidCandidate]:
        """Validate and wrap raw retriever candidates.

        Validation rules (any violation silently skips the candidate):

        * Candidate must be a ``dict``.
        * ``chunk_id`` must be a non-empty string.
        * ``retrieval_text`` must be a non-empty string.
        * ``payload`` must be a ``dict``.

        The original retriever score is read from the ``"score"`` key and
        defaults to ``0.0`` when absent, so that the original score is
        always present in the output regardless of whether the retriever
        populated it.

        Args:
            candidates: Raw dicts from
                :meth:`retriever.HybridRetriever.retrieve`.

        Returns:
            A list of :class:`_ValidCandidate` objects, one per accepted
            candidate, in the same order as the input.
        """
        valid: list[_ValidCandidate] = []

        for idx, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                logger.warning(
                    "Candidate at index {} is not a dict (type {}) — skipped.",
                    idx,
                    type(candidate).__name__,
                )
                continue

            chunk_id = candidate.get("chunk_id", "")
            retrieval_text = candidate.get("retrieval_text", "")
            payload = candidate.get("payload")

            if not chunk_id or not isinstance(chunk_id, str):
                logger.warning(
                    "Candidate at index {} missing or invalid 'chunk_id' — skipped.",
                    idx,
                )
                continue

            if not retrieval_text or not isinstance(retrieval_text, str):
                logger.warning(
                    "Candidate '{}' has empty or invalid 'retrieval_text' — skipped.",
                    chunk_id,
                )
                continue

            if not isinstance(payload, dict):
                logger.warning(
                    "Candidate '{}' missing or invalid 'payload' — skipped.",
                    chunk_id,
                )
                continue

            raw_score = candidate.get(_RETRIEVER_SCORE_KEY, 0.0)
            try:
                original_score = float(raw_score)
            except (TypeError, ValueError):
                logger.warning(
                    "Candidate '{}' has non-numeric 'score' ({!r}); "
                    "defaulting to 0.0.",
                    chunk_id,
                    raw_score,
                )
                original_score = 0.0

            valid.append(
                _ValidCandidate(
                    chunk_id=chunk_id,
                    retrieval_text=retrieval_text,
                    original_score=original_score,
                    payload=payload,
                    document=str(candidate.get("document") or ""),
                    section=str(candidate.get("section") or ""),
                    article=str(candidate.get("article") or ""),
                    original=candidate,
                )
            )

        skipped = len(candidates) - len(valid)
        if skipped:
            logger.warning(
                "{} of {} candidate(s) failed validation and were skipped.",
                skipped,
                len(candidates),
            )

        return valid

    # ------------------------------------------------------------------
    # Pair construction
    # ------------------------------------------------------------------

    def build_pairs(
        self,
        query: str,
        candidates: list[_ValidCandidate],
    ) -> list[tuple[str, str]]:
        """Construct ``(query, retrieval_text)`` pairs for the CrossEncoder.

        Args:
            query: The raw user query string.
            candidates: Validated candidates from
                :meth:`validate_candidates`.

        Returns:
            A list of 2-tuples in the same order as *candidates*, ready
            for a single batched ``CrossEncoder.predict()`` call.
        """
        return [(query, candidate.retrieval_text) for candidate in candidates]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _run_inference(
        self,
        pairs: list[tuple[str, str]],
    ) -> np.ndarray | None:
        """Run batched CrossEncoder inference on *pairs*.

        Args:
            pairs: A list of ``(query, passage)`` 2-tuples.

        Returns:
            A 1-D NumPy array of raw logit scores (one per pair), or
            ``None`` if inference fails for any reason (logged).
        """
        try:
            raw = self._model.predict(
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return np.asarray(raw, dtype=float).ravel()
        except Exception as exc:  # noqa: BLE001
            logger.error("CrossEncoder inference failed: {}", exc)
            return None

    # ------------------------------------------------------------------
    # Sorting
    # ------------------------------------------------------------------

    def sort_results(
        self,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Sort reranked result dicts by ``rerank_score`` descending.

        Args:
            results: Output dicts with a ``"rerank_score"`` key.

        Returns:
            The same list sorted in-place, returned for convenience.
        """
        results.sort(key=lambda r: r[_RERANK_SCORE_KEY], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Rerank retrieval candidates using the CrossEncoder model.

        Pipeline
        --------
        1. Validate *query* and *candidates*.
        2. Ensure the model is loaded (lazy, cached).
        3. Build ``(query, retrieval_text)`` pairs.
        4. Run a single batched inference call.
        5. Attach ``rerank_score`` and ``original_score`` to every output.
        6. Sort descending by ``rerank_score``.
        7. Truncate to *top_k* and return.

        Args:
            query: The raw user query.  Must be a non-empty string.
            candidates: Raw result dicts from
                :meth:`retriever.HybridRetriever.retrieve`.  May be
                empty; an empty list is returned immediately.
            top_k: Maximum number of results to return.  Defaults to
                the ``RERANKER_TOP_K`` config value.  Capped at the
                number of valid candidates if fewer are available.

        Returns:
            A list of output dicts, each containing all original
            retriever fields plus ``"rerank_score"`` and
            ``"original_score"``, sorted by ``"rerank_score"``
            descending.  Returns an empty list on any unrecoverable
            error (model load failure, inference failure) after logging.

        Raises:
            ValueError: If *query* is empty or whitespace-only.
        """
        t_total_start = time.perf_counter()

        # ── 1. Validate query ──────────────────────────────────────────
        if not query or not query.strip():
            raise ValueError("query must not be empty.")

        resolved_top_k = top_k if top_k is not None else self._top_k

        stats = _RerankerStats(query=query, candidates_in=len(candidates))
        logger.info(
            "Reranker received {} candidate(s) for query: {!r}",
            len(candidates),
            query,
        )

        # ── 2. Handle empty input immediately ─────────────────────────
        if not candidates:
            logger.warning("Empty candidate list — returning immediately.")
            stats.total_time = time.perf_counter() - t_total_start
            self.print_statistics(stats)
            return []

        # ── 3. Validate candidates ─────────────────────────────────────
        valid_candidates = self.validate_candidates(candidates)
        stats.candidates_valid = len(valid_candidates)

        if not valid_candidates:
            logger.error(
                "All {} candidate(s) failed validation — nothing to rerank.",
                len(candidates),
            )
            stats.total_time = time.perf_counter() - t_total_start
            self.print_statistics(stats)
            return []

        # ── 4. Load model (lazy, cached) ───────────────────────────────
        if not self.load_model():
            logger.error(
                "Model unavailable — cannot rerank. "
                "Returning candidates sorted by original retriever score."
            )
            fallback = self._build_fallback_output(valid_candidates, resolved_top_k)
            stats.candidates_out = len(fallback)
            stats.device = "unavailable"
            stats.total_time = time.perf_counter() - t_total_start
            self.print_statistics(stats)
            return fallback

        stats.device = self._resolved_device

        # ── 5. Build pairs ─────────────────────────────────────────────
        t_pairs_start = time.perf_counter()
        pairs = self.build_pairs(query, valid_candidates)
        stats.build_pairs_time = time.perf_counter() - t_pairs_start

        logger.debug(
            "Built {} (query, passage) pair(s) for CrossEncoder inference.",
            len(pairs),
        )

        # ── 6. Batch inference ─────────────────────────────────────────
        t_inference_start = time.perf_counter()
        scores = self._run_inference(pairs)
        stats.inference_time = time.perf_counter() - t_inference_start

        if scores is None:
            logger.error(
                "Inference failed — returning candidates sorted by "
                "original retriever score."
            )
            fallback = self._build_fallback_output(valid_candidates, resolved_top_k)
            stats.candidates_out = len(fallback)
            stats.total_time = time.perf_counter() - t_total_start
            self.print_statistics(stats)
            return fallback

        logger.info(
            "CrossEncoder inference complete — {} score(s) in {:.3f}s on '{}'.",
            len(scores),
            stats.inference_time,
            self._resolved_device,
        )

        # ── 7. Assemble output dicts ───────────────────────────────────
        output: list[dict[str, Any]] = []
        for candidate, raw_score in zip(valid_candidates, scores):
            rerank_score = float(raw_score)
            result = {
                key: value
                for key, value in candidate.original.items()
                if key != _RETRIEVER_SCORE_KEY  # replaced by original_score below
            }
            result[_RERANK_SCORE_KEY] = rerank_score
            result[_ORIGINAL_SCORE_KEY] = candidate.original_score
            output.append(result)

        # ── 8. Sort and truncate ───────────────────────────────────────
        output = self.sort_results(output)
        output = output[:resolved_top_k]

        stats.candidates_out = len(output)
        stats.total_time = time.perf_counter() - t_total_start

        logger.info(
            "Reranking complete — {} result(s) returned (top_k={}).",
            len(output),
            resolved_top_k,
        )
        self.print_statistics(stats)
        return output

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def print_statistics(self, stats: _RerankerStats) -> None:
        """Print and log a statistics summary for one reranking run.

        Args:
            stats: The completed :class:`_RerankerStats` for the run.
        """
        print(f"\n{'─' * 48}")
        print(f"  Reranker Statistics")
        print(f"{'─' * 48}")
        print(f"  Query             : {stats.query!r}")
        print(f"  Candidates In     : {stats.candidates_in}")
        print(f"  Candidates Valid  : {stats.candidates_valid}")
        print(f"  Candidates Out    : {stats.candidates_out}")
        print(f"  Device            : {stats.device or 'n/a'}")
        print(f"  Pair Build Time   : {stats.build_pairs_time:.3f}s")
        print(f"  Inference Time    : {stats.inference_time:.3f}s")
        print(f"  Total Time        : {stats.total_time:.3f}s")
        print(f"{'─' * 48}\n")

        logger.info(
            "Reranker summary — query={!r} in={} valid={} out={} device={} "
            "pairs={:.3f}s inference={:.3f}s total={:.3f}s",
            stats.query,
            stats.candidates_in,
            stats.candidates_valid,
            stats.candidates_out,
            stats.device or "n/a",
            stats.build_pairs_time,
            stats.inference_time,
            stats.total_time,
        )

        self._last_stats = stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_fallback_output(
        valid_candidates: list[_ValidCandidate],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Build a degraded output sorted by original retriever score.

        Used when the cross-encoder is unavailable or inference fails,
        so the pipeline is never blocked by a reranker outage.

        Args:
            valid_candidates: Candidates that passed validation.
            top_k: Maximum number of results to return.

        Returns:
            Output dicts sorted by ``original_score`` descending, capped
            at *top_k*, each carrying ``rerank_score = original_score``
            so downstream consumers can treat the field as always-present.
        """
        results = []
        for candidate in valid_candidates:
            result = dict(candidate.original)
            result[_RERANK_SCORE_KEY] = candidate.original_score
            result[_ORIGINAL_SCORE_KEY] = candidate.original_score
            results.append(result)

        results.sort(key=lambda r: r[_RERANK_SCORE_KEY], reverse=True)
        return results[:top_k]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Demonstrate the reranker with a synthetic candidate list.

    Example
    -------
    .. code-block:: bash

        python src/reranker.py
        python -m src.reranker
    """
    import json

    query = " ".join(sys.argv[1:]) or "What is the punishment for murder under BNS?"

    mock_candidates: list[dict[str, Any]] = [
        {
            "score": 0.91,
            "chunk_id": "BNS_SEC101",
            "document": "BNS",
            "section": "Section 101",
            "article": "",
            "chunk_type": "section",
            "retrieval_text": (
                "Document: BNS\nChapter: VI\nSection: Section 101\n"
                "Title: Murder\nType: section\n"
                "Whoever commits murder shall be punished with death or "
                "imprisonment for life and shall also be liable to fine."
            ),
            "payload": {"chunk_id": "BNS_SEC101", "chunk_hash": "abc123"},
        },
        {
            "score": 0.78,
            "chunk_id": "BNS_SEC102",
            "document": "BNS",
            "section": "Section 102",
            "article": "",
            "chunk_type": "section",
            "retrieval_text": (
                "Document: BNS\nChapter: VI\nSection: Section 102\n"
                "Title: Culpable homicide not amounting to murder\nType: section\n"
                "Whoever commits culpable homicide not amounting to murder shall "
                "be punished with imprisonment for life."
            ),
            "payload": {"chunk_id": "BNS_SEC102", "chunk_hash": "def456"},
        },
        {
            "score": 0.65,
            "chunk_id": "CONST_ART21",
            "document": "Constitution of India",
            "section": "",
            "article": "21",
            "chunk_type": "article",
            "retrieval_text": (
                "Document: Constitution of India\nPart: III\n"
                "Article: Article 21\nTitle: Right to life and personal liberty\n"
                "Type: article\nNo person shall be deprived of his life or "
                "personal liberty except according to procedure established by law."
            ),
            "payload": {"chunk_id": "CONST_ART21", "chunk_hash": "ghi789"},
        },
    ]

    reranker = CrossEncoderReranker()
    results = reranker.rerank(query=query, candidates=mock_candidates)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()