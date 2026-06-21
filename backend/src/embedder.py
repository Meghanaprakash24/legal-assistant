"""
src/embedder.py
----------------
Production-grade dense embedding generator for the Indian Legal RAG system.

Responsibilities
----------------
* Load chunked JSON records from  data/chunked/
* Validate every chunk before embedding
* Deduplicate chunks by ``chunk_hash`` and reuse embeddings for duplicates
* Embed ONLY ``retrieval_text`` (never the raw ``text`` field) using a
  locally-run SentenceTransformers model (BAAI/bge-base-en-v1.5)
* Batch-embed with automatic GPU/MPS/CPU device selection
* Emit Qdrant-ready records:  {"id", "vector", "payload"}
* Save per-document embeddings JSON to  data/embeddings/
* Print per-document statistics

Does NOT connect to Qdrant, build BM25, rerank, call an LLM, or perform
retrieval. This module ONLY generates dense embeddings.

Python 3.11+  |  PEP 8  |  Google-style docstrings
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
from loguru import logger
from tqdm import tqdm

import config

# ---------------------------------------------------------------------------
# Lazy/optional heavy imports
# ---------------------------------------------------------------------------
# torch and sentence-transformers are imported lazily inside the class so
# that this module can still be imported (e.g. for type checking or by
# downstream modules) even in environments where they are not yet installed.
# Any failure during the real load is caught and logged, never crashes the
# whole pipeline run for the remaining documents.

# ---------------------------------------------------------------------------
# Required chunk fields for validation
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "chunk_id",
    "retrieval_text",
    "chunk_hash",
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EmbeddedRecord:
    """A single Qdrant-ready embedded record.

    Attributes:
        id: The original ``chunk_id`` — used as the Qdrant point ID.
        vector: Dense embedding as a plain Python list of floats.
        payload: Metadata dict suitable for direct use as a Qdrant payload.
    """

    id: str
    vector: list[float]
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON output."""
        return asdict(self)


@dataclass
class _DocStats:
    """Mutable counter bag threaded through a single document's embedding run."""

    document: str
    total_chunks: int = 0
    embedded: int = 0
    skipped: int = 0
    duplicates: int = 0
    dimension: int = 0
    processing_time: float = 0.0


# ---------------------------------------------------------------------------
# Main embedder class
# ---------------------------------------------------------------------------


class LegalEmbedder:
    """Generate dense embeddings for hierarchy-aware legal chunks.

    Usage
    -----
    >>> embedder = LegalEmbedder()
    >>> embedder.run()                      # embed every known document
    >>> embedder.embed_document("BNS")       # embed a single document

    The SentenceTransformers model is loaded exactly once per
    ``LegalEmbedder`` instance and reused across all documents and batches.
    """

    def __init__(
        self,
        chunked_dir: Path = config.CHUNKED_DIR,
        embeddings_dir: Path = config.EMBEDDINGS_DIR,
        model_name: str = config.EMBEDDING_MODEL_NAME,
        batch_size: int = config.EMBEDDING_BATCH_SIZE,
        normalize_embeddings: bool = config.NORMALIZE_EMBEDDINGS,
    ) -> None:
        """Initialise paths, configuration, and logging.

        The embedding model itself is NOT loaded here — it is loaded lazily
        on first use via :meth:`load_model` so that constructing a
        ``LegalEmbedder`` instance never fails due to missing model weights
        or hardware issues.

        Args:
            chunked_dir: Directory containing ``*_chunks.json`` files.
            embeddings_dir: Directory where embeddings JSON is written.
            model_name: HuggingFace/SentenceTransformers model identifier.
            batch_size: Number of texts encoded per forward pass.
            normalize_embeddings: Whether to L2-normalise output vectors.
        """
        self._chunked_dir = chunked_dir
        self._embeddings_dir = embeddings_dir
        self._model_name = model_name
        self._batch_size = batch_size
        self._normalize_embeddings = normalize_embeddings

        self._model: Any = None  # SentenceTransformer, loaded lazily
        self._device: str = "cpu"
        self._dimension: int = 0

        self._configure_logging()
        self._embeddings_dir.mkdir(parents=True, exist_ok=True)

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
            config.LOG_DIR / "embedder.log",
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

        Preference order: CUDA → MPS (Apple Silicon) → CPU. Any import or
        runtime failure while probing a device is treated as "unavailable"
        and the method falls back to the next option in the chain.

        Returns:
            One of ``"cuda"``, ``"mps"``, or ``"cpu"``.
        """
        try:
            import torch
        except ImportError:
            logger.warning("PyTorch not importable — falling back to CPU.")
            return "cpu"

        try:
            if torch.cuda.is_available():
                logger.info(
                    "CUDA device detected: {}", torch.cuda.get_device_name(0)
                )
                return "cuda"
        except Exception as exc:  # noqa: BLE001
            logger.warning("CUDA probe failed: {} — trying MPS.", exc)

        try:
            if torch.backends.mps.is_available():
                logger.info("Apple MPS device detected.")
                return "mps"
        except Exception as exc:  # noqa: BLE001
            logger.warning("MPS probe failed: {} — falling back to CPU.", exc)

        logger.info("No GPU/MPS available — using CPU.")
        return "cpu"

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> bool:
        """Load the SentenceTransformers model exactly once (cached).

        Subsequent calls are no-ops if the model is already loaded. The
        embedding dimension is auto-detected from the loaded model and
        never hardcoded.

        Returns:
            ``True`` if the model is ready to use, ``False`` on failure.
        """
        if self._model is not None:
            return True

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            logger.error(
                "sentence-transformers is not installed: {}. "
                "Install it with `pip install sentence-transformers`.",
                exc,
            )
            return False

        self._device = self.detect_device()

        try:
            t0 = time.perf_counter()
            self._model = SentenceTransformer(
                self._model_name, device=self._device
            )
            elapsed = time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to load model '{}': {}", self._model_name, exc
            )
            self._model = None
            return False

        try:
            self._dimension = int(
                self._model.get_sentence_embedding_dimension()
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not auto-detect embedding dimension: {} — "
                "will infer from first batch instead.",
                exc,
            )
            self._dimension = 0

        logger.info(
            "Model loaded: '{}' | device='{}' | dimension={} | {:.2f}s",
            self._model_name,
            self._device,
            self._dimension or "unknown",
            elapsed,
        )
        return True

    # ------------------------------------------------------------------
    # Loading chunk files
    # ------------------------------------------------------------------

    def load_chunk_file(self, label: str) -> list[dict[str, Any]] | None:
        """Load the chunk JSON array for a single document label.

        Args:
            label: Document label, e.g. ``"BNS"``.

        Returns:
            Parsed list of chunk dicts, or ``None`` on any error.
        """
        stem = config.DOCUMENT_CHUNK_STEMS.get(label)
        if stem is None:
            logger.error("Unknown document label '{}'.", label)
            return None

        path = self._chunked_dir / f"{stem}.json"
        if not path.exists():
            logger.error(
                "Chunk file not found for '{}': {}", label, path
            )
            return None
        if path.stat().st_size == 0:
            logger.warning("Empty chunk file for '{}': {}", label, path)
            return None

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON in {}: {}", path, exc)
            return None
        except UnicodeDecodeError as exc:
            logger.error("Encoding error in {}: {}", path, exc)
            return None

        if not isinstance(raw, list):
            logger.error(
                "Expected JSON array in {}, got {}.",
                path,
                type(raw).__name__,
            )
            return None

        logger.info("Loaded '{}' — {} chunk(s) from {}.", label, len(raw), path.name)
        return raw

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_chunk(self, chunk: dict[str, Any]) -> bool:
        """Validate that a chunk dict has everything required for embedding.

        Rejection criteria
        ------------------
        * Missing ``chunk_id``, ``retrieval_text``, or ``chunk_hash`` keys.
        * ``retrieval_text`` is empty or not a string.
        * ``chunk_id`` is empty.
        * ``chunk_hash`` is empty.

        Args:
            chunk: Raw chunk dict loaded from a ``*_chunks.json`` file.

        Returns:
            ``True`` if the chunk is safe to embed, ``False`` otherwise.
            A warning is logged for every rejected chunk.
        """
        for key in _REQUIRED_FIELDS:
            if key not in chunk:
                logger.warning(
                    "Rejected chunk — missing required field '{}': {}",
                    key,
                    chunk.get("chunk_id", "<unknown>"),
                )
                return False

        chunk_id = chunk.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            logger.warning("Rejected chunk — empty or invalid chunk_id.")
            return False

        retrieval_text = chunk.get("retrieval_text")
        if not isinstance(retrieval_text, str) or not retrieval_text.strip():
            logger.warning(
                "Rejected chunk '{}' — empty retrieval_text.", chunk_id
            )
            return False

        chunk_hash = chunk.get("chunk_hash")
        if not isinstance(chunk_hash, str) or not chunk_hash.strip():
            logger.warning(
                "Rejected chunk '{}' — empty chunk_hash.", chunk_id
            )
            return False

        return True

    # ------------------------------------------------------------------
    # Text preparation
    # ------------------------------------------------------------------

    def prepare_texts(
        self, chunks: list[dict[str, Any]]
    ) -> tuple[list[str], list[dict[str, Any]]]:
        """Extract the embeddable texts from validated chunks.

        Always embeds ``retrieval_text`` — never the raw ``text`` field —
        since ``retrieval_text`` already encodes the full legal hierarchy
        (document, chapter, section, clause, title) around the body text.

        Args:
            chunks: Pre-validated chunk dicts.

        Returns:
            Tuple of ``(texts, chunks)`` where ``texts[i]`` corresponds to
            ``chunks[i]``. Both lists are the same length and order.
        """
        texts = [chunk["retrieval_text"] for chunk in chunks]
        return texts, chunks

    # ------------------------------------------------------------------
    # Batch embedding
    # ------------------------------------------------------------------

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts in mini-batches using the loaded model.

        Args:
            texts: List of strings to embed (already deduplicated by the
                caller if desired).

        Returns:
            NumPy array of shape ``(len(texts), dimension)``. Returns an
            empty array of shape ``(0, 0)`` if the model is not loaded or
            ``texts`` is empty.

        Raises:
            RuntimeError: Re-raised after logging if encoding fails for
                reasons other than a clean empty input (e.g. OOM).
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        if self._model is None:
            logger.error("embed_batch called before the model was loaded.")
            return np.empty((0, 0), dtype=np.float32)

        all_vectors: list[np.ndarray] = []
        num_batches = (len(texts) + self._batch_size - 1) // self._batch_size

        progress = tqdm(
            total=len(texts),
            desc="Embedding",
            unit="chunk",
            leave=False,
        )

        try:
            for i in range(0, len(texts), self._batch_size):
                batch = texts[i : i + self._batch_size]
                try:
                    vectors = self._model.encode(
                        batch,
                        batch_size=self._batch_size,
                        convert_to_numpy=True,
                        normalize_embeddings=self._normalize_embeddings,
                        show_progress_bar=False,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Encoding failed for batch starting at index {}: {}",
                        i,
                        exc,
                    )
                    # Fill the failed batch with zero-vectors so output
                    # alignment with `texts` is preserved; caller can detect
                    # and skip these via dimension/zero checks if needed.
                    fallback_dim = self._dimension or (
                        all_vectors[0].shape[-1] if all_vectors else 768
                    )
                    vectors = np.zeros(
                        (len(batch), fallback_dim), dtype=np.float32
                    )
                all_vectors.append(np.asarray(vectors, dtype=np.float32))
                progress.update(len(batch))
        finally:
            progress.close()

        result = np.vstack(all_vectors) if all_vectors else np.empty((0, 0))

        if self._dimension == 0 and result.size:
            self._dimension = int(result.shape[-1])
            logger.info(
                "Embedding dimension inferred from output: {}", self._dimension
            )

        logger.debug(
            "Embedded {} text(s) across {} batch(es).", len(texts), num_batches
        )
        return result

    # ------------------------------------------------------------------
    # Document-level orchestration
    # ------------------------------------------------------------------

    def embed_document(self, label: str) -> list[EmbeddedRecord]:
        """Run the full embedding pipeline for a single document.

        Steps: load chunks → validate → deduplicate by ``chunk_hash`` →
        embed unique texts → fan back out to all valid chunks (duplicates
        reuse the embedding computed for their hash) → build Qdrant-ready
        records → save → print statistics.

        Args:
            label: Document label, e.g. ``"BNS"``, ``"Constitution"``.

        Returns:
            List of ``EmbeddedRecord`` objects persisted to disk. Empty
            list if the document could not be processed.
        """
        if label not in config.DOCUMENT_CHUNK_STEMS:
            logger.error("Unknown document label '{}'. Skipping.", label)
            return []

        if not self.load_model():
            logger.error(
                "Model unavailable — skipping document '{}'.", label
            )
            return []

        t0 = time.perf_counter()
        stats = _DocStats(document=label)

        raw_chunks = self.load_chunk_file(label)
        if raw_chunks is None:
            return []
        stats.total_chunks = len(raw_chunks)

        valid_chunks = [c for c in raw_chunks if self.validate_chunk(c)]
        stats.skipped = stats.total_chunks - len(valid_chunks)

        if not valid_chunks:
            logger.warning("No valid chunks to embed for '{}'.", label)
            self.print_statistics(stats)
            return []

        texts, valid_chunks = self.prepare_texts(valid_chunks)

        # ── Deduplicate by chunk_hash ────────────────────────────────
        unique_hash_to_index: dict[str, int] = {}
        unique_texts: list[str] = []
        chunk_hash_for_index: list[str] = []

        for chunk, text in zip(valid_chunks, texts):
            chunk_hash = chunk["chunk_hash"]
            if chunk_hash not in unique_hash_to_index:
                unique_hash_to_index[chunk_hash] = len(unique_texts)
                unique_texts.append(text)
            else:
                stats.duplicates += 1
            chunk_hash_for_index.append(chunk_hash)

        if stats.duplicates:
            logger.info(
                "'{}': {} duplicate chunk(s) detected by chunk_hash — "
                "embedding reused instead of recomputed.",
                label,
                stats.duplicates,
            )

        # ── Embed only unique texts ──────────────────────────────────
        unique_vectors = self.embed_batch(unique_texts)
        stats.dimension = self._dimension

        if unique_vectors.size == 0 and unique_texts:
            logger.error(
                "Embedding failed entirely for '{}' — no vectors produced.",
                label,
            )
            self.print_statistics(stats)
            return []

        # ── Fan back out to every valid chunk, reusing dedup vectors ──
        records: list[EmbeddedRecord] = []
        for chunk, chunk_hash in zip(valid_chunks, chunk_hash_for_index):
            vector_index = unique_hash_to_index[chunk_hash]
            vector = unique_vectors[vector_index]
            record = self._build_embedded_record(chunk, vector)
            records.append(record)
            stats.embedded += 1

        stats.processing_time = time.perf_counter() - t0

        self.save_embeddings(records, label)
        self.print_statistics(stats)
        return records

    def run(self) -> dict[str, list[EmbeddedRecord]]:
        """Embed every document known to the pipeline.

        Failures on one document never abort the remaining documents.

        Returns:
            Mapping of document label → list of ``EmbeddedRecord`` objects.
        """
        results: dict[str, list[EmbeddedRecord]] = {}
        for label in config.DOCUMENT_CHUNK_STEMS:
            try:
                results[label] = self.embed_document(label)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Unexpected error while embedding '{}': {}", label, exc
                )
                results[label] = []
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_embeddings(
        self, records: list[EmbeddedRecord], label: str
    ) -> Path:
        """Serialise Qdrant-ready records to ``<label>_embeddings.json``.

        Args:
            records: List of ``EmbeddedRecord`` objects to persist.
            label: Document label, e.g. ``"BNS"``.

        Returns:
            Path to the written file.
        """
        stem = config.DOCUMENT_EMBEDDING_STEMS[label]
        output_path = self._embeddings_dir / f"{stem}.json"
        payload = [r.to_dict() for r in records]

        try:
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.error("Failed to write {}: {}", output_path, exc)
            return output_path

        logger.info(
            "Saved embeddings: {} ({} record(s))", output_path, len(records)
        )
        return output_path

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def print_statistics(self, stats: _DocStats) -> None:
        """Print a formatted statistics block to stdout.

        Args:
            stats: Completed ``_DocStats`` for the document.
        """
        print(f"\n{'─' * 40}")
        print(f"  {stats.document}")
        print(f"{'─' * 40}")
        print(f"  Chunks      : {stats.total_chunks}")
        print(f"  Embedded    : {stats.embedded}")
        print(f"  Skipped     : {stats.skipped}")
        print(f"  Duplicates  : {stats.duplicates}")
        print(f"  Dimension   : {stats.dimension or 'unknown'}")
        print(f"  Time        : {stats.processing_time:.2f} sec")
        print(f"{'─' * 40}\n")

        logger.info(
            "'{}' summary — chunks={} embedded={} skipped={} "
            "duplicates={} dimension={} time={:.2f}s",
            stats.document,
            stats.total_chunks,
            stats.embedded,
            stats.skipped,
            stats.duplicates,
            stats.dimension,
            stats.processing_time,
        )

    # ------------------------------------------------------------------
    # Private — Qdrant-ready record builder
    # ------------------------------------------------------------------

    def _build_embedded_record(
        self, chunk: dict[str, Any], vector: np.ndarray
    ) -> EmbeddedRecord:
        """Build a single Qdrant-ready record from a validated chunk.

        The payload preserves the original chunk metadata (everything
        except the vector itself) so it can be used directly as a Qdrant
        point payload with no further transformation, and additionally
        records the embedding model name and dimension for traceability.

        Args:
            chunk: A validated chunk dict (already passed
                :meth:`validate_chunk`).
            vector: The dense embedding for this chunk's ``retrieval_text``.

        Returns:
            A populated ``EmbeddedRecord``.
        """
        payload: dict[str, Any] = {
            "document": chunk.get("document", ""),
            "chapter": chunk.get("chapter", ""),
            "part": chunk.get("part", ""),
            "section": chunk.get("section", ""),
            "section_no": chunk.get("section_no"),
            "article": chunk.get("article", ""),
            "clause": chunk.get("clause", ""),
            "title": chunk.get("title", ""),
            "retrieval_text": chunk.get("retrieval_text", ""),
            "text": chunk.get("text", ""),
            "hierarchy": chunk.get("hierarchy", []),
            "parent_chunk_id": chunk.get("parent_chunk_id", ""),
            "chunk_type": chunk.get("chunk_type", ""),
            "keywords": chunk.get("keywords", []),
            "chunk_hash": chunk.get("chunk_hash", ""),
            "version": chunk.get("version", 1),
            "page": chunk.get("page"),
            "source": chunk.get("source", ""),
            "embedding_model": self._model_name,
            "embedding_dimension": self._dimension,
        }

        return EmbeddedRecord(
            id=chunk["chunk_id"],
            vector=vector.tolist(),
            payload=payload,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full embedding pipeline from the command line.

    Example
    -------
    .. code-block:: bash

        python src/embedder.py
    """
    embedder = LegalEmbedder()
    embedder.run()


if __name__ == "__main__":
    main()