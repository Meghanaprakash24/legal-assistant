"""
src/indexer.py
---------------
Production-grade Qdrant indexing module for the Indian Legal RAG system.

Responsibilities
----------------
* Connect once to Qdrant Cloud and reuse the client across all operations
* Create / recreate / delete the ``legal_documents`` collection, with the
  vector size auto-detected from the first embedding (never hardcoded)
* Load embedded chunk records from  data/embeddings/
* Validate every record before indexing
* Deduplicate by ``chunk_hash`` against what is already stored in Qdrant
* Batch-upsert points (default batch_size = 100, configurable via config.py)
* Support full, incremental, and single-document indexing
* Expose collection management and health-check utilities
* Print per-run statistics

Does NOT generate embeddings, perform retrieval, build BM25 indexes, call
any LLM, or use LangChain/LlamaIndex. This module ONLY indexes vectors into
Qdrant.

Point IDs
---------
Qdrant only accepts unsigned integers or UUIDs as point IDs. Source chunk
identifiers (e.g. ``"BNS_SEC2"``) are neither, so they are never used
directly as the point ID. Instead, each point's ID is a deterministic
UUID5 derived from its ``chunk_id`` via
``uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id)``. This keeps re-indexing the
same chunk idempotent (the same chunk_id always maps to the same point
ID), while the original ``chunk_id`` is preserved inside the payload.

Python 3.11+  |  PEP 8  |  Google-style docstrings
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Iterator

from loguru import logger
from tqdm import tqdm

import config

# ---------------------------------------------------------------------------
# Lazy/optional heavy import
# ---------------------------------------------------------------------------
# qdrant-client is imported lazily inside methods that need it so this
# module can still be imported (e.g. for type checking, or by other
# pipeline stages) even before the dependency is installed. Any failure
# during the real connection attempt is caught, logged, and never crashes
# the rest of the pipeline.

# ---------------------------------------------------------------------------
# Required payload fields for validation
# ---------------------------------------------------------------------------

_REQUIRED_RECORD_FIELDS: Final[tuple[str, ...]] = ("id", "vector", "payload")
_REQUIRED_PAYLOAD_FIELDS: Final[tuple[str, ...]] = (
    "retrieval_text",
    "chunk_hash",
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class _RunStats:
    """Mutable counter bag threaded through a single indexing run."""

    collection: str
    documents_indexed: int = 0
    vectors_uploaded: int = 0
    duplicates: int = 0
    skipped: int = 0
    dimension: int = 0
    batches_uploaded: int = 0
    processing_time: float = 0.0

    @property
    def total_seen(self) -> int:
        """Total records considered across all processed documents."""
        return self.vectors_uploaded + self.duplicates + self.skipped


@dataclass
class PreparedPoint:
    """An internal, validated representation of a single point to upsert.

    Attributes:
        point_id: Deterministic UUID5 derived from ``chunk_id`` -- this is
            the actual ID stored in Qdrant. Qdrant only accepts unsigned
            integers or UUIDs as point IDs, so the original chunk
            identifier (e.g. ``"BNS_SEC2"``) can never be used directly.
        chunk_id: The original chunk identifier, preserved for logging
            and traceability, and stored inside the payload.
        vector: Dense embedding as a list of floats.
        payload: Full metadata payload, stored as-is in Qdrant.
        chunk_hash: Extracted for fast duplicate-detection lookups.
    """

    point_id: str
    chunk_id: str
    vector: list[float]
    payload: dict[str, Any]
    chunk_hash: str


# ---------------------------------------------------------------------------
# Main indexer class
# ---------------------------------------------------------------------------


class QdrantIndexer:
    """Index embedded legal chunks into a Qdrant collection.

    Usage
    -----
    >>> indexer = QdrantIndexer()
    >>> indexer.connect()
    >>> indexer.run()                       # full index of every document
    >>> indexer.index_document("BNS")       # index a single document
    >>> indexer.index_all(incremental=True) # incremental re-index

    The Qdrant client is created exactly once per ``QdrantIndexer`` instance
    via :meth:`connect` and reused for every subsequent operation.
    """

    def __init__(
        self,
        embeddings_dir: Path = config.EMBEDDINGS_DIR,
        collection_name: str = config.COLLECTION_NAME,
        batch_size: int = config.INDEXING_BATCH_SIZE,
        url: str = config.QDRANT_URL,
        api_key: str = config.QDRANT_API_KEY,
        timeout: float = config.QDRANT_TIMEOUT,
    ) -> None:
        """Initialise paths, configuration, and logging.

        The Qdrant client itself is NOT created here — call :meth:`connect`
        explicitly (or rely on :meth:`run` / :meth:`index_all` /
        :meth:`index_document`, which call it automatically) so that
        constructing a ``QdrantIndexer`` never fails due to network or
        credential issues.

        Args:
            embeddings_dir: Directory containing ``*_embeddings.json`` files.
            collection_name: Target Qdrant collection name.
            batch_size: Number of points uploaded per upsert call.
            url: Qdrant Cloud cluster URL.
            api_key: Qdrant Cloud API key.
            timeout: Request timeout in seconds for the Qdrant client.
        """
        self._embeddings_dir = embeddings_dir
        self._collection_name = collection_name
        self._batch_size = batch_size
        self._url = url
        self._api_key = api_key
        self._timeout = timeout

        self._client: Any = None  # qdrant_client.QdrantClient, set in connect()
        self._dimension: int = 0
        self._existing_hashes: set[str] = set()
        self._hashes_loaded: bool = False

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
            config.LOG_DIR / "indexer.log",
            level="DEBUG",
            rotation=config.LOG_ROTATION,
            retention=config.LOG_RETENTION,
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Connection & health
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """Create (or reuse) the Qdrant client connection.

        Subsequent calls are no-ops if a client already exists. Credentials
        are read exclusively from configuration (``config.py`` / environment
        variables) — never hardcoded.

        Returns:
            ``True`` if a client is available and reachable, ``False``
            otherwise.
        """
        if self._client is not None:
            return True

        if not self._url:
            logger.error(
                "QDRANT_URL is not configured. Set it via the QDRANT_URL "
                "environment variable or config.py."
            )
            return False

        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            logger.error(
                "qdrant-client is not installed: {}. "
                "Install it with `pip install qdrant-client`.",
                exc,
            )
            return False

        try:
            self._client = QdrantClient(
                url=self._url,
                port=None,
                api_key=self._api_key or None,
                timeout=self._timeout,
                check_compatibility=False,
                prefer_grpc=False,
                trust_env=config.QDRANT_TRUST_ENV,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to create Qdrant client: {}", exc)
            self._client = None
            return False

        if not self.health_check():
            logger.error("Qdrant connection established but health check failed.")
            self._client = None
            return False

        logger.info("Connected to Qdrant at '{}'.", self._url)
        return True

    def health_check(self) -> bool:
        """Verify the Qdrant cluster is reachable and responding.

        Returns:
            ``True`` if the cluster responds successfully, ``False`` on any
            network, authentication, or timeout failure.
        """
        if self._client is None:
            logger.warning("health_check called before connect().")
            return False

        try:
            self._client.get_collections()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.error("Qdrant health check failed: {}", exc)
            return False

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def collection_exists(self, name: str | None = None) -> bool:
        """Check whether a collection exists in Qdrant.

        Args:
            name: Collection name to check. Defaults to the configured
                collection name.

        Returns:
            ``True`` if the collection exists, ``False`` otherwise (or on
            error, logged as a warning).
        """
        if not self._require_client():
            return False

        target = name or self._collection_name
        try:
            return self._client.collection_exists(target)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not determine existence of collection '{}': {}",
                target,
                exc,
            )
            return False

    def create_collection(
        self, dimension: int, name: str | None = None
    ) -> bool:
        """Create a Qdrant collection if it does not already exist.

        Args:
            dimension: Vector size, auto-detected from the first embedding
                by the caller — never hardcoded here.
            name: Collection name. Defaults to the configured collection
                name.

        Returns:
            ``True`` if the collection exists after this call (either newly
            created or already present), ``False`` on failure.
        """
        if not self._require_client():
            return False

        target = name or self._collection_name

        if self.collection_exists(target):
            logger.info("Collection '{}' already exists — reusing.", target)
            return True

        try:
            from qdrant_client.models import Distance, VectorParams
        except ImportError as exc:
            logger.error("qdrant-client models unavailable: {}", exc)
            return False

        try:
            self._client.create_collection(
                collection_name=target,
                vectors_config=VectorParams(
                    size=dimension, distance=Distance.COSINE
                ),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to create collection '{}': {}", target, exc)
            return False

        logger.info(
            "Collection '{}' created — dimension={}, distance=Cosine.",
            target,
            dimension,
        )
        return True

    def delete_collection(self, name: str | None = None) -> bool:
        """Delete a Qdrant collection.

        Args:
            name: Collection name. Defaults to the configured collection
                name.

        Returns:
            ``True`` if deletion succeeded (or the collection was already
            absent), ``False`` on failure.
        """
        if not self._require_client():
            return False

        target = name or self._collection_name

        if not self.collection_exists(target):
            logger.info("Collection '{}' does not exist — nothing to delete.", target)
            return True

        try:
            self._client.delete_collection(collection_name=target)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to delete collection '{}': {}", target, exc)
            return False

        logger.info("Collection '{}' deleted.", target)
        self._existing_hashes.clear()
        self._hashes_loaded = False
        return True

    def recreate_collection(
        self, dimension: int, name: str | None = None
    ) -> bool:
        """Delete and recreate a collection from scratch.

        Args:
            dimension: Vector size for the new collection.
            name: Collection name. Defaults to the configured collection
                name.

        Returns:
            ``True`` if the collection was successfully recreated.
        """
        target = name or self._collection_name
        logger.info("Recreating collection '{}'.", target)

        if not self.delete_collection(target):
            return False
        return self.create_collection(dimension, target)

    def count_vectors(self, name: str | None = None) -> int:
        """Return the number of points currently stored in a collection.

        Args:
            name: Collection name. Defaults to the configured collection
                name.

        Returns:
            Point count, or ``0`` on failure / if the collection is absent.
        """
        if not self._require_client():
            return 0

        target = name or self._collection_name
        if not self.collection_exists(target):
            return 0

        try:
            result = self._client.count(collection_name=target, exact=True)
            return int(result.count)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to count vectors in '{}': {}", target, exc)
            return 0

    def collection_info(self, name: str | None = None) -> dict[str, Any]:
        """Return summary information about a collection.

        Args:
            name: Collection name. Defaults to the configured collection
                name.

        Returns:
            Dict with keys ``name``, ``exists``, ``vector_count``,
            ``dimension``, ``distance``, and ``status``. Fields default to
            empty/zero values when the collection is absent or an error
            occurs.
        """
        target = name or self._collection_name
        info: dict[str, Any] = {
            "name": target,
            "exists": False,
            "vector_count": 0,
            "dimension": 0,
            "distance": "",
            "status": "unknown",
        }

        if not self._require_client():
            return info

        if not self.collection_exists(target):
            return info

        info["exists"] = True
        info["vector_count"] = self.count_vectors(target)

        try:
            details = self._client.get_collection(collection_name=target)
            vectors_config = details.config.params.vectors
            # vectors_config may be a single VectorParams or a dict of named vectors.
            if hasattr(vectors_config, "size"):
                info["dimension"] = int(vectors_config.size)
                info["distance"] = str(vectors_config.distance)
            info["status"] = str(details.status)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fetch details for '{}': {}", target, exc)

        return info

    # ------------------------------------------------------------------
    # Loading embeddings
    # ------------------------------------------------------------------

    def load_embeddings(self, label: str) -> list[dict[str, Any]] | None:
        """Load the embeddings JSON array for a single document label.

        Args:
            label: Document label, e.g. ``"BNS"``.

        Returns:
            Parsed list of embedding record dicts, or ``None`` on any error.
        """
        stem = config.DOCUMENT_EMBEDDING_STEMS.get(label)
        if stem is None:
            logger.error("Unknown document label '{}'.", label)
            return None

        path = self._embeddings_dir / f"{stem}.json"
        if not path.exists():
            logger.error("Embeddings file not found for '{}': {}", label, path)
            return None
        if path.stat().st_size == 0:
            logger.warning("Empty embeddings file for '{}': {}", label, path)
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
                "Expected JSON array in {}, got {}.", path, type(raw).__name__
            )
            return None

        logger.info(
            "Loaded '{}' — {} embedding record(s) from {}.",
            label,
            len(raw),
            path.name,
        )
        return raw

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_embedding(
        self, record: dict[str, Any], expected_dimension: int = 0
    ) -> bool:
        """Validate a single embedding record before indexing.

        Rejection criteria
        ------------------
        * Missing ``id``, ``vector``, or ``payload`` top-level keys.
        * ``id`` is empty.
        * ``vector`` is not a non-empty list of numbers.
        * ``vector`` length disagrees with ``expected_dimension`` (when > 0).
        * ``payload`` is missing ``retrieval_text`` or ``chunk_hash``, or
          either is empty.

        Args:
            record: A raw embedding record dict (``{"id", "vector",
                "payload"}``).
            expected_dimension: The dimension established for this indexing
                run. Pass ``0`` to skip the dimension-consistency check
                (e.g. for the very first record).

        Returns:
            ``True`` if the record is safe to index, ``False`` otherwise.
            A warning is logged for every rejected record.
        """
        for key in _REQUIRED_RECORD_FIELDS:
            if key not in record:
                logger.warning(
                    "Rejected record — missing top-level field '{}': {}",
                    key,
                    record.get("id", "<unknown>"),
                )
                return False

        chunk_id = record.get("id")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            logger.warning("Rejected record — empty or invalid id.")
            return False

        vector = record.get("vector")
        if not isinstance(vector, list) or not vector:
            logger.warning("Rejected record '{}' — empty or invalid vector.", chunk_id)
            return False
        if not all(isinstance(v, (int, float)) for v in vector):
            logger.warning(
                "Rejected record '{}' — vector contains non-numeric values.",
                chunk_id,
            )
            return False
        if expected_dimension and len(vector) != expected_dimension:
            logger.warning(
                "Rejected record '{}' — vector dimension {} != expected {}.",
                chunk_id,
                len(vector),
                expected_dimension,
            )
            return False

        payload = record.get("payload")
        if not isinstance(payload, dict) or not payload:
            logger.warning("Rejected record '{}' — empty or invalid payload.", chunk_id)
            return False

        for key in _REQUIRED_PAYLOAD_FIELDS:
            value = payload.get(key)
            if not isinstance(value, str) or not value.strip():
                logger.warning(
                    "Rejected record '{}' — payload missing/empty '{}'.",
                    chunk_id,
                    key,
                )
                return False

        return True

    # ------------------------------------------------------------------
    # Point preparation
    # ------------------------------------------------------------------

    def prepare_points(
        self, records: list[dict[str, Any]], dimension: int
    ) -> list[PreparedPoint]:
        """Validate and convert raw embedding records into prepared points.

        The original ``chunk_id`` is never used directly as the Qdrant
        point ID — Qdrant only accepts unsigned integers or UUIDs as point
        IDs, and chunk IDs such as ``"BNS_SEC2"`` are neither. Instead, a
        deterministic UUID5 (``uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id)``)
        is derived from the chunk ID, so re-indexing the same chunk always
        produces the same point ID and upserts remain idempotent. The
        original ``chunk_id`` is preserved inside the payload.

        Args:
            records: Raw embedding record dicts loaded from disk.
            dimension: Expected vector dimension for this run.

        Returns:
            List of ``PreparedPoint`` objects that passed validation. Any
            record that fails :meth:`validate_embedding` is silently
            excluded (the caller is responsible for counting skips).
        """
        prepared: list[PreparedPoint] = []
        for record in records:
            if not self.validate_embedding(record, expected_dimension=dimension):
                continue

            chunk_id = record["id"]
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

            # Copy rather than mutate the loaded record in place, then
            # embed the original chunk_id so it travels into Qdrant inside
            # the payload (it is no longer recoverable from the point ID
            # itself once the UUID5 transform has been applied).
            payload = dict(record["payload"])
            payload["chunk_id"] = chunk_id

            prepared.append(
                PreparedPoint(
                    point_id=point_id,
                    chunk_id=chunk_id,
                    vector=[float(v) for v in record["vector"]],
                    payload=payload,
                    chunk_hash=payload["chunk_hash"],
                )
            )
        return prepared

    # ------------------------------------------------------------------
    # Duplicate detection
    # ------------------------------------------------------------------

    def _load_existing_hashes(self) -> None:
        """Populate the in-memory set of ``chunk_hash`` values already
        stored in the target collection.

        Scrolls through the entire collection once, fetching only the
        ``chunk_hash`` payload field, and caches the result on the
        instance so it is never re-fetched within the same run.
        """
        if self._hashes_loaded:
            return

        self._existing_hashes = set()
        self._hashes_loaded = True

        if not self._require_client():
            return
        if not self.collection_exists(self._collection_name):
            return

        try:
            next_offset = None
            while True:
                points, next_offset = self._client.scroll(
                    collection_name=self._collection_name,
                    limit=1000,
                    offset=next_offset,
                    with_payload=["chunk_hash"],
                    with_vectors=False,
                )
                for point in points:
                    chunk_hash = (point.payload or {}).get("chunk_hash")
                    if chunk_hash:
                        self._existing_hashes.add(chunk_hash)
                if next_offset is None:
                    break
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to preload existing chunk hashes — duplicate "
                "detection may be incomplete this run: {}",
                exc,
            )

        logger.info(
            "Preloaded {} existing chunk_hash value(s) from '{}'.",
            len(self._existing_hashes),
            self._collection_name,
        )

    # ------------------------------------------------------------------
    # Batch upload
    # ------------------------------------------------------------------

    def upsert_batch(self, points: list[PreparedPoint]) -> bool:
        """Upload a single batch of prepared points to Qdrant.

        Retries transient failures (network errors, timeouts) up to
        ``config.MAX_UPSERT_RETRIES`` times with linear backoff before
        giving up on the batch.

        Args:
            points: A batch of validated ``PreparedPoint`` objects.

        Returns:
            ``True`` if the batch was successfully upserted, ``False`` if
            all retries were exhausted.
        """
        if not points:
            return True
        if not self._require_client():
            return False

        try:
            from qdrant_client.models import PointStruct
        except ImportError as exc:
            logger.error("qdrant-client models unavailable: {}", exc)
            return False

        qdrant_points = [
            PointStruct(id=p.point_id, vector=p.vector, payload=p.payload)
            for p in points
        ]

        last_error: Exception | None = None
        for attempt in range(1, config.MAX_UPSERT_RETRIES + 1):
            try:
                self._client.upsert(
                    collection_name=self._collection_name,
                    points=qdrant_points,
                    wait=True,
                )
                return True
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "Batch upsert attempt {}/{} failed: {}",
                    attempt,
                    config.MAX_UPSERT_RETRIES,
                    exc,
                )
                if attempt < config.MAX_UPSERT_RETRIES:
                    time.sleep(config.RETRY_BACKOFF_SECONDS * attempt)

        logger.error(
            "Batch upsert failed permanently after {} attempt(s): {}",
            config.MAX_UPSERT_RETRIES,
            last_error,
        )
        return False

    # ------------------------------------------------------------------
    # Document-level orchestration
    # ------------------------------------------------------------------

    def index_document(
        self,
        label: str,
        stats: _RunStats | None = None,
        incremental: bool = True,
    ) -> _RunStats:
        """Index a single document's embeddings into Qdrant.

        Steps: load embeddings → establish/validate dimension → ensure
        collection exists → (optionally) preload existing hashes for
        duplicate detection → validate + prepare points → skip duplicates →
        upload in batches → update statistics.

        Args:
            label: Document label, e.g. ``"BNS"``, ``"Constitution"``.
            stats: An existing ``_RunStats`` to accumulate into (used by
                :meth:`index_all` to produce a combined summary). A new one
                is created if not provided.
            incremental: When ``True``, chunks whose ``chunk_hash`` already
                exists in the collection are skipped. When ``False``,
                duplicate checking against the existing collection is
                bypassed (still deduplicates within the batch itself).

        Returns:
            The (possibly shared) ``_RunStats`` instance, updated in place.
        """
        if stats is None:
            stats = _RunStats(collection=self._collection_name)

        if not self.connect():
            logger.error("Cannot index '{}' — Qdrant connection unavailable.", label)
            return stats

        t0 = time.perf_counter()

        records = self.load_embeddings(label)
        if records is None:
            return stats
        if not records:
            logger.warning("No embedding records for '{}'.", label)
            return stats

        # Establish dimension from the first record if not yet known.
        if self._dimension == 0:
            first_vector = records[0].get("vector")
            if isinstance(first_vector, list) and first_vector:
                self._dimension = len(first_vector)
                logger.info(
                    "Vector dimension auto-detected: {}", self._dimension
                )

        if self._dimension == 0:
            logger.error(
                "Could not determine vector dimension for '{}' — aborting.",
                label,
            )
            return stats

        if not self.create_collection(self._dimension):
            logger.error(
                "Collection unavailable for '{}' — aborting indexing.", label
            )
            return stats

        if incremental:
            self._load_existing_hashes()

        prepared = self.prepare_points(records, self._dimension)
        skipped_count = len(records) - len(prepared)
        stats.skipped += skipped_count

        # ── Deduplicate against existing collection + within this batch ──
        batch_seen_hashes: set[str] = set()
        to_upload: list[PreparedPoint] = []
        for point in prepared:
            if incremental and point.chunk_hash in self._existing_hashes:
                stats.duplicates += 1
                continue
            if point.chunk_hash in batch_seen_hashes:
                stats.duplicates += 1
                continue
            batch_seen_hashes.add(point.chunk_hash)
            to_upload.append(point)

        if stats.duplicates:
            logger.info(
                "'{}': {} duplicate chunk(s) skipped via chunk_hash.",
                label,
                stats.duplicates,
            )

        # ── Batch upload with progress bar ────────────────────────────
        uploaded = 0
        progress = tqdm(
            total=len(to_upload), desc=f"Indexing {label}", unit="vec", leave=False
        )
        try:
            for batch in self._chunked(to_upload, self._batch_size):
                if self.upsert_batch(batch):
                    uploaded += len(batch)
                    stats.batches_uploaded += 1
                    self._existing_hashes.update(p.chunk_hash for p in batch)
                else:
                    stats.skipped += len(batch)
                progress.update(len(batch))
        finally:
            progress.close()

        stats.vectors_uploaded += uploaded
        stats.dimension = self._dimension
        stats.documents_indexed += 1
        stats.processing_time += time.perf_counter() - t0

        logger.info(
            "'{}' indexed — uploaded={} duplicates={} skipped={}.",
            label,
            uploaded,
            stats.duplicates,
            stats.skipped,
        )
        return stats

    def index_all(self, incremental: bool = True) -> _RunStats:
        """Index every known document into Qdrant in a single run.

        Args:
            incremental: When ``True``, performs an incremental index that
                skips chunks already present (by ``chunk_hash``). When
                ``False``, every valid chunk is uploaded regardless of what
                already exists in the collection (still deduplicated within
                the run itself).

        Returns:
            A combined ``_RunStats`` summarising the entire run.
        """
        stats = _RunStats(collection=self._collection_name)

        if not self.connect():
            logger.error("Cannot run index_all — Qdrant connection unavailable.")
            return stats

        for label in config.DOCUMENT_EMBEDDING_STEMS:
            try:
                self.index_document(label, stats=stats, incremental=incremental)
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Unexpected error while indexing '{}': {}", label, exc
                )

        self.print_statistics(stats)
        return stats

    def run(self) -> _RunStats:
        """Convenience entry point: connect and run a full incremental index.

        Returns:
            The combined ``_RunStats`` for the run.
        """
        return self.index_all(incremental=True)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def print_statistics(self, stats: _RunStats) -> None:
        """Print a formatted statistics block to stdout.

        Args:
            stats: Completed ``_RunStats`` for the run.
        """
        print(f"\n{'─' * 44}")
        print(f"  Collection         : {stats.collection}")
        print(f"{'─' * 44}")
        print(f"  Documents Indexed  : {stats.documents_indexed}")
        print(f"  Vectors Uploaded   : {stats.vectors_uploaded}")
        print(f"  Duplicates         : {stats.duplicates}")
        print(f"  Skipped            : {stats.skipped}")
        print(f"  Dimension          : {stats.dimension or 'unknown'}")
        print(f"  Batches Uploaded   : {stats.batches_uploaded}")
        print(f"  Time               : {stats.processing_time:.2f} sec")
        print(f"{'─' * 44}\n")

        logger.info(
            "Run summary — collection='{}' documents={} uploaded={} "
            "duplicates={} skipped={} dimension={} batches={} time={:.2f}s",
            stats.collection,
            stats.documents_indexed,
            stats.vectors_uploaded,
            stats.duplicates,
            stats.skipped,
            stats.dimension,
            stats.batches_uploaded,
            stats.processing_time,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_client(self) -> bool:
        """Ensure a Qdrant client is available, attempting to connect if not.

        Returns:
            ``True`` if a client is ready to use, ``False`` otherwise.
        """
        if self._client is not None:
            return True
        return self.connect()

    @staticmethod
    def _chunked(
        items: list[PreparedPoint], size: int
    ) -> Iterator[list[PreparedPoint]]:
        """Yield successive batches of *size* items from *items*.

        Args:
            items: Full list of prepared points.
            size: Maximum number of items per batch.

        Yields:
            Successive sub-lists of length at most ``size``.
        """
        for i in range(0, len(items), size):
            yield items[i : i + size]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full indexing pipeline from the command line.

    Example
    -------
    .. code-block:: bash

        python src/indexer.py
    """
    indexer = QdrantIndexer()
    indexer.run()


if __name__ == "__main__":
    main()
