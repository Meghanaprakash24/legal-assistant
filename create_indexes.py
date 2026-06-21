"""
create_indexes.py
------------------
One-time setup script: creates the Qdrant payload indexes required for
metadata filtering (document, section, article, chunk_type, parent_chunk_id).

Qdrant requires an explicit index on any payload field used in a
FieldCondition/MatchValue filter -- without it, filtered searches fail with:

    Bad request: Index required but not found for "document" of one of
    the following types: [keyword]

Run this ONCE after indexing (or any time you point HybridRetriever at a
new/recreated collection):

    python create_indexes.py

Safe to re-run -- Qdrant returns success if the index already exists.

Python 3.11+
"""

from __future__ import annotations

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

import config

# Every field retriever.py / indexer.py filter on via FieldCondition.
_KEYWORD_FIELDS: tuple[str, ...] = (
    "document",
    "section",
    "article",
    "chunk_type",
    "parent_chunk_id",
)


def main() -> None:
    """Create a keyword payload index for every filterable field."""
    if not config.QDRANT_URL:
        raise SystemExit(
            "QDRANT_URL is not set. Export it before running this script."
        )

    client = QdrantClient(
        url=config.QDRANT_URL,
        port=None,
        api_key=config.QDRANT_API_KEY or None,
        timeout=config.QDRANT_TIMEOUT,
        prefer_grpc=False,
    )

    if not client.collection_exists(config.COLLECTION_NAME):
        raise SystemExit(
            f"Collection '{config.COLLECTION_NAME}' does not exist yet. "
            "Run the indexer first: python -m src.indexer"
        )

    for field_name in _KEYWORD_FIELDS:
        try:
            client.create_payload_index(
                collection_name=config.COLLECTION_NAME,
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.info("Index ready for payload field '{}'.", field_name)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to create index for '{}': {}", field_name, exc)

    logger.success(
        "Payload indexes created for collection '{}'. "
        "Metadata filtering is now enabled.",
        config.COLLECTION_NAME,
    )


if __name__ == "__main__":
    main()
