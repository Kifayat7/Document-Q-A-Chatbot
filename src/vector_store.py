
"""ChromaDB wrapper: one persistent collection per user."""

from __future__ import annotations

import hashlib
import math
import re

from config.settings import Settings
from src.utils import user_key


def collection_name(user_id: str, embedding_model: str) -> str:
    """Generate a unique collection name for each user and embedding model."""

    emb = hashlib.sha256(
        re.sub(r"\s+", "", embedding_model).encode()
    ).hexdigest()[:6]

    return f"u{user_key(user_id)[:20]}e{emb}"


class VectorStore:
    """Per-user vector store with isolated collections."""

    def __init__(self, settings: Settings, embeddings, user_id: str):

        from langchain_chroma import Chroma

        settings.chroma_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.name = collection_name(
            user_id,
            settings.embedding_model
        )

        self._store = Chroma(
            collection_name=self.name,
            embedding_function=embeddings,
            persist_directory=str(settings.chroma_dir),
            collection_metadata={
                "hnsw:space": "cosine"
            },
        )

        self._closed = False

    def add_chunks(self, chunks: list, doc_id: str) -> int:
        """Add document chunks to ChromaDB."""

        if not chunks:
            return 0

        ids = [
            f"{doc_id}:{c.metadata.get('chunk', i)}"
            for i, c in enumerate(chunks)
        ]

        self._store.add_documents(
            chunks,
            ids=ids
        )

        return len(ids)

    def search(
        self,
        query: str,
        k: int = 4,
        doc_ids: list[str] | None = None
    ) -> list[tuple]:
        """Return documents with normalized cosine similarity scores."""

        if self._closed:
            raise RuntimeError("VectorStore is closed.")

        if doc_ids is not None:
            if not doc_ids:
                return []

            where = (
                {"doc_id": doc_ids[0]}
                if len(doc_ids) == 1
                else {"doc_id": {"$in": list(doc_ids)}}
            )
        else:
            where = None

        # Chroma returns cosine DISTANCE, not similarity.
        results = self._store.similarity_search_with_score(
            query,
            k=k,
            filter=where
        )

        normalized_results = []

        for document, distance in results:

            distance = float(distance)

            # Cosine distance ranges from 0 to 2.
            # Convert it into a 0-to-1 similarity score.
            similarity = 1.0 - (distance / 2.0)

            similarity = max(
                0.0,
                min(1.0, similarity)
            )

            normalized_results.append(
                (document, similarity)
            )

        return normalized_results

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks belonging to one document."""

        if self._closed:
            raise RuntimeError("VectorStore is closed.")

        ids = self._store.get(
            where={"doc_id": doc_id},
            include=[]
        ).get("ids", [])

        if ids:
            self._store.delete(ids=ids)

        return len(ids)

    def count(self) -> int:
        """Return the number of stored chunks."""

        if self._closed:
            raise RuntimeError("VectorStore is closed.")

        return len(
            self._store.get(
                include=[]
            ).get("ids", [])
        )

    def delete_all(self) -> None:
        """Delete this user's collection."""

        if self._closed:
            raise RuntimeError("VectorStore is closed.")

        self._store.delete_collection()

    def close(self) -> None:
        """Release the underlying ChromaDB system resources."""

        if self._closed:
            return

        try:
            client = getattr(
                self._store,
                "_client",
                None
            )

            system = getattr(
                client,
                "_system",
                None
            )

            # Chroma currently does not expose a consistent
            # public close method across all versions.
            if system is not None:
                stop = getattr(system, "stop", None)

                if callable(stop):
                    stop()

        finally:
            self._closed = True
            self._store = None
