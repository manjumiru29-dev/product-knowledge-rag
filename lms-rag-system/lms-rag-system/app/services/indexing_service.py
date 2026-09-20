"""
Document Indexing Service
--------------------------
Populates the *rag_documents* table from the three content tables that
exist in init.sql:  products, articles, faqs.

Key fixes vs. the tutorial code
--------------------------------
* Uses the correct table/column names from init.sql
  (Product_name, Category_name, etc.)
* Adds a ``bulk_index_from_db`` method that reads the live DB and
  upserts embeddings so the system is ready to query immediately.
* Stores a human-readable text chunk that gives the LLM enough context.
* commit() is called once per bulk batch, not per row.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from app.database.connection import get_db, execute_query
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

# ── helpers ──────────────────────────────────────────────────────────────────

def _vec_str(embedding: List[float]) -> str:
    """Convert a Python float list to the '[x,y,…]' string pgvector expects."""
    return "[" + ",".join(str(v) for v in embedding) + "]"


class IndexingService:
    """Indexes content from the e-commerce DB into *rag_documents*."""

    def __init__(self) -> None:
        self.embedding_service = EmbeddingService()
        logger.info("IndexingService initialised")

    # ── single document ───────────────────────────────────────────────────────
    def index_document(
        self,
        source_type: str,        # 'product' | 'article' | 'faq'
        source_id: int,
        content: str,
        title: Optional[str] = None,
        chunk_index: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Upsert one rag_documents row and return its id.
        ON CONFLICT updates the embedding if the content changed.
        """
        embedding = self.embedding_service.generate_embedding(content)
        metadata_json = json.dumps(metadata) if metadata else None

        with get_db() as db:
            result = db.execute(
                text("""
                    INSERT INTO rag_documents
                        (source_type, source_id, chunk_index, title, content,
                         embedding, metadata)
                    VALUES
                        (:source_type, :source_id, :chunk_index, :title, :content,
                         CAST(:embedding AS vector), CAST(:metadata AS jsonb))
                    ON CONFLICT (source_type, source_id, chunk_index)
                    DO UPDATE SET
                        title      = EXCLUDED.title,
                        content    = EXCLUDED.content,
                        embedding  = EXCLUDED.embedding,
                        metadata   = EXCLUDED.metadata,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING id
                """),
                {
                    "source_type": source_type,
                    "source_id": source_id,
                    "chunk_index": chunk_index,
                    "title": title,
                    "content": content,
                    "embedding": _vec_str(embedding),
                    "metadata": metadata_json,
                },
            )
            doc_id = result.scalar()
            db.commit()

        logger.info("Indexed %s id=%s  → rag_documents id=%s", source_type, source_id, doc_id)
        return doc_id

    # ── bulk index from DB ────────────────────────────────────────────────────
    def bulk_index_from_db(self, batch_size: int = 50) -> Dict[str, int]:
        """
        Read products, articles, and FAQs from the database and index any
        row that does not yet have an embedding in rag_documents.

        Returns a summary dict: {'product': N, 'article': N, 'faq': N}
        """
        totals: Dict[str, int] = {"product": 0, "article": 0, "faq": 0}

        # ── products ──────────────────────────────────────────────────────────
        products = execute_query("""
            SELECT p.id,
                   p."Product_name" AS title,
                   p.description,
                   p.price,
                   p.discount_percentage,
                   p.status,
                   p.sku,
                   c."Category_name" AS category
            FROM   products p
            JOIN   categories c ON c.id = p.category_id
            WHERE  p.id NOT IN (
                SELECT source_id FROM rag_documents WHERE source_type = 'product'
            )
        """)

        for i in range(0, len(products), batch_size):
            batch = products[i : i + batch_size]
            contents = []
            for row in batch:
                discount_note = (
                    f" ({row['discount_percentage']}% off)"
                    if row.get("discount_percentage") and float(row["discount_percentage"]) > 0
                    else ""
                )
                contents.append(
                    f"Product: {row['title']}\n"
                    f"SKU: {row['sku']}\n"
                    f"Category: {row['category']}\n"
                    f"Price: ${row['price']}{discount_note}\n"
                    f"Status: {row['status']}\n"
                    f"Description: {row['description'] or 'N/A'}"
                )

            embeddings = self.embedding_service.generate_embeddings_batch(contents)
            self._bulk_upsert("product", batch, contents, embeddings)
            totals["product"] += len(batch)
            logger.info("Indexed %d products so far", totals["product"])

        # ── articles ──────────────────────────────────────────────────────────
        articles = execute_query("""
            SELECT a.id,
                   a.title,
                   a.content,
                   a.article_type,
                   a.difficulty_level,
                   a.read_time,
                   c."Category_name" AS category
            FROM   articles a
            JOIN   categories c ON c.id = a.category_id
            WHERE  a.id NOT IN (
                SELECT source_id FROM rag_documents WHERE source_type = 'article'
            )
        """)

        for i in range(0, len(articles), batch_size):
            batch = articles[i : i + batch_size]
            contents = [
                f"Article: {row['title']}\n"
                f"Type: {row['article_type']}\n"
                f"Difficulty: {row['difficulty_level']}\n"
                f"Category: {row['category']}\n"
                f"Read time: {row['read_time']} min\n"
                f"Content: {row['content']}"
                for row in batch
            ]
            embeddings = self.embedding_service.generate_embeddings_batch(contents)
            self._bulk_upsert("article", batch, contents, embeddings)
            totals["article"] += len(batch)
            logger.info("Indexed %d articles so far", totals["article"])

        # ── faqs ──────────────────────────────────────────────────────────────
        faqs = execute_query("""
            SELECT f.id,
                   f.question,
                   f.answer,
                   c."Category_name" AS category
            FROM   faqs f
            JOIN   categories c ON c.id = f.category_id
            WHERE  f.id NOT IN (
                SELECT source_id FROM rag_documents WHERE source_type = 'faq'
            )
        """)

        for i in range(0, len(faqs), batch_size):
            batch = faqs[i : i + batch_size]
            contents = [
                f"FAQ: {row['question']}\nAnswer: {row['answer']}\nCategory: {row['category']}"
                for row in batch
            ]
            embeddings = self.embedding_service.generate_embeddings_batch(contents)
            self._bulk_upsert("faq", batch, contents, embeddings)
            totals["faq"] += len(batch)
            logger.info("Indexed %d FAQs so far", totals["faq"])

        logger.info("Bulk indexing complete: %s", totals)
        return totals

    # ── internal helpers ──────────────────────────────────────────────────────
    def _bulk_upsert(
        self,
        source_type: str,
        rows: List[dict],
        contents: List[str],
        embeddings: List[List[float]],
    ) -> None:
        """Upsert a batch of rows into rag_documents in a single transaction."""
        with get_db() as db:
            for row, content, embedding in zip(rows, contents, embeddings):
                title = row.get("title") or row.get("question")
                metadata = {
                    k: v for k, v in row.items()
                    if k not in ("id", "content", "question", "answer", "description")
                    and v is not None
                }
                db.execute(
                    text("""
                        INSERT INTO rag_documents
                            (source_type, source_id, chunk_index, title,
                             content, embedding, metadata)
                        VALUES
                            (:source_type, :source_id, 0, :title,
                             :content, CAST(:embedding AS vector), CAST(:metadata AS jsonb))
                        ON CONFLICT (source_type, source_id, chunk_index)
                        DO UPDATE SET
                            title      = EXCLUDED.title,
                            content    = EXCLUDED.content,
                            embedding  = EXCLUDED.embedding,
                            metadata   = EXCLUDED.metadata,
                            updated_at = CURRENT_TIMESTAMP
                    """),
                    {
                        "source_type": source_type,
                        "source_id": row["id"],
                        "title": title,
                        "content": content,
                        "embedding": _vec_str(embedding),
                        "metadata": json.dumps(metadata),
                    },
                )
            db.commit()