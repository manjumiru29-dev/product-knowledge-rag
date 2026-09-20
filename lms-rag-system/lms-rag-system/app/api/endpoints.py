"""
API Endpoints
--------------
FastAPI router for the e-commerce RAG system.

Endpoints
---------
POST /query      — ask a question; returns LLM answer + sources
GET  /retrieve   — similarity search
POST /index      — index a single custom document
POST /index/bulk — bulk-index all brands, categories, products, articles, FAQs from DB
GET  /stats      — row counts from rag_documents
"""

import logging
from typing import Any, Dict, List, Optional 

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.chains.rag_chain import RAGChain
from app.services.indexing_service import IndexingService
from app.services.retrieval_service import RetrievalService
from app.database.connection import get_db
from sqlalchemy import text

logger = logging.getLogger(__name__)
router = APIRouter()

# Lazy singletons — instantiated on first use so import-time errors are clear.
_rag_chain: Optional[RAGChain] = None
_indexing: Optional[IndexingService] = None
_retrieval: Optional[RetrievalService] = None


def _get_rag_chain() -> RAGChain:
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain()
    return _rag_chain


def _get_indexing() -> IndexingService:
    global _indexing
    if _indexing is None:
        _indexing = IndexingService()
    return _indexing


def _get_retrieval() -> RetrievalService:
    global _retrieval
    if _retrieval is None:
        _retrieval = RetrievalService()
    return _retrieval


# ── request / response models ─────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Customer question")
    source_type: Optional[str] = Field(
        None,
        description=(
            "Restrict search to 'product', 'article', 'faq', 'brand', or 'category'. "
            "Omit to search all types."
        ),
    )


class IndexRequest(BaseModel):
    source_type: str = Field(
        ...,
        description="'product' | 'article' | 'faq' | 'brand' | 'category'",
    )
    source_id: int  = Field(..., description="ID of the row in the source table")
    content: str    = Field(..., description="Text to embed and store")
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# ── query ─────────────────────────────────────────────────────────────────────
@router.post("/query", summary="Ask a question")
async def query_rag(request: QueryRequest) -> Dict[str, Any]:
    """
    Run the full RAG pipeline:
    1. Embed the question.
    2. Retrieve the top-5 most relevant documents.
    3. Send question + context to the LLM.
    4. Return the answer together with source citations.

    Searchable source types: product, article, faq, brand, category.
    """
    try:
        return _get_rag_chain().query(
            question=request.question,
            source_type=request.source_type or "",
        )
    except Exception as exc:
        logger.error("POST /query error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ── retrieve only ─────────────────────────────────────────────────────────────
@router.get("/retrieve", summary="Similarity search without LLM")
async def retrieve(
    query: str = Query(..., description="Search text"),
    top_k: int = Query(5, ge=1, le=20),
    source_type: Optional[str] = Query(
        None,
        description="'product' | 'article' | 'faq' | 'brand' | 'category'",
    ),
    category: Optional[str] = Query(None, description="Filter by category name (partial match)"),
    threshold: float = Query(0.5, ge=0.0, le=1.0),
) -> Dict[str, Any]:
    try:
        docs = _get_retrieval().retrieve_relevant_documents(
            query=query,
            top_k=top_k,
            source_type=source_type,
            category_name=category,
            similarity_threshold=threshold,
        )
        return {"query": query, "count": len(docs), "results": docs}
    except Exception as exc:
        logger.error("GET /retrieve error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ── single index ──────────────────────────────────────────────────────────────
@router.post("/index", summary="Index a single document")
async def index_document(request: IndexRequest) -> Dict[str, Any]:
    """
    Index one document manually.
    source_type must be one of: product, article, faq, brand, category.
    """
    valid_types = {"product", "article", "faq", "brand", "category"}
    if request.source_type not in valid_types:
        raise HTTPException(
            status_code=422,
            detail=f"source_type must be one of: {', '.join(sorted(valid_types))}",
        )
    try:
        doc_id = _get_indexing().index_document(
            source_type=request.source_type,
            source_id=request.source_id,
            content=request.content,
            title=request.title,
            metadata=request.metadata,
        )
        return {"success": True, "rag_document_id": doc_id}
    except Exception as exc:
        logger.error("POST /index error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ── bulk index ────────────────────────────────────────────────────────────────
@router.post("/index/bulk", summary="Bulk-index all content from DB")
async def bulk_index() -> Dict[str, Any]:
    """
    Reads all brands, categories, products, articles, and FAQs that are not
    yet in rag_documents, generates embeddings, and inserts them.
    Safe to call multiple times (skips already-indexed rows).

    Indexed types:
    - brand      — brand name, industry, country, founded year, website
    - category   — category name, description, parent category
    - product    — name, description, price, SKU, status, category
    - faq        — question, answer, category
    - article    — title, content, type, difficulty, read time
    """
    try:
        totals = _get_indexing().bulk_index_from_db()
        return {
            "success": True,
            "indexed": totals,
            "total": sum(totals.values()),
        }
    except Exception as exc:
        logger.error("POST /index/bulk error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ── stats ─────────────────────────────────────────────────────────────────────
@router.get("/stats", summary="Row counts in rag_documents")
async def stats() -> Dict[str, Any]:
    """
    Returns the number of indexed documents per source type
    and the overall total.
    """
    from app.database.connection import execute_query

    rows = execute_query("""
        SELECT source_type, COUNT(*) AS total
        FROM   rag_documents
        GROUP BY source_type
        ORDER BY source_type
    """)
    counts = {r["source_type"]: int(r["total"]) for r in rows}
    return {
        "rag_documents": counts,
        "total": sum(counts.values()),
    }

   # ── ADD THESE NEW ENDPOINTS ────────────────────────────────

@router.get("/list")
async def list_items(
    source_type: str,
    top_k: int = 100,
) -> Dict[str, Any]:
    """
    List items directly from actual database tables.
    source_type: 'product' | 'category' | 'article' | 'faq'
    """
    try:
        with get_db() as db:

            # ── Products ───────────────────────────────────
            if source_type == "product":
                query = text("""
                    SELECT
                        p.id,
                        p.product_name  AS name,
                        p.sku,
                        p.description,
                        p.price,
                        p.discount_percentage,
                        p.status,
                        c.category_name AS category
                    FROM products p
                    LEFT JOIN categories c
                        ON p.category_id = c.id
                    WHERE p.status = 'available'
                    ORDER BY p.product_name ASC
                    LIMIT :top_k
                """)

            # ── Categories ─────────────────────────────────
            elif source_type == "category":
                query = text("""
                    SELECT
                        id,
                        category_name   AS name,
                        slug,
                        description,
                        icon
                    FROM categories
                    WHERE is_active = TRUE
                    AND parent_id IS NULL
                    ORDER BY sort_order ASC
                    LIMIT :top_k
                """)

            # ── Articles ───────────────────────────────────
            elif source_type == "article":
                query = text("""
                    SELECT
                        a.id,
                        a.title         AS name,
                        a.article_type,
                        a.difficulty_level,
                        a.read_time,
                        c.category_name AS category
                    FROM articles a
                    LEFT JOIN categories c
                        ON a.category_id = c.id
                    ORDER BY a.title ASC
                    LIMIT :top_k
                """)

            # ── FAQs ───────────────────────────────────────
            elif source_type == "faq":
                query = text("""
                    SELECT
                        f.id,
                        f.question       AS name,
                        f.answer         AS description,
                        c.category_name  AS category
                    FROM faqs f
                    LEFT JOIN categories c
                        ON f.category_id = c.id
                    ORDER BY f.order_num ASC
                    LIMIT :top_k
                """)

            # ── Brand (extracted from product names) ───────
            elif source_type == "brand":
                query = text("""
                    SELECT DISTINCT
                        SPLIT_PART(product_name, ' ', 1)
                        AS name,
                        COUNT(*) AS product_count
                    FROM products
                    GROUP BY SPLIT_PART(product_name, ' ', 1)
                    ORDER BY name ASC
                    LIMIT :top_k
                """)

            else:
                return {
                    "source_type": source_type,
                    "items": [],
                    "count": 0,
                    "error": f"Unknown source_type: {source_type}"
                }

            result = db.execute(query, {"top_k": top_k})
            columns = list(result.keys())
            items = []
            for row in result.fetchall():
                item = {}
                for i, col in enumerate(columns):
                    val = row[i]
                    if hasattr(val, "isoformat"):
                        val = val.isoformat()
                    item[col] = val
                items.append(item)

        return {
            "source_type": source_type,
            "items":       items,
            "count":       len(items),
        }

    except Exception as e:
        logger.error(f"List error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=str(e)
        )