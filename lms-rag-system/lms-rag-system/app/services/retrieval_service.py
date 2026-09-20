"""
RAG Retrieval Service
"""
import logging
from typing import Any, Dict, List, Optional
from sqlalchemy import text
from app.database.connection import get_db
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


def _vec_str(embedding: List[float]) -> str:
    return "[" + ",".join(str(v) for v in embedding) + "]"


class RetrievalService:

    def __init__(self) -> None:
        self.embedding_service = EmbeddingService()
        logger.info("RetrievalService initialised")

    def retrieve_relevant_documents(
        self,
        query: str,
        top_k: int = 5,
        source_type: Optional[str] = None,
        category_name: Optional[str] = None,
        similarity_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:

        query_embedding = self.embedding_service.generate_embedding(query)

        where_clauses = ["1 - (rd.embedding <=> CAST(:q_emb AS vector)) >= :threshold"]
        params: Dict[str, Any] = {
            "q_emb": _vec_str(query_embedding),
            "threshold": similarity_threshold,
            "top_k": top_k,
        }

        if source_type:
            where_clauses.append("rd.source_type = :source_type")
            params["source_type"] = source_type

        if category_name:
            where_clauses.append(
                "LOWER(COALESCE(cp.category_name, ca.category_name, cf.category_name, cat.category_name)) ILIKE :cat"
            )
            params["cat"] = f"%{category_name.lower()}%"

        where_sql = " AND ".join(where_clauses)

        sql = f"""
            SELECT
                rd.id, rd.source_type, rd.source_id, rd.title, rd.content, rd.metadata,
                1 - (rd.embedding <=> CAST(:q_emb AS vector)) AS similarity,
                p.product_name, p.price, p.discount_percentage,
                p.status AS product_status, p.sku,
                COALESCE(b.brand_name,        br.brand_name)        AS brand_name,
                COALESCE(b.industry,          br.industry)           AS industry,
                COALESCE(b.country_of_origin, br.country_of_origin) AS country_of_origin,
                COALESCE(b.founded_year,      br.founded_year)       AS founded_year,
                COALESCE(b.website,           br.website)            AS website,
                a.article_type,
                COALESCE(cp.category_name, ca.category_name, cf.category_name, cat.category_name) AS category_name,
                COALESCE(cpp.category_name, cpa.category_name, cpf.category_name)                 AS category_parent_name,
                cat.description AS category_doc_description
            FROM rag_documents rd
            LEFT JOIN products   p   ON rd.source_type = 'product'  AND rd.source_id = p.id
            LEFT JOIN brands     b   ON p.brand_id = b.id
            LEFT JOIN categories cp  ON p.category_id  = cp.id
            LEFT JOIN categories cpp ON cp.parent_id   = cpp.id
            LEFT JOIN articles   a   ON rd.source_type = 'article'  AND rd.source_id = a.id
            LEFT JOIN categories ca  ON a.category_id  = ca.id
            LEFT JOIN categories cpa ON ca.parent_id   = cpa.id
            LEFT JOIN faqs       f   ON rd.source_type = 'faq'      AND rd.source_id = f.id
            LEFT JOIN categories cf  ON f.category_id  = cf.id
            LEFT JOIN categories cpf ON cf.parent_id   = cpf.id
            LEFT JOIN brands     br  ON rd.source_type = 'brand'    AND rd.source_id = br.id
            LEFT JOIN categories cat ON rd.source_type = 'category' AND rd.source_id = cat.id
            WHERE {where_sql}
            ORDER BY similarity DESC
            LIMIT :top_k
        """

        with get_db() as db:
            result = db.execute(text(sql), params)
            columns = list(result.keys())
            docs = []
            for row in result.fetchall():
                doc = {col: row[i] for i, col in enumerate(columns)}
                doc["similarity"] = float(doc["similarity"])
                docs.append(doc)

        logger.info("Retrieved %d docs for query=%r", len(docs), query[:60])
        return docs

    def format_context(self, documents: List[Dict[str, Any]]) -> str:
        if not documents:
            return "No relevant content found in the knowledge base."

        lines = ["=== Relevant Content ===\n"]
        for idx, doc in enumerate(documents, 1):
            stype = doc["source_type"].upper()
            lines.append(f"[{idx}] {stype} — {doc['title'] or '(untitled)'}")
            lines.append(f"    Relevance        : {doc['similarity']:.2%}")

            if doc["source_type"] == "product":
                if doc.get("category_name"):
                    lines.append(f"    Category         : {doc['category_name']}")
                if doc.get("category_parent_name"):
                    lines.append(f"    Parent Category  : {doc['category_parent_name']}")
                price, disc = doc.get("price"), doc.get("discount_percentage")
                if price is not None:
                    ps = f"Rs.{price}"
                    if disc and float(disc) > 0:
                        ps += f" ({disc}% off)"
                    lines.append(f"    Price            : {ps}")
                if doc.get("product_status"):
                    lines.append(f"    Status           : {doc['product_status']}")
                if doc.get("sku"):
                    lines.append(f"    SKU              : {doc['sku']}")
                if doc.get("brand_name"):
                    lines.append(f"    Brand            : {doc['brand_name']}")
                if doc.get("industry"):
                    lines.append(f"    Industry         : {doc['industry']}")
                if doc.get("country_of_origin"):
                    lines.append(f"    Country          : {doc['country_of_origin']}")
                if doc.get("founded_year"):
                    lines.append(f"    Founded          : {doc['founded_year']}")
                if doc.get("website"):
                    lines.append(f"    Website          : {doc['website']}")

            elif doc["source_type"] == "brand":
                if doc.get("brand_name"):
                    lines.append(f"    Brand            : {doc['brand_name']}")
                if doc.get("industry"):
                    lines.append(f"    Industry         : {doc['industry']}")
                if doc.get("country_of_origin"):
                    lines.append(f"    Country          : {doc['country_of_origin']}")
                if doc.get("founded_year"):
                    lines.append(f"    Founded          : {doc['founded_year']}")
                if doc.get("website"):
                    lines.append(f"    Website          : {doc['website']}")

            elif doc["source_type"] == "category":
                if doc.get("category_name"):
                    lines.append(f"    Category         : {doc['category_name']}")
                if doc.get("category_doc_description"):
                    lines.append(f"    Description      : {doc['category_doc_description']}")

            elif doc["source_type"] == "article":
                if doc.get("category_name"):
                    lines.append(f"    Category         : {doc['category_name']}")
                if doc.get("article_type"):
                    lines.append(f"    Type             : {doc['article_type']}")

            elif doc["source_type"] == "faq":
                if doc.get("category_name"):
                    lines.append(f"    Category         : {doc['category_name']}")

            lines.append(f"    Content          : {doc['content'][:500]}{'…' if len(doc['content']) > 500 else ''}")
            lines.append("")

        return "\n".join(lines)