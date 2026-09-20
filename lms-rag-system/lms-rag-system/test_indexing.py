import json
import logging
import os
from dotenv import load_dotenv

load_dotenv()

from app.services.indexing_service import IndexingService, _vec_str
from app.services.embedding_service import EmbeddingService
from app.database.connection import execute_query, get_db
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BATCH_SIZE = 50


def batch_upsert(source_type, rows, contents, titles, embedding_svc):
    embeddings = embedding_svc.generate_embeddings_batch(contents)
    with get_db() as db:
        for row, content, title, embedding in zip(rows, contents, titles, embeddings):
            db.execute(
                text("""
                    INSERT INTO rag_documents
                        (source_type, source_id, chunk_index, title, content, embedding, metadata)
                    VALUES
                        (:source_type, :source_id, 0, :title, :content,
                         CAST(:embedding AS vector), NULL)
                    ON CONFLICT (source_type, source_id, chunk_index)
                    DO UPDATE SET
                        title      = EXCLUDED.title,
                        content    = EXCLUDED.content,
                        embedding  = EXCLUDED.embedding,
                        updated_at = CURRENT_TIMESTAMP
                """),
                {
                    "source_type": source_type,
                    "source_id": row["id"],
                    "title": title,
                    "content": content,
                    "embedding": _vec_str(embedding),
                },
            )
        db.commit()


def index_all():
    embedding_svc = EmbeddingService()
    total = 0

    # ── Brands ────────────────────────────────────────────────────────────────
    logger.info("Indexing brands...")
    brands = execute_query("""
        SELECT id, brand_name, industry, country_of_origin, founded_year, website, is_active
        FROM brands
    """)
    if brands:
        contents = [
            f"Brand: {r['brand_name']}. Industry: {r['industry']}. "
            f"Country: {r['country_of_origin']}. Founded: {r['founded_year']}. "
            f"Website: {r['website']}. Active: {r['is_active']}."
            for r in brands
        ]
        titles = [r['brand_name'] for r in brands]
        batch_upsert("brand", brands, contents, titles, embedding_svc)
        total += len(brands)
        logger.info(f"✅ Brands: {len(brands)} indexed")

    # ── Categories ────────────────────────────────────────────────────────────
    logger.info("Indexing categories...")
    categories = execute_query("""
        SELECT c.id, c.category_name, c.description, p.category_name AS parent_name
        FROM categories c
        LEFT JOIN categories p ON c.parent_id = p.id
    """)
    if categories:
        contents = [
            f"Category: {r['category_name']}. "
            f"Description: {r['description'] or ''}. "
            f"Parent Category: {r['parent_name'] or 'None'}."
            for r in categories
        ]
        titles = [r['category_name'] for r in categories]
        batch_upsert("category", categories, contents, titles, embedding_svc)
        total += len(categories)
        logger.info(f"✅ Categories: {len(categories)} indexed")

    # ── Products ──────────────────────────────────────────────────────────────
    logger.info("Indexing products...")
    products = execute_query("""
        SELECT
            p.id, p.product_name, p.description, p.price, p.sku,
            p.status, p.discount_percentage,
            b.brand_name, b.industry, b.country_of_origin,
            b.founded_year, b.website,
            c.category_name,
            cp.category_name AS category_parent_name
        FROM products p
        LEFT JOIN brands b ON b.id = p.brand_id
        JOIN categories c ON c.id = p.category_id
        LEFT JOIN categories cp ON cp.id = c.parent_id
    """)
    if products:
        for i in range(0, len(products), BATCH_SIZE):
            batch = products[i: i + BATCH_SIZE]
            contents = [
                f"Product: {r['product_name']}. Brand: {r['brand_name']}. "
                f"Industry: {r['industry']}. Country: {r['country_of_origin']}. "
                f"Founded: {r['founded_year']}. Website: {r['website']}. "
                f"Category: {r['category_name']}. "
                f"Parent Category: {r['category_parent_name'] or 'None'}. "
                f"Price: Rs.{r['price']}. Discount: {r['discount_percentage'] or 0}%. "
                f"Status: {r['status']}. SKU: {r['sku']}. "
                f"Description: {r['description'] or 'N/A'}."
                for r in batch
            ]
            titles = [r['product_name'] for r in batch]
            batch_upsert("product", batch, contents, titles, embedding_svc)
            total += len(batch)
        logger.info(f"✅ Products: {len(products)} indexed")

    # ── Articles ──────────────────────────────────────────────────────────────
    logger.info("Indexing articles...")
    articles = execute_query("""
        SELECT
            a.id, a.title, a.content, a.article_type,
            c.category_name,
            cp.category_name AS category_parent_name
        FROM articles a
        JOIN categories c ON c.id = a.category_id
        LEFT JOIN categories cp ON cp.id = c.parent_id
    """)
    if articles:
        for i in range(0, len(articles), BATCH_SIZE):
            batch = articles[i: i + BATCH_SIZE]
            contents = [
                f"Article: {r['title']}. Type: {r['article_type']}. "
                f"Category: {r['category_name']}. "
                f"Parent Category: {r['category_parent_name'] or 'None'}. "
                f"Content: {r['content']}"
                for r in batch
            ]
            titles = [r['title'] for r in batch]
            batch_upsert("article", batch, contents, titles, embedding_svc)
            total += len(batch)
            logger.info(f"  articles: {min(i+BATCH_SIZE, len(articles))}/{len(articles)}")
        logger.info(f"✅ Articles: {len(articles)} indexed")

    # ── FAQs ──────────────────────────────────────────────────────────────────
    logger.info("Indexing FAQs...")
    faqs = execute_query("""
        SELECT
            f.id, f.question, f.answer,
            c.category_name,
            cp.category_name AS category_parent_name
        FROM faqs f
        JOIN categories c ON c.id = f.category_id
        LEFT JOIN categories cp ON cp.id = c.parent_id
    """)
    if faqs:
        for i in range(0, len(faqs), BATCH_SIZE):
            batch = faqs[i: i + BATCH_SIZE]
            contents = [
                f"FAQ Question: {r['question']}. Answer: {r['answer']}. "
                f"Category: {r['category_name']}. "
                f"Parent Category: {r['category_parent_name'] or 'None'}."
                for r in batch
            ]
            titles = [r['question'] for r in batch]
            batch_upsert("faq", batch, contents, titles, embedding_svc)
            total += len(batch)
            logger.info(f"  FAQs: {min(i+BATCH_SIZE, len(faqs))}/{len(faqs)}")
        logger.info(f"✅ FAQs: {len(faqs)} indexed")

    logger.info(f"✅ Total indexed: {total}")
    return total


if __name__ == "__main__":
    index_all()