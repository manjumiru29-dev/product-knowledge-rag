-- ============================================================
-- RAG EXTENSION: adds vector-search capability on top of the
-- existing e-commerce schema (init.sql must be loaded first).
-- ============================================================

-- 1. Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. RAG documents table (384 dims to match all-MiniLM-L6-v2)
CREATE TABLE IF NOT EXISTS rag_documents (
    id            SERIAL PRIMARY KEY,
    source_type   VARCHAR(20)  NOT NULL
                  CHECK (source_type IN ('product', 'article', 'faq', 'brand, categories')),
    source_id     INT          NOT NULL,
    chunk_index   INT          NOT NULL DEFAULT 0,
    title         TEXT,
    content       TEXT         NOT NULL,
    embedding     vector(384),           -- all-MiniLM-L6-v2 dimension
    metadata      JSONB,
    created_at    TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (source_type, source_id, chunk_index)
);

-- 3. Indexes
CREATE INDEX IF NOT EXISTS rag_docs_embedding_idx
    ON rag_documents USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS rag_docs_source_idx
    ON rag_documents (source_type, source_id);

CREATE INDEX IF NOT EXISTS rag_docs_source_type_idx
    ON rag_documents (source_type);