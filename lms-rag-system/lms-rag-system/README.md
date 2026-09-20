# E-Commerce RAG System

Retrieval-Augmented Generation over the **products / articles / faqs** tables
in `init.sql`, built with PostgreSQL + pgvector, LangGraph, and FastAPI.

---

## Document Analysis: What the Tutorial Got Wrong

### Schema Mismatch (Critical)
The tutorial was written for an **LMS** database (`courses`, `lessons`,
`course_materials`).  The actual `init.sql` is an **e-commerce help-centre** schema:

| Tutorial assumed | init.sql actually has |
|---|---|
| `courses` | `products` (`Product_name`, `sku`, `price`, …) |
| `lessons` | `articles` (`article_type`, `difficulty_level`, …) |
| `course_materials` (with `embedding` column) | `faqs` + separate `rag_documents` table (added by this project) |
| `category VARCHAR(50)` on courses | `categories` table with `Category_name`, hierarchical via `parent_id` |

### Bug: `execute_query` was declared `async` but never awaited
```python
# Tutorial code (broken)
materials = await execute_query("""SELECT …""")

# Correct
materials = execute_query("""SELECT …""")
```

### Bug: Wrong column names
```python
# Tutorial used
"Product_name"   → products."Product_name"   ✓  (quoted; mixed-case)
"Category_name"  → categories."Category_name" ✓  (quoted; mixed-case)

# Without quotes PostgreSQL folds to lowercase → column not found
```

### Bug: `course_materials_embedding_idx` on non-existent table
The tutorial runs `CREATE INDEX … ON course_materials`.
`init.sql` has no such table.  This project creates `rag_documents` instead.

### Bug: LangGraph `StateGraph` API changed
Recent langgraph versions renamed some methods.  The code here uses the
stable `set_entry_point` / `add_edge` API.

### Bug: Similarity threshold too high for a cold index
Default of `0.7` returned zero results until the index is warm.
Changed to `0.5`.

---

## Architecture

```
User question
      │
      ▼
EmbeddingService.generate_embedding()   ← OpenAI ada-002
      │
      ▼
RetrievalService.retrieve_relevant_documents()
  ├─ Vector cosine-similarity search on rag_documents
  └─ LEFT JOINs to products / articles / faqs / categories
      │
      ▼
RetrievalService.format_context()
      │
      ▼
ChatOpenAI (gpt-4-turbo-preview)
      │
      ▼
Answer + source citations
```

**LangGraph workflow:** `retrieve → generate → END`

---

## Database Tables (from init.sql)

| Table | Rows | Role |
|---|---|---|
| `users` | 2 000 | Customers, admins, moderators |
| `categories` | 110 | Hierarchical product/article taxonomy |
| `products` | 2 000 | Catalog with price, SKU, specs (JSONB) |
| `articles` | 2 000 | Guides, tutorials, troubleshooting |
| `faqs` | 2 000 | Quick Q&A |
| `feedback` | 2 000 | Star ratings, reviews, helpful votes |
| `search_history` | 2 000 | Analytics: queries, clicks |
| **`rag_documents`** | — | **Added by this project** (embeddings live here) |

---

## Quick Start

```bash
# 1. Copy and fill in credentials
cp .env.example .env

# 2. Start Postgres with pgvector
docker-compose up -d

# 3. Load the base schema
docker exec -i ecom-rag-postgres \
  psql -U rag_user -d ecom_rag_db < init.sql

# 4. Apply the RAG migration (creates rag_documents table)
docker exec -i ecom-rag-postgres \
  psql -U rag_user -d ecom_rag_db < rag_migration.sql

# 5. Install Python deps
pip install -e .

# 6. Run the server
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

---

## Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/query` | Ask a question → LLM answer + sources |
| GET | `/api/retrieve` | Similarity search (no LLM) |
| POST | `/api/index` | Index a single custom document |
| POST | `/api/index/bulk` | Index all products / articles / FAQs |
| GET | `/api/stats` | Row counts in rag_documents |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |

---

## Example Requests

### Ask a question
```bash
curl -X POST http://localhost:8002/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Do you have any wireless headphones under $100?"}'
```

### Filter to products only
```bash
curl -X POST http://localhost:8002/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I set up my new router?", "source_type": "article"}'
```

### Similarity search without LLM
```bash
curl "http://localhost:8002/api/retrieve?query=bluetooth+speaker&top_k=3&source_type=product"
```

### Bulk index everything
```bash
curl -X POST http://localhost:8002/api/index/bulk
```

### Check indexing progress
```bash
curl http://localhost:8002/api/stats
```
