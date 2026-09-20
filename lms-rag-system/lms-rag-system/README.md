# Product Knowledge RAG System

A locally grounded **Retrieval-Augmented Generation (RAG)** system that helps users get accurate product information from a PostgreSQL product knowledge database.

The system retrieves relevant information from the local database and uses it to generate answers. This helps keep responses grounded in the available product knowledge instead of relying on external information.

---

## 📌 Project Overview

The Product Knowledge RAG System is designed to answer questions related to product information such as:

- Product names
- Product prices
- Product descriptions
- Brands
- Categories
- Specifications
- FAQs
- Other product-related knowledge

The system follows a **Retrieve → Generate** approach.

When a user asks a question:

1. The question is received by the application.
2. Relevant information is retrieved from the product knowledge database.
3. The retrieved information is provided to the RAG pipeline.
4. The system generates an answer based on the retrieved information.

---

## 🎯 Objectives

- Provide answers based on a local product knowledge database.
- Retrieve relevant information for user questions.
- Reduce incorrect or unsupported answers.
- Use vector similarity to find relevant information.
- Provide a simple interface for interacting with the product knowledge base.

---

## 🔄 RAG Workflow

```text
                 User Question
                       │
                       ▼
                FastAPI / API
                       │
                       ▼
                Query Processing
                       │
                       ▼
              Embedding Generation
                       │
                       ▼
              Vector Similarity Search
                       │
                       ▼
              PostgreSQL + pgvector
                       │
                       ▼
              Relevant Information
                       │
                       ▼
                  RAG Chain
                       │
                       ▼
                Generated Answer
