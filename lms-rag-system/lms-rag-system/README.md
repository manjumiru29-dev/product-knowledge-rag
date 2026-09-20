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
## 🔍 Main Components

1. Embedding Service

The embedding service converts text into numerical vector representations.

These vectors are used to find information that is semantically similar to the user's question.

2. Indexing Service

The indexing service processes product knowledge and prepares it for retrieval using vector embeddings.

3. Retrieval Service

The retrieval service searches the vector database and identifies relevant information for a user's question.

4. RAG Chain

The RAG chain connects the retrieved information with the generation process to produce an answer based on the available knowledge.

5. FastAPI

FastAPI provides the backend API through which queries and other application operations can be handled.

6. Streamlit Chatbot

The Streamlit interface provides a simple way for users to interact with the product knowledge chatbot.

---
## 🗄️ Database

The project uses PostgreSQL with pgvector for storing product knowledge and vector embeddings.

The database can contain information related to:

Products
Categories
Brands
FAQs
Articles
Search history

Vector embeddings are used to perform similarity-based retrieval.

## 📊 Key Features

- Local product knowledge retrieval
- Vector similarity search
- PostgreSQL database integration
- pgvector support
- RAG-based answer generation
- FastAPI backend
- Streamlit chatbot interface
- Product, FAQ, brand and category knowledge
- Docker-based database setup
- Modular application structure


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


