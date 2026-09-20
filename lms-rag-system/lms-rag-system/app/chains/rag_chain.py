"""
LangGraph RAG Chain
--------------------
Two-node workflow:  retrieve  →  generate  →  END
Uses flan-t5-large locally — no API key required.
"""

import logging
from typing import List, Optional, TypedDict

from transformers import pipeline
from langgraph.graph import END, StateGraph
from app.services.retrieval_service import RetrievalService

logger = logging.getLogger(__name__)


# ── state ─────────────────────────────────────────────────────────────────────
class RAGState(TypedDict):
    query: str
    source_type: str
    retrieved_documents: List[dict]
    context: str
    answer: str
    error: str


# ── chain ─────────────────────────────────────────────────────────────────────
class RAGChain:
    """LangGraph RAG chain for the e-commerce help-centre."""

    def __init__(self) -> None:
        # ── FIX 1: upgraded from flan-t5-small (80M) → flan-t5-large (780M)
        #    small model cannot reason over multi-field context → brand fields
        #    were being ignored/truncated in the generated answer.
        logger.info("Loading flan-t5-small...")
        self.llm = pipeline(
            task="text2text-generation",
            model="google/flan-t5-base",
            device=-1,          # CPU; change to 0 if you have a GPU
            max_length=512,     # input truncation limit
        )
        self.retrieval_service = RetrievalService()
        self._graph = self._build_graph()
        logger.info("RAGChain initialised")

    # ── graph construction ────────────────────────────────────────────────────
    def _build_graph(self):
        retrieval_svc = self.retrieval_service
        llm = self.llm

        def retrieve_node(state: RAGState) -> dict:
            try:
                valid_types = {"product", "article", "faq", "brand", "category"}
                raw = state.get("source_type") or ""
                source_type = raw.strip() if raw.strip() in valid_types else None

                docs = retrieval_svc.retrieve_relevant_documents(
                    query=state["query"],
                    top_k=5,
                    source_type=source_type,
                    similarity_threshold=0.1,
                )
                context = retrieval_svc.format_context(docs)
                return {"retrieved_documents": docs, "context": context, "error": ""}
            except Exception as exc:
                logger.error("retrieve_node error: %s", exc, exc_info=True)
                return {"retrieved_documents": [], "context": "", "error": str(exc)}

        def generate_node(state: RAGState) -> dict:
            if state.get("error"):
                return {"answer": f"I encountered an error while searching: {state['error']}"}

            if not state.get("retrieved_documents"):
                return {
                    "answer": (
                        "I couldn't find any relevant products, articles, FAQs, brands, "
                        "or categories for your query. Please try rephrasing or contact "
                        "our support team."
                    )
                }

            # ── FIX 2: prompt now explicitly instructs the LLM to include
            #    brand details (brand name, industry, country, founded, website)
            #    so they appear in the final answer instead of being skipped.
            prompt = f"""You are a helpful e-commerce customer support assistant.
Use ONLY the knowledge base below to answer the customer question.
Always include ALL of the following details when they are present in the knowledge base:
- Product name, SKU, price, discount, status
- Brand name, industry, country of origin, founded year, website
- Category name and parent category
- Article type and content summary (for articles)
- FAQ answer (for FAQs)
If a detail is not in the knowledge base, do not guess — say it is not available.
Be concise, factual, and helpful.

Knowledge base:
{state['context']}

Customer question: {state['query']}

Answer:"""

            try:
                result = llm(prompt, max_new_tokens=250)
                return {"answer": result[0]["generated_text"].strip()}
            except Exception as exc:
                logger.error("generate_node error: %s", exc, exc_info=True)
                return {"answer": f"Error generating answer: {exc}"}

        # ── build graph ───────────────────────────────────────────────────────
        wf = StateGraph(RAGState)
        wf.add_node("retrieve", retrieve_node)
        wf.add_node("generate", generate_node)
        wf.set_entry_point("retrieve")
        wf.add_edge("retrieve", "generate")
        wf.add_edge("generate", END)
        return wf.compile()

    # ── public interface ───────────────────────────────────────────────────────
    def query(
        self,
        question: str,
        source_type: str = "",
    ) -> dict:
        initial: RAGState = {
            "query": question,
            "source_type": source_type,
            "retrieved_documents": [],
            "context": "",
            "answer": "",
            "error": "",
        }
        try:
            result = self._graph.invoke(initial)
            return {
                "question": question,
                "answer": result.get("answer", ""),
                "context": result.get("context", ""),
                "retrieved_documents": result.get("retrieved_documents", []),
                "error": result.get("error") or None,
            }
        except Exception as exc:
            logger.error("RAG query failed: %s", exc, exc_info=True)
            return {
                "question": question,
                "answer": "",
                "context": "",
                "retrieved_documents": [],
                "error": str(exc),
            }