"""
FastAPI Application —  RAG System
--------------------------------------------
Start with:
    uvicorn app.main:app --host localhost --port 8002 --reload
"""

import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── app factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG System",
    description=(
        "Retrieval-Augmented Generation ."
        #"Based on the tutorial schema from ,adapted to the actual "
        #"init.sql dataset."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
allowed_origins = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── routes ────────────────────────────────────────────────────────────────────
from app.api.endpoints import router as api_router  # noqa: E402

app.include_router(api_router, prefix="/api")


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "service": "rag-system", "version": "1.0.0"}


@app.get("/", tags=["meta"])
def root():
    return {
        "service": "rag-system",
        "docs": "/docs",
        "health": "/health",
        "endpoints": [
            "POST /api/query",
            "GET  /api/retrieve",
            "POST /api/index",
            "POST /api/index/bulk",
            "GET  /api/stats",
        ],
    }


# ── dev runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "localhost"),
        port=int(os.getenv("PORT", 8002)),
        reload=os.getenv("ENVIRONMENT") == "development",
    )
