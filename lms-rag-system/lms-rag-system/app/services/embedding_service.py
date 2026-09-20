import logging
from typing import List
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:

    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.dimensions = 384
        logger.info("EmbeddingService loaded: all-MiniLM-L6-v2 (384 dims)")

    def _to_pgvector_str(self, embedding: List[float]) -> str:
        return '[' + ','.join(f'{v:.8f}' for v in embedding) + ']'

    def generate_embedding(self, text: str) -> List[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()