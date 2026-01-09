"""Vector search utilities using FAISS."""

from typing import List, Tuple
from pathlib import Path
import pickle
import faiss
import numpy as np

from app.embeddings import embed_texts

ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "embeddings" / "faiss_index.bin"
META_PATH = ROOT / "embeddings" / "metadata.pkl"


class Retriever:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, "rb") as f:
            self.chunks: List[str] = pickle.load(f)

    def search(self, query: str) -> List[Tuple[str, float]]:
        """
        Retrieve top-k relevant text chunks for a query.

        Returns:
            List of (chunk_text, distance)
        """
        query_emb = embed_texts([query])
        distances, indices = self.index.search(
            query_emb.astype(np.float32), self.top_k
        )

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(dist)))

        return results
