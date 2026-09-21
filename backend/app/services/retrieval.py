import logging

import chromadb
from sentence_transformers import SentenceTransformer

from app.core.config import load_rag_config, settings

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self):
        self.cfg = load_rag_config()
        self.model = SentenceTransformer(self.cfg["embedding_model"])
        client = chromadb.PersistentClient(path=str(settings.vector_store_dir))
        self.collection = client.get_collection(self.cfg["collection_name"])
        logger.info("Vector store loaded: %d chunks", self.collection.count())

    def retrieve(self, question: str, k: int | None = None) -> list[dict]:
        k = k or self.cfg["top_k"]
        q_emb = self.model.encode([question], normalize_embeddings=True).tolist()
        res = self.collection.query(query_embeddings=q_emb, n_results=k * 3)

        hits, seen = [], set()
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            key = (meta["source"], meta["page"])
            if key in seen:  # chunk واحد بس من كل صفحة
                continue
            seen.add(key)
            hits.append(
                {
                    "text": doc,
                    "source": meta["source"],
                    "page": meta["page"],
                    "score": round(1 - dist, 3),
                }
            )
            if len(hits) == k:
                break
        return hits