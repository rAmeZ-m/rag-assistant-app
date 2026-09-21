import logging
import re
import json

import ollama

from app.core.config import settings

logger = logging.getLogger(__name__)

NOT_FOUND = "I couldn't find this in the documents."


class LLMUnavailable(Exception):
    pass

def build_prompt(question: str, hits: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i + 1}] (source: {h['source']}, page {h['page']})\n{h['text']}"
        for i, h in enumerate(hits)
    )
    return f"""You are a document assistant. Answer the question using ONLY the context below.
Rules:
- If the answer is not in the context, say exactly: "{NOT_FOUND}"
- Cite sources inline ONLY as [1], [2] (the chunk numbers). Do not write file names or page numbers.
- Do not use outside knowledge.
- Reuse the wording of the context as much as possible.
- Use only the context chunks that directly answer the question. Ignore unrelated chunks.
- Keep the answer short (at most 3 sentences).

Context:
{context}

Question: {question}
Answer:"""


def _ngrams(text: str, n: int = 4) -> set:
    w = re.findall(r"\w+", text.lower())
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def attribute(answer: str, hits: list[dict], min_overlap: int = 2) -> list[int]:
    a = _ngrams(answer)
    scored = sorted(
        ((len(a & _ngrams(h["text"])), i) for i, h in enumerate(hits)),
        reverse=True,
    )
    return [i for c, i in scored if c >= min_overlap][:3]


def clean_answer(text: str) -> str:
    text = re.sub(r"\s*\((?:source|sources)\s*:[^)]*\)", "", text)
    text = re.sub(
        r"(?:according to|as (?:mentioned|stated|shown) in)\s*(?:figure \d+ in\s*)?\[\d+\](?:\s*(?:,|and)\s*\[\d+\])*,?\s*",
        "",
        text,
        flags=re.I,
    )
    text = re.sub(r"\s*(?:,\s*)?(?:as (?:mentioned|stated|shown) in|see)\s*\[\d+\]", "", text)
    text = re.sub(r"\s*\[\d+\]", "", text)
    text = re.sub(r",\.", ".", text)
    text = re.sub(r"\s+([.,])", r"\1", text).strip()
    return text[:1].upper() + text[1:]


class Generator:
    def __init__(self, cfg: dict):
        self.model = cfg["llm_model"]
        self.num_ctx = cfg["num_ctx"]
        self.threshold = cfg["similarity_threshold"]
        self.client = ollama.Client(host=settings.ollama_host)
        try:
            with open(settings.titles_path, encoding="utf-8") as f:
                self.titles = json.load(f)
        except FileNotFoundError:
            self.titles = {}
        logger.info("Generator ready: model=%s", self.model)

    def generate(self, question: str, hits: list[dict]) -> tuple[str, list[str]]:
        if not hits or hits[0]["score"] < self.threshold:
            return NOT_FOUND, []

        try:
            resp = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": build_prompt(question, hits)}],
                options={"temperature": 0, "num_ctx": self.num_ctx},
            )
        except Exception as e:
            logger.error("Ollama call failed: %s", e)
            raise LLMUnavailable(str(e)) from e

        answer = clean_answer(resp["message"]["content"])
        if "couldn't find this in the documents" in answer:
            return NOT_FOUND, []

        idx = attribute(answer, hits)
        if not idx: 
            return NOT_FOUND, []

        tags = "".join(f"[{j + 1}]" for j in range(len(idx)))
        sources = [
            f"[{j + 1}] {self.titles.get(hits[i]['source'], hits[i]['source'])} (page {hits[i]['page']})"
            for j, i in enumerate(idx)
        ]
        return f"{answer} {tags}", sources