import logging

from fastapi import APIRouter, HTTPException, Request

from app.schemas.query import QueryRequest, QueryResponse
from app.services.generation import LLMUnavailable

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
def health(request: Request):
    retriever = request.app.state.retriever
    return {
        "status": "ok",
        "chunks": retriever.collection.count(),
        "llm_model": request.app.state.generator.model,
    }


@router.post("/query", response_model=QueryResponse)
def query(body: QueryRequest, request: Request):
    retriever = request.app.state.retriever
    generator = request.app.state.generator
    try:
        hits = retriever.retrieve(body.question)
        answer, sources = generator.generate(body.question, hits)
    except LLMUnavailable:
        raise HTTPException(
            status_code=503,
            detail="The language model is unavailable. Make sure Ollama is running.",
        )
    return QueryResponse(answer=answer, sources=sources)