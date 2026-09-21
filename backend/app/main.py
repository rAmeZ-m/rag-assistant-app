import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.query import router
from app.core.config import settings
from app.services.generation import Generator
from app.services.retrieval import Retriever
from app.utils.logging_config import setup_logging

setup_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading vector store and models...")
    app.state.retriever = Retriever()
    app.state.generator = Generator(app.state.retriever.cfg)
    yield
    logger.info("Shutting down")


app = FastAPI(title="RAG Document Assistant", version="1.0.0", lifespan=lifespan)

origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)