from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException

from .agent import Agent
from .catalog import SHL_PRODUCT_CATALOG_PATH, load_catalog
from .llm import LLMClient
from .retrieval import CatalogIndex
from .schema import ChatRequest, ChatResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("shl-agent")

app = FastAPI(title="SHL Assessment Recommendation Agent")

_catalog_items = load_catalog(SHL_PRODUCT_CATALOG_PATH)
_index = CatalogIndex(_catalog_items)
_llm = LLMClient()
_agent = Agent(_index, _llm)

logger.info(
    "Loaded %d catalog items from %s (Individual Test Solutions only)",
    len(_catalog_items),
    SHL_PRODUCT_CATALOG_PATH,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not req.messages:
        raise HTTPException(status_code=400, detail="messages must be non-empty")
    try:
        return _agent.handle(req.messages)
    except Exception:
        logger.exception("agent error")
        return ChatResponse(
            reply="Sorry, I hit an internal error processing that. Could you rephrase your request?",
            recommendations=None,
            end_of_conversation=False,
        )
