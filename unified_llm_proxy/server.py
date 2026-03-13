"""
Unified LLM API Proxy — 服务入口
启动方式：uvicorn unified_llm_proxy.server:app --host 0.0.0.0 --port 9000
"""

import os
import sys

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from fastapi import FastAPI
from .routes_chat import router as chat_router
from .routes_models import router as models_router
from .routes_anthropic import router as anthropic_router

app = FastAPI(title="Unified LLM API (OpenAI + Anthropic Compatible)", version="1.0.0")

app.include_router(chat_router)
app.include_router(models_router)
app.include_router(anthropic_router)


@app.get("/")
async def root():
    return {"status": "ok", "message": "Unified LLM API Proxy. Use /v1/chat/completions, /v1/messages, or /v1/models."}
