import time

from fastapi import APIRouter, Header
from fastapi.responses import JSONResponse

from .auth import verify_api_key, verify_api_key_any, query_usage, query_usage_all
from .registry import MODEL_REGISTRY

router = APIRouter()

# 隐藏模型不出现在 /v1/models 列表中（按需配置）
_HIDDEN_MODELS = set()


@router.get("/v1/models")
async def list_models(authorization: str = Header(default="", alias="Authorization")):
    api_key = verify_api_key(authorization)
    if not api_key:
        return JSONResponse(
            status_code=401,
            content={"error": {"message": "Invalid API key", "type": "authentication_error", "code": None}},
        )

    models = []
    for ext_name, (internal_name, backend, category) in MODEL_REGISTRY.items():
        if ext_name in _HIDDEN_MODELS:
            continue
        models.append({
            "id": ext_name,
            "object": "model",
            "created": 1700000000,
            "owned_by": backend,
            "permission": [],
            "root": ext_name,
            "parent": None,
        })

    return JSONResponse(content={
        "object": "list",
        "data": models,
    })


@router.get("/v1/usage")
async def get_usage(
    authorization: str = Header(default="", alias="Authorization"),
    x_api_key: str = Header(default="", alias="x-api-key"),
):
    """查询当前 API Key 的分模型调用次数"""
    api_key = verify_api_key_any(authorization, x_api_key)
    if not api_key:
        return JSONResponse(
            status_code=401,
            content={"error": {"message": "Invalid API key", "type": "authentication_error", "code": None}},
        )
    return {"api_key": api_key, "usage": query_usage(api_key)}


@router.get("/v1/usage/all")
async def get_usage_all():
    """查询所有 API Key 的分模型调用次数"""
    return query_usage_all()
