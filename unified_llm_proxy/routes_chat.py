import json
import time
import traceback
import uuid

from fastapi import APIRouter, Request, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from .auth import verify_api_key, record_usage
from .registry import (
    MODEL_REGISTRY,
    get_backend_client,
    get_local_extra_params,
)
from .adapters.genai_adapter import handle_genai_request
from .adapters.image_gen_adapter import handle_image_gen_request

router = APIRouter()

# 不传 temperature 的模型（按需配置）
_NO_TEMPERATURE_MODELS = set()


def _openai_error(status_code: int, message: str, error_type: str = "invalid_request_error"):
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "code": None}},
    )


@router.post("/v1/chat/completions")
async def chat_completions(request: Request, authorization: str = Header(default="", alias="Authorization")):
    # 1. 鉴权
    api_key = verify_api_key(authorization)
    if not api_key:
        return _openai_error(401, "Invalid API key", "authentication_error")

    # 2. 解析请求
    body = await request.json()
    model = body.get("model", "")
    messages = body.get("messages", [])
    stream = body.get("stream", False)

    if model not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        return _openai_error(404, f"Model '{model}' not found. Available: {available}")

    internal_name, backend, category = MODEL_REGISTRY[model]

    # 3. 非 OpenAI 协议的模型走适配器
    if backend == "genai":
        return await handle_genai_request(body, model, internal_name, api_key, stream)

    if backend == "image-gen":
        return await handle_image_gen_request(body, model, internal_name, api_key)

    # 4. 透明代理（openai-compatible / azure / local）
    try:
        client, actual_model = get_backend_client(backend, internal_name)
    except Exception as e:
        record_usage(api_key, model, False)
        return _openai_error(500, f"Backend error: {e}", "server_error")

    # 构建请求参数
    kwargs = {
        "model": actual_model,
        "messages": messages,
    }

    # 可选参数透传
    for param in ("max_tokens", "top_p", "frequency_penalty", "stop", "n"):
        if param in body:
            kwargs[param] = body[param]

    # temperature 处理
    if internal_name not in _NO_TEMPERATURE_MODELS:
        kwargs["temperature"] = body.get("temperature", 0.7)

    # 本地模型注入额外参数
    if backend == "local":
        extra = get_local_extra_params(internal_name)
        kwargs["extra_body"] = extra.pop("extra_body")
        kwargs.setdefault("temperature", extra["temperature"])
        kwargs.setdefault("top_p", extra["top_p"])
        kwargs["presence_penalty"] = extra["presence_penalty"]

    # thinking 透传
    if "extra_body" not in kwargs and body.get("extra_body"):
        kwargs["extra_body"] = body["extra_body"]

    # stream
    kwargs["stream"] = stream

    if not stream:
        # ====== 非流式 ======
        try:
            resp = client.chat.completions.create(**kwargs)
            data = resp.model_dump()
            data["model"] = model  # 替换为外部名称
            record_usage(api_key, model, True)
            return JSONResponse(content=data)
        except Exception as e:
            traceback.print_exc()
            record_usage(api_key, model, False)
            return _openai_error(500, str(e), "server_error")
    else:
        # ====== 流式 ======
        def stream_generator():
            success = False
            try:
                stream_resp = client.chat.completions.create(**kwargs)
                for chunk in stream_resp:
                    data = chunk.model_dump()
                    data["model"] = model
                    yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"
                success = True
            except Exception as e:
                traceback.print_exc()
                error_data = {"error": {"message": str(e), "type": "server_error"}}
                yield f"data: {json.dumps(error_data)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                record_usage(api_key, model, success)

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
