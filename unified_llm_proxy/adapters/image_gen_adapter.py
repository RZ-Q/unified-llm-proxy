"""
Image Generation 适配器：将 OpenAI messages 格式转换为生图 API 调用，
结果包装成 OpenAI ChatCompletion 响应格式。
"""

import json
import time
import traceback
import uuid

import requests as _requests
from fastapi.responses import JSONResponse

from ..auth import record_usage
from ..registry import BACKEND_API_KEY, BACKEND_BASE_URL

# 生图 API 端点（根据实际后端配置）
IMAGE_GEN_URL = BACKEND_BASE_URL.rstrip("/").removesuffix("/v1") + "/v1/services/aigc/multimodal-generation/generation"


def _extract_prompt(messages: list) -> str:
    """从 OpenAI messages 中提取最后一条 user 消息的文本"""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            texts = []
            for part in content:
                if part.get("type") == "text":
                    texts.append(part.get("text", ""))
            return "\n".join(texts)
    return ""


def _build_response(model: str, image_urls: list):
    """将图片 URL 列表包装为 OpenAI ChatCompletion 格式"""
    if image_urls:
        lines = [f"![image]({url})" for url in image_urls]
        content = "\n\n".join(lines)
    else:
        content = "[Image generation failed: no images returned]"

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def handle_image_gen_request(body: dict, model: str, internal_name: str, api_key: str):
    """处理生图模型的请求（不支持流式）"""
    messages = body.get("messages", [])
    prompt = _extract_prompt(messages)

    if not prompt:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "No prompt found in messages", "type": "invalid_request_error", "code": None}},
        )

    size = body.get("size", "1280*1280")
    n = body.get("n", 1)

    headers = {
        "Authorization": f"Bearer {BACKEND_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": internal_name,
        "input": {
            "messages": [
                {"role": "user", "content": [{"text": prompt}]}
            ]
        },
        "parameters": {
            "negative_prompt": "",
            "prompt_extend": True,
            "watermark": False,
            "n": n,
            "size": size,
        },
    }

    try:
        resp = _requests.post(IMAGE_GEN_URL, headers=headers, json=payload, timeout=180)
        resp.raise_for_status()
        data = resp.json()

        choices = data.get("output", {}).get("choices", [])
        image_urls = []
        for choice in choices:
            content_parts = choice.get("message", {}).get("content", [])
            for part in content_parts:
                if part.get("type") == "image" and part.get("image"):
                    image_urls.append(part["image"])

        record_usage(api_key, model, bool(image_urls))
        return JSONResponse(content=_build_response(model, image_urls))

    except Exception as e:
        traceback.print_exc()
        record_usage(api_key, model, False)
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "server_error", "code": None}},
        )
