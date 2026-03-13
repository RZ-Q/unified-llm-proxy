"""
GenAI 适配器：将 OpenAI messages 格式转换为 Google GenAI SDK 调用，
结果包装成 OpenAI ChatCompletion 响应格式。
适用于通过 Google GenAI SDK 调用的模型（如 Gemini 系列生图模型）。
"""

import base64
import io
import json
import os
import time
import traceback
import uuid

from fastapi.responses import StreamingResponse, JSONResponse

from ..auth import record_usage
from ..registry import BACKEND_API_KEY, BACKEND_BASE_URL


def _extract_from_messages(messages: list) -> tuple:
    """从 OpenAI messages 中提取最后一条 user 消息的文本和图片"""
    text_parts = []
    image_data_list = []

    last_user_msg = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_msg = msg
            break

    if not last_user_msg:
        return "", []

    content = last_user_msg.get("content", "")
    if isinstance(content, str):
        return content, []

    for part in content:
        if part.get("type") == "text":
            text_parts.append(part.get("text", ""))
        elif part.get("type") == "image_url":
            url = part.get("image_url", {}).get("url", "")
            if url.startswith("data:"):
                header, b64data = url.split(",", 1)
                mime = header.split(":")[1].split(";")[0]
                image_data_list.append(("base64", b64data, mime))
            elif url.startswith(("http://", "https://")):
                image_data_list.append(("url", url, "image/jpeg"))

    return "\n".join(text_parts), image_data_list


def _build_openai_response(model: str, text: str = None, images_b64: list = None):
    """构建 OpenAI ChatCompletion 格式的响应"""
    content_parts = []
    if text:
        content_parts.append(text)
    if images_b64:
        for b64, mime in images_b64:
            content_parts.append(f"![image](data:{mime};base64,{b64})")

    content = "\n\n".join(content_parts) if content_parts else ""

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


def _build_stream_events(model: str, text: str = None, images_b64: list = None):
    """生成流式 SSE 事件"""
    msg_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    content_parts = []
    if text:
        content_parts.append(text)
    if images_b64:
        for b64, mime in images_b64:
            content_parts.append(f"![image](data:{mime};base64,{b64})")
    full_content = "\n\n".join(content_parts) if content_parts else ""

    chunk = {
        "id": msg_id, "object": "chat.completion.chunk", "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(chunk)}\n\n"

    chunk = {
        "id": msg_id, "object": "chat.completion.chunk", "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"content": full_content}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(chunk)}\n\n"

    chunk = {
        "id": msg_id, "object": "chat.completion.chunk", "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


# GenAI 后端 URL（去掉 /v1/ 后缀）
_GENAI_BASE_URL = BACKEND_BASE_URL.rstrip("/").removesuffix("/v1").removesuffix("/")


async def handle_genai_request(body: dict, model: str, internal_name: str, api_key: str, stream: bool):
    """处理 GenAI 模型的请求"""
    from google import genai
    from google.genai import types
    from PIL import Image

    messages = body.get("messages", [])
    prompt_text, image_data_list = _extract_from_messages(messages)

    contents = [prompt_text]
    for img_type, data, mime in image_data_list:
        if img_type == "base64":
            raw = base64.b64decode(data)
            contents.append(types.Part.from_bytes(data=raw, mime_type=mime))
        elif img_type == "url":
            contents.append(types.Part.from_uri(file_uri=data, mime_type=mime))

    client = genai.Client(
        api_key=BACKEND_API_KEY,
        http_options=types.HttpOptions(base_url=_GENAI_BASE_URL, api_version="v1"),
    )

    try:
        response = client.models.generate_content(model=internal_name, contents=contents)

        text_parts = []
        images_b64 = []

        if response.parts is None:
            record_usage(api_key, model, False)
            resp = _build_openai_response(model, text="[Model returned empty response]")
            return JSONResponse(content=resp)

        for part in response.parts:
            if part.text is not None:
                text_parts.append(part.text)
            elif part.inline_data is not None:
                b64 = base64.b64encode(part.inline_data.data).decode()
                mime = part.inline_data.mime_type or "image/png"
                images_b64.append((b64, mime))

        text = "\n".join(text_parts) if text_parts else None
        record_usage(api_key, model, True)

        if stream:
            return StreamingResponse(
                _build_stream_events(model, text, images_b64),
                media_type="text/event-stream",
            )
        else:
            return JSONResponse(content=_build_openai_response(model, text, images_b64))

    except Exception as e:
        traceback.print_exc()
        record_usage(api_key, model, False)
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "server_error", "code": None}},
        )
