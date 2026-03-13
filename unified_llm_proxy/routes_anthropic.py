"""
Anthropic Messages API 兼容层
POST /v1/messages — 接收 Anthropic 格式请求，转换为 OpenAI 格式调后端，响应转回 Anthropic 格式。
用于 Claude Code 等使用 Anthropic SDK 的客户端。

关键能力：
- 消息格式双向转换（Anthropic <-> OpenAI）
- 工具调用（tool_use / tool_calls）双向转换
- 流式 SSE 事件（包括 tool_use 流式）
- 对支持 Anthropic 原生协议的后端做直接透传（零格式转换）
"""

import json
import time
import traceback
import uuid

import httpx
from fastapi import APIRouter, Request, Header
from fastapi.responses import StreamingResponse, JSONResponse

from .auth import verify_api_key, record_usage
from .registry import (
    MODEL_REGISTRY,
    BACKEND_API_KEY,
    BACKEND_BASE_URL,
    get_backend_client,
    get_local_extra_params,
)
from .adapters.genai_adapter import handle_genai_request
from .adapters.image_gen_adapter import handle_image_gen_request

# 后端 Anthropic Messages 端点（若后端原生支持 Anthropic 协议，则直接透传）
# 格式示例："https://your-backend-url.com/v1/messages"
BACKEND_MESSAGES_URL = BACKEND_BASE_URL.rstrip("/").removesuffix("/v1") + "/v1/messages"

router = APIRouter()

_NO_TEMPERATURE_MODELS = set()


# ======================== 工具格式转换 ========================

def _anthropic_tools_to_openai(tools: list) -> list:
    """Anthropic tools -> OpenAI tools"""
    openai_tools = []
    for tool in tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        })
    return openai_tools


def _anthropic_tool_choice_to_openai(tool_choice: dict) -> str | dict:
    """Anthropic tool_choice -> OpenAI tool_choice"""
    if not tool_choice:
        return "auto"
    tc_type = tool_choice.get("type", "auto")
    if tc_type == "auto":
        return "auto"
    elif tc_type == "any":
        return "required"
    elif tc_type == "tool":
        return {"type": "function", "function": {"name": tool_choice.get("name", "")}}
    elif tc_type == "none":
        return "none"
    return "auto"


# ======================== 消息格式转换 ========================

def _anthropic_to_openai_messages(messages: list, system=None) -> list:
    """将 Anthropic messages 格式转为 OpenAI messages 格式（含 tool_use/tool_result）"""
    openai_msgs = []

    # system
    if system:
        if isinstance(system, str):
            openai_msgs.append({"role": "system", "content": system})
        elif isinstance(system, list):
            texts = [p.get("text", "") for p in system if p.get("type") == "text"]
            openai_msgs.append({"role": "system", "content": "\n".join(texts)})

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # 纯文本
        if isinstance(content, str):
            openai_msgs.append({"role": role, "content": content})
            continue

        # content 是数组
        if role == "assistant":
            text_parts = []
            tool_calls = []
            for block in content:
                bt = block.get("type", "")
                if bt == "text":
                    text_parts.append(block.get("text", ""))
                elif bt == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    })
                elif bt == "thinking":
                    pass

            msg_dict = {"role": "assistant"}
            msg_dict["content"] = "\n".join(text_parts) if text_parts else None
            if tool_calls:
                msg_dict["tool_calls"] = tool_calls
            openai_msgs.append(msg_dict)

        elif role == "user":
            openai_parts = []
            tool_results = []
            for block in content:
                bt = block.get("type", "")
                if bt == "text":
                    openai_parts.append({"type": "text", "text": block.get("text", "")})
                elif bt == "image":
                    source = block.get("source", {})
                    if source.get("type") == "base64":
                        mt = source.get("media_type", "image/png")
                        d = source.get("data", "")
                        openai_parts.append({"type": "image_url", "image_url": {"url": f"data:{mt};base64,{d}"}})
                    elif source.get("type") == "url":
                        openai_parts.append({"type": "image_url", "image_url": {"url": source.get("url", "")}})
                elif bt == "tool_result":
                    tc = block.get("content", "")
                    if isinstance(tc, list):
                        tc = "\n".join(p.get("text", "") for p in tc if p.get("type") == "text")
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": str(tc),
                    })

            if openai_parts:
                if len(openai_parts) == 1 and openai_parts[0].get("type") == "text":
                    openai_msgs.append({"role": "user", "content": openai_parts[0]["text"]})
                else:
                    openai_msgs.append({"role": "user", "content": openai_parts})
            for tr in tool_results:
                openai_msgs.append(tr)
        else:
            openai_msgs.append({"role": role, "content": str(content)})

    return openai_msgs


# ======================== 响应格式转换 ========================

def _openai_resp_to_anthropic(openai_data: dict, model: str) -> dict:
    """将 OpenAI ChatCompletion 响应转为 Anthropic Messages 响应（含 tool_calls -> tool_use）"""
    choice = openai_data.get("choices", [{}])[0]
    message = choice.get("message", {})
    content_str = message.get("content") or message.get("reasoning_content") or ""
    finish = choice.get("finish_reason", "stop")
    tool_calls = message.get("tool_calls") or []

    content_blocks = []
    if content_str:
        content_blocks.append({"type": "text", "text": content_str})

    for tc in tool_calls:
        func = tc.get("function", {})
        try:
            input_data = json.loads(func.get("arguments", "{}"))
        except json.JSONDecodeError:
            input_data = {"raw": func.get("arguments", "")}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
            "name": func.get("name", ""),
            "input": input_data,
        })

    if tool_calls or finish == "tool_calls":
        stop_reason = "tool_use"
    else:
        stop_reason_map = {"stop": "end_turn", "length": "max_tokens", "content_filter": "end_turn"}
        stop_reason = stop_reason_map.get(finish, "end_turn")

    usage = openai_data.get("usage", {})
    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def _anthropic_error(status_code: int, message: str, error_type: str = "invalid_request_error"):
    return JSONResponse(
        status_code=status_code,
        content={"type": "error", "error": {"type": error_type, "message": message}},
    )


# ======================== 流式 Anthropic SSE ========================

def _stream_anthropic(client, kwargs, model: str, api_key: str):
    """生成 Anthropic 流式 SSE 事件（支持 text + tool_use）"""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    success = False

    try:
        start_event = {
            "type": "message_start",
            "message": {
                "id": msg_id, "type": "message", "role": "assistant",
                "content": [], "model": model,
                "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }
        yield f"event: message_start\ndata: {json.dumps(start_event)}\n\n"

        kwargs["stream"] = True
        stream_resp = client.chat.completions.create(**kwargs)

        output_tokens = 0
        block_index = 0
        text_block_started = False
        has_tool_calls = False
        tool_call_accum = {}

        for chunk in stream_resp:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue

            # --- 文本 content ---
            text = delta.content
            if not text and hasattr(delta, "reasoning_content"):
                text = delta.reasoning_content
            if text:
                if not text_block_started:
                    block_start = {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {"type": "text", "text": ""},
                    }
                    yield f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n"
                    text_block_started = True

                block_delta = {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "text_delta", "text": text},
                }
                yield f"event: content_block_delta\ndata: {json.dumps(block_delta)}\n\n"
                output_tokens += 1

            # --- tool_calls ---
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                has_tool_calls = True

                if text_block_started:
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': block_index})}\n\n"
                    block_index += 1
                    text_block_started = False

                for tc_delta in delta.tool_calls:
                    tc_idx = tc_delta.index
                    func = tc_delta.function

                    if tc_idx not in tool_call_accum:
                        tool_call_accum[tc_idx] = {
                            "id": tc_delta.id or f"toolu_{uuid.uuid4().hex[:12]}",
                            "name": func.name if func and func.name else "",
                            "arguments": "",
                            "block_index": block_index,
                        }
                        tool_block = {
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": tool_call_accum[tc_idx]["id"],
                                "name": tool_call_accum[tc_idx]["name"],
                                "input": {},
                            },
                        }
                        yield f"event: content_block_start\ndata: {json.dumps(tool_block)}\n\n"
                        block_index += 1

                    if func and func.arguments:
                        tool_call_accum[tc_idx]["arguments"] += func.arguments
                        input_delta = {
                            "type": "content_block_delta",
                            "index": tool_call_accum[tc_idx]["block_index"],
                            "delta": {"type": "input_json_delta", "partial_json": func.arguments},
                        }
                        yield f"event: content_block_delta\ndata: {json.dumps(input_delta)}\n\n"

            if hasattr(chunk, "usage") and chunk.usage:
                if hasattr(chunk.usage, "completion_tokens") and chunk.usage.completion_tokens:
                    output_tokens = chunk.usage.completion_tokens

        if text_block_started:
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

        for tc_idx in sorted(tool_call_accum.keys()):
            bi = tool_call_accum[tc_idx]["block_index"]
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': bi})}\n\n"

        stop_reason = "tool_use" if has_tool_calls else "end_turn"
        msg_delta = {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        }
        yield f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n"
        yield f"event: message_stop\ndata: {{\"type\": \"message_stop\"}}\n\n"
        success = True

    except Exception as e:
        traceback.print_exc()
        error_event = {"type": "error", "error": {"type": "server_error", "message": str(e)}}
        yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
    finally:
        record_usage(api_key, model, success)


# ======================== 后端 Anthropic 直接透传 ========================

async def _forward_to_backend_anthropic(body: dict, model: str, internal_name: str, api_key: str, stream: bool):
    """
    将 Anthropic 请求直接转发到后端 /v1/messages 端点（要求后端原生支持 Anthropic 协议）。
    仅替换 model 名称和 API Key，其余完整透传（包括 tools、tool_use、tool_result）。
    """
    forward_body = {**body, "model": internal_name}
    headers = {
        "x-api-key": BACKEND_API_KEY,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    if not stream:
        try:
            async with httpx.AsyncClient(timeout=300) as client:
                resp = await client.post(BACKEND_MESSAGES_URL, headers=headers, json=forward_body)
            data = resp.json()
            if resp.status_code != 200:
                record_usage(api_key, model, False)
                err_msg = data.get("error", {}).get("message", str(data))
                return _anthropic_error(resp.status_code, err_msg, "server_error")
            data["model"] = model
            record_usage(api_key, model, True)
            return JSONResponse(content=data)
        except Exception as e:
            traceback.print_exc()
            record_usage(api_key, model, False)
            return _anthropic_error(500, str(e), "server_error")
    else:
        forward_body["stream"] = True

        def stream_generator():
            success = False
            try:
                import requests as _requests
                resp = _requests.post(
                    BACKEND_MESSAGES_URL, headers=headers, json=forward_body,
                    stream=True, timeout=300,
                )
                if resp.status_code != 200:
                    error_data = resp.json()
                    err_msg = error_data.get("error", {}).get("message", str(error_data))
                    error_event = {"type": "error", "error": {"type": "server_error", "message": err_msg}}
                    yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
                    return

                for line in resp.iter_lines():
                    if not line:
                        continue
                    decoded = line.decode("utf-8")
                    if decoded.startswith("data: ") and internal_name in decoded:
                        decoded = decoded.replace(f'"model":"{internal_name}"', f'"model":"{model}"')
                        decoded = decoded.replace(f'"model": "{internal_name}"', f'"model": "{model}"')
                    yield decoded + "\n"
                    if decoded.startswith("data: "):
                        yield "\n"
                success = True
            except Exception as e:
                traceback.print_exc()
                error_event = {"type": "error", "error": {"type": "server_error", "message": str(e)}}
                yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
            finally:
                record_usage(api_key, model, success)

        return StreamingResponse(stream_generator(), media_type="text/event-stream")


# ======================== 路由 ========================

@router.post("/v1/messages")
async def messages(
    request: Request,
    authorization: str = Header(default="", alias="Authorization"),
    x_api_key: str = Header(default="", alias="x-api-key"),
):
    # 1. 鉴权
    api_key = verify_api_key(x_api_key) if x_api_key else None
    if not api_key:
        api_key = verify_api_key(authorization)
    if not api_key:
        return _anthropic_error(401, "Invalid API key", "authentication_error")

    # 2. 解析请求
    body = await request.json()
    model = body.get("model", "")
    anthropic_messages = body.get("messages", [])
    system = body.get("system", None)
    stream = body.get("stream", False)
    max_tokens = body.get("max_tokens", 4096)
    tools = body.get("tools", [])
    tool_choice = body.get("tool_choice", None)

    if model not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        return _anthropic_error(404, f"Model '{model}' not found. Available: {available}")

    internal_name, backend, category = MODEL_REGISTRY[model]

    # 3. 若后端原生支持 Anthropic 协议 → 直接透传，不做任何格式转换
    #    tool_use/tool_result 等复杂消息结构完整保留，避免双重转换导致丢失
    #    注意：需要后端 URL 支持 /v1/messages 端点
    if backend == "openai-compatible":
        return await _forward_to_backend_anthropic(body, model, internal_name, api_key, stream)

    # 4. 其他后端需要转换为 OpenAI 格式
    openai_messages = _anthropic_to_openai_messages(anthropic_messages, system)

    if backend == "genai":
        openai_body = {"messages": openai_messages, "model": model}
        resp = await handle_genai_request(openai_body, model, internal_name, api_key, False)
        openai_data = json.loads(resp.body.decode())
        if "error" in openai_data:
            return _anthropic_error(500, openai_data["error"]["message"], "server_error")
        return JSONResponse(content=_openai_resp_to_anthropic(openai_data, model))

    if backend == "image-gen":
        openai_body = {"messages": openai_messages, "model": model}
        resp = await handle_image_gen_request(openai_body, model, internal_name, api_key)
        openai_data = json.loads(resp.body.decode())
        if "error" in openai_data:
            return _anthropic_error(500, openai_data["error"]["message"], "server_error")
        return JSONResponse(content=_openai_resp_to_anthropic(openai_data, model))

    # 5. 其他透明代理后端（azure / local）→ 转 OpenAI 格式
    try:
        client, actual_model = get_backend_client(backend, internal_name)
    except Exception as e:
        record_usage(api_key, model, False)
        return _anthropic_error(500, f"Backend error: {e}", "server_error")

    kwargs = {
        "model": actual_model,
        "messages": openai_messages,
        "max_tokens": max_tokens,
    }

    if tools:
        kwargs["tools"] = _anthropic_tools_to_openai(tools)
        kwargs["tool_choice"] = _anthropic_tool_choice_to_openai(tool_choice)

    if "temperature" in body and internal_name not in _NO_TEMPERATURE_MODELS:
        kwargs["temperature"] = body["temperature"]
    elif internal_name not in _NO_TEMPERATURE_MODELS:
        kwargs["temperature"] = 0.7

    if "top_p" in body:
        kwargs["top_p"] = body["top_p"]

    if backend == "local":
        extra = get_local_extra_params(internal_name)
        kwargs["extra_body"] = extra.pop("extra_body")
        kwargs.setdefault("temperature", extra["temperature"])
        kwargs.setdefault("top_p", extra["top_p"])
        kwargs["presence_penalty"] = extra["presence_penalty"]

    if stream:
        return StreamingResponse(
            _stream_anthropic(client, kwargs, model, api_key),
            media_type="text/event-stream",
        )
    else:
        try:
            kwargs["stream"] = False
            resp = client.chat.completions.create(**kwargs)
            openai_data = resp.model_dump()
            anthropic_resp = _openai_resp_to_anthropic(openai_data, model)
            record_usage(api_key, model, True)
            return JSONResponse(content=anthropic_resp)
        except Exception as e:
            traceback.print_exc()
            record_usage(api_key, model, False)
            return _anthropic_error(500, str(e), "server_error")
