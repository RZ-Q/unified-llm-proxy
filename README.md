# Unified LLM Proxy

A lightweight FastAPI proxy that unifies multiple LLM backends (OpenAI-compatible, Azure, local deployments, Google GenAI) behind a single API, supporting **both OpenAI Chat Completions and Anthropic Messages protocols**.

Works seamlessly with the `openai` SDK, `anthropic` SDK, and **Claude Code**.

## Features

- **Dual Protocol Support** — `/v1/chat/completions` (OpenAI) and `/v1/messages` (Anthropic)
- **Multi-Backend Routing** — Route requests to different LLM providers based on model name
- **Tool Calling** — Full Anthropic tool_use/tool_result support, including streaming
- **Anthropic ↔ OpenAI Conversion** — Automatic bidirectional format conversion for non-native backends
- **Native Passthrough** — Zero-conversion forwarding for backends that natively support the requested protocol
- **Streaming** — SSE streaming for both OpenAI and Anthropic formats
- **API Key Management** — Hide real backend credentials behind your own API key layer
- **Usage Tracking** — Per-key, per-model call statistics stored in SQLite
- **Image Generation** — Support for text-to-image and image-to-image via adapters

## Architecture

```
Client Request → Auth → Protocol Router
                        ├─ /v1/chat/completions (OpenAI)  ─┐
                        ├─ /v1/messages (Anthropic)  ───────┤
                        ├─ /v1/models                       │
                        ├─ /v1/usage                        │
                        └─ /v1/usage/all                    │
                                                            ↓
                                                      Model Router
                          ├─ openai-compatible → Transparent proxy (or Anthropic passthrough)
                          ├─ azure → Azure OpenAI proxy
                          ├─ local → Local model proxy (vLLM / Ollama)
                          ├─ genai → Google GenAI SDK adapter
                          └─ image-gen → Image generation adapter
```

## Quick Start

```bash
# 1. Clone
git clone https://github.com/RZ-Q/unified-llm-proxy.git
cd unified-llm-proxy

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure API keys
cp api_keys.json.example api_keys.json
# Edit api_keys.json with your client-facing API keys

# 4. Configure backend (edit registry.py or use environment variables)
export BACKEND_API_KEY="your-real-backend-api-key"
export BACKEND_BASE_URL="https://your-backend-url.com/v1/"

# 5. Start
uvicorn unified_llm_proxy.server:app --host 0.0.0.0 --port 9000
```

Swagger docs at `http://localhost:9000/docs`.

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BACKEND_API_KEY` | API key for the primary backend | `your-backend-api-key` |
| `BACKEND_BASE_URL` | Base URL for the primary backend | `https://your-backend-url.com/v1/` |
| `AZURE_ENDPOINT` | Azure OpenAI endpoint | (optional) |
| `AZURE_API_KEY` | Azure OpenAI API key | (optional) |
| `AZURE_API_VERSION` | Azure API version | `2024-12-01-preview` |
| `LOCAL_BASE_URL` | Local model endpoint (vLLM/Ollama) | `http://localhost:8080/v1/` |
| `LOCAL_API_KEY` | Local model API key | `not-needed` |

### Model Registry

Edit `unified_llm_proxy/registry.py` to configure your models:

```python
MODEL_REGISTRY = {
    # external_name → (internal_name, backend_type, category)
    "my-model":  ("actual-model-name", "openai-compatible", "text-only"),
    "my-local":  ("local-model-name",  "local",             "text-only"),
    "my-azure":  ("deployment-name",   "azure",             "multi-modal"),
}
```

Backend types:
- `openai-compatible` — Any OpenAI-compatible API (OpenRouter, Together, DeepInfra, etc.)
- `azure` — Azure OpenAI
- `local` — Local deployment (vLLM, Ollama, etc.)
- `genai` — Google GenAI SDK
- `image-gen` — Custom image generation API

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | OpenAI Chat Completions API |
| POST | `/v1/messages` | Anthropic Messages API |
| GET | `/v1/models` | List available models |
| GET | `/v1/usage` | Usage stats for current API key |
| GET | `/v1/usage/all` | Usage stats for all API keys |

### Authentication

```bash
# OpenAI style
curl -H "Authorization: Bearer <your-api-key>" ...

# Anthropic style
curl -H "x-api-key: <your-api-key>" ...
```

## SDK Usage

### OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(api_key="<your-api-key>", base_url="http://localhost:9000/v1")
resp = client.chat.completions.create(
    model="text-model-a",
    messages=[{"role": "user", "content": "Hello"}],
)
print(resp.choices[0].message.content)
```

### Anthropic SDK

```python
import anthropic

client = anthropic.Anthropic(api_key="<your-api-key>", base_url="http://localhost:9000")
resp = client.messages.create(
    model="text-model-a",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
)
print(resp.content[0].text)
```

### Claude Code

Add to `~/.claude.json`:

```json
{
  "env": {
    "ANTHROPIC_AUTH_TOKEN": "<your-api-key>",
    "ANTHROPIC_BASE_URL": "http://localhost:9000",
    "ANTHROPIC_MODEL": "text-model-a"
  }
}
```

## License

MIT
