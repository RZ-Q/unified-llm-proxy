import os

from openai import OpenAI, AzureOpenAI

# ======================== 常量（请根据实际环境配置） ========================

# 主要后端 API Key 和地址
BACKEND_API_KEY = os.environ.get("BACKEND_API_KEY", "your-backend-api-key")
BACKEND_BASE_URL = os.environ.get("BACKEND_BASE_URL", "https://your-backend-url.com/v1/")

# Azure 配置（可选，若使用 Azure 后端）
AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT", "https://your-azure-endpoint.openai.azure.com/")
AZURE_API_KEY = os.environ.get("AZURE_API_KEY", "your-azure-api-key")
AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION", "2024-12-01-preview")

# 本地模型配置（可选，若使用本地 vLLM / Ollama 等）
LOCAL_BASE_URL = os.environ.get("LOCAL_BASE_URL", "http://localhost:8080/v1/")
LOCAL_API_KEY = os.environ.get("LOCAL_API_KEY", "not-needed")

# ======================== 模型注册表 ========================
# 外部名称 → (内部名称, 后端类型, 类别)
#
# 后端类型:
#   "openai-compatible" — OpenAI 兼容 API（如 OpenRouter、Together、DeepInfra 等）
#   "azure"             — Azure OpenAI
#   "local"             — 本地部署（vLLM / Ollama）
#   "genai"             — Google GenAI SDK（用于 Gemini 系列生图模型）
#   "image-gen"         — 自定义生图 API
#
# 类别:
#   "text-only"    — 纯文本模型
#   "multi-modal"  — 支持图片输入的模型
#   "image-gen"    — 图片生成模型

MODEL_REGISTRY = {
    # --- 纯文本模型（示例） ---
    "text-model-a":         ("text-model-a",         "openai-compatible", "text-only"),
    "text-model-b":         ("text-model-b",         "openai-compatible", "text-only"),
    "text-model-c":         ("text-model-c",         "openai-compatible", "text-only"),
    "local-model-a":        ("local-model-a",        "local",             "text-only"),
    "local-model-b":        ("local-model-b",        "local",             "text-only"),

    # --- 多模态模型（示例） ---
    "azure-vision-a":       ("azure-vision-a",       "azure",             "multi-modal"),
    "azure-vision-b":       ("azure-vision-b",       "azure",             "multi-modal"),
    "vision-model-a":       ("internal-vision-a",    "openai-compatible", "multi-modal"),
    "vision-model-b":       ("internal-vision-b",    "openai-compatible", "multi-modal"),
    "vision-model-c":       ("internal-vision-c",    "openai-compatible", "multi-modal"),
    "vision-model-d":       ("internal-vision-d",    "openai-compatible", "multi-modal"),

    # --- GenAI 生图模型（示例） ---
    "genai-image-a":        ("genai-image-a",        "genai",             "multi-modal"),

    # --- 自定义生图 API（示例） ---
    "image-gen-a":          ("image-gen-a",          "image-gen",         "image-gen"),
}


# ======================== 后端客户端工厂 ========================

# 本地模型的额外采样参数（可根据实际模型调整）
_LOCAL_EXTRA_PARAMS = {
    "extra_body": {
        "chat_template_kwargs": {"enable_thinking": False},
        "top_k": 20,
        "min_p": 0.0,
    },
    "presence_penalty": 1.5,
    "temperature": 0.7,
    "top_p": 0.8,
}


def get_backend_client(backend: str, internal_name: str):
    """
    返回 (openai_client, actual_model_name) 用于透明代理。
    仅支持 openai-compatible / azure / local 后端。
    """
    if backend == "openai-compatible":
        client = OpenAI(api_key=BACKEND_API_KEY, base_url=BACKEND_BASE_URL)
        return client, internal_name

    elif backend == "azure":
        client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            api_version=AZURE_API_VERSION,
        )
        # Azure 部署名称通常与模型名称相同，也可维护一份映射表
        return client, internal_name

    elif backend == "local":
        client = OpenAI(api_key=LOCAL_API_KEY, base_url=LOCAL_BASE_URL)
        return client, internal_name

    else:
        raise ValueError(f"Backend '{backend}' does not support OpenAI proxy")


def get_local_extra_params(internal_name: str) -> dict:
    """获取本地模型需要的额外参数（可按模型名称返回不同配置）"""
    return {**_LOCAL_EXTRA_PARAMS}
