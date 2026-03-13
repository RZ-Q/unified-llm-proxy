import json
import os
import sqlite3
import threading
from datetime import datetime
from typing import Dict, Optional

# ======================== API Key 管理 ========================

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_KEYS_FILE = os.path.join(_BASE_DIR, "api_keys.json")
DB_PATH = os.path.join(_BASE_DIR, "usage.db")

def load_api_keys() -> set:
    with open(API_KEYS_FILE, "r") as f:
        return set(json.load(f))

API_KEYS = load_api_keys()

# ======================== 用量统计（SQLite） ========================

_db_lock = threading.Lock()

def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            api_key TEXT NOT NULL,
            model TEXT NOT NULL,
            success INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_usage_api_key ON usage(api_key)")
    conn.commit()
    conn.close()

_init_db()

def record_usage(api_key: str, model: str, success: bool):
    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO usage (api_key, model, success, created_at) VALUES (?, ?, ?, ?)",
            (api_key, model, int(success), datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()

def query_usage(api_key: str) -> Dict[str, Dict[str, int]]:
    """返回 {model: {total: N, success: N, fail: N}}"""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT model, success, COUNT(*) FROM usage WHERE api_key = ? GROUP BY model, success",
        (api_key,),
    ).fetchall()
    conn.close()

    stats: Dict[str, Dict[str, int]] = {}
    for model, success, count in rows:
        if model not in stats:
            stats[model] = {"total": 0, "success": 0, "fail": 0}
        stats[model]["total"] += count
        if success:
            stats[model]["success"] += count
        else:
            stats[model]["fail"] += count
    return stats


def query_usage_all() -> Dict[str, Dict[str, Dict[str, int]]]:
    """返回 {api_key: {model: {total, success, fail}}}"""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT api_key, model, success, COUNT(*) FROM usage GROUP BY api_key, model, success",
    ).fetchall()
    conn.close()

    stats: Dict[str, Dict[str, Dict[str, int]]] = {}
    for api_key, model, success, count in rows:
        if api_key not in stats:
            stats[api_key] = {}
        if model not in stats[api_key]:
            stats[api_key][model] = {"total": 0, "success": 0, "fail": 0}
        stats[api_key][model]["total"] += count
        if success:
            stats[api_key][model]["success"] += count
        else:
            stats[api_key][model]["fail"] += count
    return stats


def verify_api_key(authorization: str) -> Optional[str]:
    """验证 API Key，返回 key 或 None"""
    if not authorization:
        return None
    api_key = authorization.removeprefix("Bearer ").strip()
    if api_key in API_KEYS:
        return api_key
    return None


def verify_api_key_any(authorization: str = "", x_api_key: str = "") -> Optional[str]:
    """验证 API Key（支持 Bearer 和 x-api-key 两种方式），返回 key 或 None"""
    if x_api_key and x_api_key in API_KEYS:
        return x_api_key
    return verify_api_key(authorization)
