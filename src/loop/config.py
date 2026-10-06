import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
DB_PATH = DATA_DIR / "loop.duckdb"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
LLM_CACHE_DIR = DATA_DIR / "cache" / "llm"
REPORTS_DIR = ROOT_DIR / "reports"
EVAL_GOLD_DIR = ROOT_DIR / "eval" / "gold"

def get_env_str(name: str, default: str | None = None) -> str | None:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    return val.strip()

def get_env_float(name: str) -> float | None:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return None
    try:
        return float(val.strip())
    except ValueError:
        return None

def get_env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return default
    try:
        return int(val.strip())
    except ValueError:
        return default

LOOP_SEED: int = get_env_int("LOOP_SEED", 7)
GEMINI_API_KEY: str | None = get_env_str("GEMINI_API_KEY") or get_env_str("GOOGLE_API_KEY")
GOOGLE_API_KEY: str | None = GEMINI_API_KEY
ANTHROPIC_API_KEY: str | None = get_env_str("ANTHROPIC_API_KEY")
LOOP_MODEL: str = get_env_str("LOOP_MODEL", "gemini-2.5-flash") or "gemini-2.5-flash"
LOOP_PRICE_IN_PER_MTOK: float | None = get_env_float("LOOP_PRICE_IN_PER_MTOK")
LOOP_PRICE_OUT_PER_MTOK: float | None = get_env_float("LOOP_PRICE_OUT_PER_MTOK")

def ensure_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_GOLD_DIR.mkdir(parents=True, exist_ok=True)
