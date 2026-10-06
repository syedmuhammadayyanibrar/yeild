import hashlib
import json
import time
from typing import Any

from loop.config import (
    ANTHROPIC_API_KEY,
    GEMINI_API_KEY,
    GOOGLE_API_KEY,
    LLM_CACHE_DIR,
    LOOP_MODEL,
)


def compute_cache_key(
    model: str,
    system_prompt: str,
    tool_schema: dict[str, Any] | None,
    user_content: str,
) -> str:
    payload = {
        "model": model,
        "system_prompt": system_prompt,
        "tool_schema": tool_schema,
        "user_content": user_content,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

class LlmClient:
    def __init__(self) -> None:
        self.cache_dir = LLM_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.google_key = GEMINI_API_KEY or GOOGLE_API_KEY
        self.anthropic_key = ANTHROPIC_API_KEY
        self.model = LOOP_MODEL

    def is_available(self) -> bool:
        return bool(self.google_key or self.anthropic_key)

    def call_cached(
        self,
        system_prompt: str,
        user_content: str,
        tool_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not self.is_available():
            return None

        key = compute_cache_key(self.model, system_prompt, tool_schema, user_content)
        cache_file = self.cache_dir / f"{key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data
            except Exception:
                pass

        res = self._execute_call(system_prompt, user_content, tool_schema)
        if res is not None:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
        return res

    def _execute_call(
        self,
        system_prompt: str,
        user_content: str,
        tool_schema: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        start_time = time.perf_counter()

        if self.google_key:
            return self._call_gemini(system_prompt, user_content, tool_schema, start_time)
        elif self.anthropic_key:
            return self._call_anthropic(system_prompt, user_content, tool_schema, start_time)
        return None

    def _call_gemini(
        self,
        system_prompt: str,
        user_content: str,
        tool_schema: dict[str, Any] | None,
        start_time: float,
    ) -> dict[str, Any] | None:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.google_key)

        for attempt in range(4):
            try:
                config_kwargs: dict[str, Any] = {
                    "system_instruction": system_prompt,
                    "temperature": 0.0,
                }
                if tool_schema:
                    config_kwargs["response_mime_type"] = "application/json"

                response = client.models.generate_content(
                    model=self.model,
                    contents=user_content,
                    config=types.GenerateContentConfig(**config_kwargs),
                )

                latency_ms = (time.perf_counter() - start_time) * 1000.0
                usage = getattr(response, "usage_metadata", None)
                in_tokens = getattr(usage, "prompt_token_count", 0) or 0
                out_tokens = getattr(usage, "candidates_token_count", 0) or 0

                raw_text = response.text or ""
                structured: dict[str, Any] | None = None
                if tool_schema:
                    try:
                        structured = json.loads(raw_text)
                    except Exception:
                        pass

                return {
                    "model": self.model,
                    "raw_text": raw_text,
                    "structured": structured,
                    "input_tokens": in_tokens,
                    "output_tokens": out_tokens,
                    "latency_ms": latency_ms,
                }
            except Exception:
                if attempt == 3:
                    return None
                time.sleep(2**attempt)
        return None

    def _call_anthropic(
        self,
        system_prompt: str,
        user_content: str,
        tool_schema: dict[str, Any] | None,
        start_time: float,
    ) -> dict[str, Any] | None:
        import anthropic

        client = anthropic.Anthropic(api_key=self.anthropic_key)

        tools: list[Any] = []
        tool_choice: dict[str, Any] | None = None
        if tool_schema:
            tools = [
                {
                    "name": tool_schema["name"],
                    "description": tool_schema.get("description", ""),
                    "input_schema": tool_schema["parameters"],
                }
            ]
            tool_choice = {"type": "tool", "name": tool_schema["name"]}

        for attempt in range(4):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "max_tokens": 1024,
                    "temperature": 0.0,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_content}],
                }
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = tool_choice

                response = client.messages.create(**kwargs)
                latency_ms = (time.perf_counter() - start_time) * 1000.0

                in_tokens = response.usage.input_tokens
                out_tokens = response.usage.output_tokens

                structured = None
                raw_text = ""
                for blk in response.content:
                    if getattr(blk, "type", "") == "tool_use":
                        structured = blk.input
                    elif getattr(blk, "type", "") == "text":
                        raw_text = blk.text

                return {
                    "model": self.model,
                    "raw_text": raw_text,
                    "structured": structured,
                    "input_tokens": in_tokens,
                    "output_tokens": out_tokens,
                    "latency_ms": latency_ms,
                }
            except Exception:
                if attempt == 3:
                    return None
                time.sleep(2**attempt)
        return None
