"""
app/llm.py — Multi-provider LLM fallback chain for directive interpretation.

Chain: Gemini (native schema) → Groq (JSON mode) → HuggingFace (parser)
Each provider wrapped in interpret_with() returning List[dict] or raising.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ─── JSON extraction helper ───────────────────────────────────────────────────

def _extract_json_array(text: str) -> Optional[List[dict]]:
    """
    Try to extract the first JSON array from raw LLM text output.
    Handles markdown code fences and extra surrounding text.
    """
    # Remove markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()

    # Try direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Find first [...] in text
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    return None


# ─── Gemini provider ──────────────────────────────────────────────────────────

def _interpret_gemini(system_prompt: str, user_message: str, num_notes: int) -> List[dict]:
    """Call Gemini API with native response_schema for structured JSON output."""
    import google.generativeai as genai  # type: ignore

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")

    genai.configure(api_key=api_key)

    # Build response schema for directive array
    directive_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "note_index": {"type": "integer"},
                "applies": {"type": "boolean"},
                "directive_type": {
                    "type": "string",
                    "enum": [
                        "solar_reduction",
                        "minimum_battery_reserve",
                        "no_charge_window",
                        "no_discharge_window",
                        "max_grid_window",
                        "no_op",
                    ],
                },
                "structured_adjustment": {"type": "object", "nullable": True},
                "explanation": {"type": "string"},
            },
            "required": ["note_index", "applies", "directive_type", "explanation"],
        },
    }

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config=genai.GenerationConfig(
            temperature=0,
            response_mime_type="application/json",
        ),
        system_instruction=system_prompt,
    )

    response = model.generate_content(user_message)
    text = response.text.strip()
    parsed = _extract_json_array(text)
    if parsed is None:
        raise ValueError(f"Gemini returned non-array JSON: {text[:200]}")
    return parsed


# ─── Groq provider ────────────────────────────────────────────────────────────

def _interpret_groq(system_prompt: str, user_message: str, num_notes: int) -> List[dict]:
    """Call Groq API with JSON mode."""
    from groq import Groq, RateLimitError  # type: ignore

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")

    client = Groq(api_key=api_key)

    for model_id in ("openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"):
        try:
            chat = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=2048,
            )
            text = chat.choices[0].message.content or ""
            # Groq json_object mode returns an object; our array might be wrapped
            try:
                obj = json.loads(text)
                # Could be wrapped like {"directives": [...]}
                if isinstance(obj, list):
                    return obj
                for v in obj.values():
                    if isinstance(v, list):
                        return v
            except json.JSONDecodeError:
                pass
            parsed = _extract_json_array(text)
            if parsed is not None:
                return parsed
            logger.warning("Groq model %s returned unparseable output", model_id)
        except RateLimitError:
            logger.warning("Groq rate limit hit on %s, trying next model", model_id)
            continue
        except Exception as e:
            logger.warning("Groq model %s error: %s", model_id, e)
            continue

    raise RuntimeError("All Groq models failed")


# ─── HuggingFace provider ─────────────────────────────────────────────────────

def _interpret_hf(system_prompt: str, user_message: str, num_notes: int) -> List[dict]:
    """
    Call HuggingFace's current Inference Providers router (OpenAI-compatible
    chat completions). The legacy api-inference.huggingface.co serverless API
    was retired by HF in 2025 — router.huggingface.co is the replacement.

    Note: the HF token needs "Inference Providers" permission enabled
    (https://huggingface.co/settings/tokens) or every model call here returns
    a 401 permission error, which is caught and treated as provider failure.
    """
    import requests as req  # type: ignore

    token = os.environ.get("HF_API_TOKEN", "")
    if not token:
        raise ValueError("HF_API_TOKEN not set")

    hf_models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
    ]

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    last_error: Optional[Exception] = None

    for model_id in hf_models:
        try:
            resp = req.post(
                "https://router.huggingface.co/v1/chat/completions",
                headers=headers,
                json={
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    "temperature": 0,
                    "max_tokens": 1024,
                },
                timeout=60,
            )
            if resp.status_code == 429:
                logger.warning("HF rate limit for %s", model_id)
                continue
            if resp.status_code != 200:
                logger.warning("HF %s returned %s: %s", model_id, resp.status_code, resp.text[:300])
                continue

            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            parsed = _extract_json_array(text)
            if parsed is not None:
                return parsed
            logger.warning("HF %s output not parseable as JSON array", model_id)

        except Exception as e:
            logger.warning("HF %s error: %s", model_id, e)
            last_error = e
            continue

    raise RuntimeError(f"All HF models failed. Last error: {last_error}")


# ─── Main interface ───────────────────────────────────────────────────────────

PROVIDERS = [
    ("gemini", _interpret_gemini),
    ("groq", _interpret_groq),
    ("huggingface", _interpret_hf),
]


def call_llm_chain(
    system_prompt: str,
    user_message: str,
    num_notes: int,
) -> List[dict]:
    """
    Try each provider in order. Returns raw list[dict] from first successful provider.
    Raises RuntimeError if all providers fail.
    """
    last_error: Optional[Exception] = None

    for name, fn in PROVIDERS:
        try:
            logger.info("Trying LLM provider: %s", name)
            result = fn(system_prompt, user_message, num_notes)
            logger.info("Provider %s succeeded", name)
            return result
        except ValueError as e:
            # Config error (key missing) — skip provider
            logger.info("Skipping %s: %s", name, e)
            continue
        except Exception as e:
            logger.warning("Provider %s failed: %s", name, e)
            last_error = e
            continue

    raise RuntimeError(
        f"All LLM providers failed. Last error: {last_error}"
    )
