"""
llm_fallback.py — Gap 2: one model-call path with fallback, shared by every
non-instructor LLM call site (guardrails, judges, classifier, generation).

Why a separate module: rag.py imports the guardrail/classifier modules at load
time, so those leaf modules cannot import back from rag.py without a cycle. This
module has no project dependencies, so anything can import it.

Resilience policy:
  - LiteLLM installed  → litellm.completion(..., fallbacks=[FALLBACK_MODEL]).
  - LiteLLM absent      → raw OpenAI; on ANY exception, retry once on FALLBACK_MODEL.

A primary-model outage or rate-limit therefore degrades to the fallback model
instead of taking down guardrails / classification / judging (the original gap:
"one outage = full stop").
"""

import os
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

PRIMARY_MODEL = "gpt-4o-mini"
FALLBACK_MODEL = "gpt-3.5-turbo"
DEFAULT_TEMPERATURE = 0.0

try:
    import litellm
    litellm.set_verbose = False
    LITELLM_ENABLED = True
except ImportError:
    LITELLM_ENABLED = False

_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def llm_complete(
    messages: list[dict],
    model: str | None = None,
    *,
    response_format: dict | None = None,
    max_tokens: int | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """Return the completion content string, with single-fallback resilience."""
    selected_model = model or PRIMARY_MODEL
    kwargs: dict[str, Any] = {"messages": messages, "temperature": temperature}
    if response_format is not None:
        kwargs["response_format"] = response_format
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    if LITELLM_ENABLED:
        response = litellm.completion(
            model=selected_model,
            fallbacks=[FALLBACK_MODEL],
            **kwargs,
        )
        return response.choices[0].message.content.strip()

    try:
        response = _client.chat.completions.create(model=selected_model, **kwargs)
    except Exception:
        response = _client.chat.completions.create(model=FALLBACK_MODEL, **kwargs)
    return response.choices[0].message.content.strip()
