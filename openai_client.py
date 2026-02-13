"""
Клиент для общения с OpenAI API (используется CLI и Telegram-ботом).
"""

import logging
from typing import Any

from openai import OpenAI

from config import OPENAI_API_KEY

logger = logging.getLogger(__name__)


def get_chat_response(
    messages: list[dict[str, str]],
    model: str,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> tuple[str, dict[str, int] | None]:
    """
    Отправляет сообщения в OpenAI Chat API и возвращает ответ и usage.

    Рассуждающие модели (reasoning) не поддерживают temperature и max_tokens —
    передавайте None (по умолчанию), чтобы не отправлять эти параметры.

    Args:
        messages: Список сообщений [{"role": "user"|"assistant"|"system", "content": "..."}]
        model: Имя модели (например gpt-5-mini-2025-08-07).
        temperature: Температура генерации (None — не передавать, для reasoning-моделей).
        max_tokens: Максимум токенов в ответе (None — не передавать, для reasoning-моделей).

    Returns:
        (content, usage_dict) — текст ответа и словарь с prompt_tokens, completion_tokens, total_tokens.
        usage_dict может быть None при отсутствии данных в ответе.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY не задан")

    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as e:
        logger.exception("OpenAI API error: %s", e)
        raise

    content = response.choices[0].message.content or ""
    usage: dict[str, int] | None = None
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return content, usage
