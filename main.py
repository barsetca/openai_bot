#!/usr/bin/env python3
"""
Интерактивный CLI для запросов к OpenAI API.
"""

import sys

from config import OPENAI_API_KEY, OPENAI_MODEL
from openai_client import get_chat_response

TEMPERATURE_MIN = 0.0
TEMPERATURE_MAX = 2.0
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1000


def prompt_message() -> str:
    """Запрашивает текстовое сообщение (prompt)."""
    print("\n  ╭─────────────────────────────────────────╮")
    print(f"  │   Запрос к OpenAI (модель {OPENAI_MODEL})    │")
    print("  ╰─────────────────────────────────────────╯\n")
    while True:
        value = input("  Введите сообщение с запросом: ").strip()
        if value:
            return value
        print("  ⚠ Сообщение не может быть пустым. Попробуйте снова.")


def prompt_temperature() -> float:
    """Запрашивает температуру с проверкой диапазона."""
    print(f"\n  Допустимый интервал температуры: {TEMPERATURE_MIN} — {TEMPERATURE_MAX}")
    hint = f"  Введите температуру (Enter = {DEFAULT_TEMPERATURE}): "
    while True:
        raw = input(hint).strip()
        if not raw:
            return DEFAULT_TEMPERATURE
        try:
            value = float(raw)
            if TEMPERATURE_MIN <= value <= TEMPERATURE_MAX:
                return value
        except ValueError:
            pass
        print(f"  ⚠ Нужно число от {TEMPERATURE_MIN} до {TEMPERATURE_MAX}. Попробуйте снова.")


def prompt_max_tokens() -> int:
    """Запрашивает максимальное количество токенов."""
    hint = f"  Введите максимальное количество токенов в ответе (Enter = {DEFAULT_MAX_TOKENS}): "
    while True:
        raw = input(hint).strip()
        if not raw:
            return DEFAULT_MAX_TOKENS
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
        print("  ⚠ Введите целое положительное число. Попробуйте снова.")


def prompt_system_message() -> str | None:
    """Опционально запрашивает system message."""
    raw = input("  Системное сообщение для роли/стиля (необязательно, Enter — пропустить): ").strip()
    return raw if raw else None


def main() -> None:
    if not OPENAI_API_KEY:
        print("Ошибка: задайте OPENAI_API_KEY в .env или в переменных окружения.", file=sys.stderr)
        sys.exit(1)

    message = prompt_message()
    temperature = prompt_temperature()
    max_tokens = prompt_max_tokens()
    system_message = prompt_system_message()

    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": message})

    print("\n  Отправляю запрос к модели...\n")
    try:
        content, usage = get_chat_response(
            messages,
            OPENAI_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        print(f"Ошибка API: {e}", file=sys.stderr)
        sys.exit(1)

    print("  ╭─────────────────────────────────────────╮")
    print("  │              Ответ модели               │")
    print("  ╰─────────────────────────────────────────╯\n")
    print(content)
    print("\n  ─────────────────────────────────────────")
    print("  Сопутствующая информация:")
    print(f"    • Температура запроса:     {temperature}")
    if usage:
        print(f"    • Токенов в запросе:       {usage['prompt_tokens']}")
        print(f"    • Токенов в ответе:       {usage['completion_tokens']}")
        print(f"    • Всего токенов:          {usage['total_tokens']}")
    print("  ─────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
