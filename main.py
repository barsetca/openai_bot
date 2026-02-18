#!/usr/bin/env python3
"""
Интерактивный CLI для запросов к OpenAI API.
Диалог: промпт (роль, контекст, задача, формат) + параметры → подтверждение → запрос.
После ответа возможен дозапрос или начало заново.
"""

import sys

from config import OPENAI_API_KEY, OPENAI_MODEL
from openai_client import get_chat_response

TEMPERATURE_MIN = 0.0
TEMPERATURE_MAX = 2.0
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1000

COMMAND_NEW = "/new"
COMMAND_QUIT = "/quit"


def _print_header() -> None:
    print("\n  ╭─────────────────────────────────────────╮")
    print(f"  │   Запрос к OpenAI (модель {OPENAI_MODEL})    │")
    print("  ╰─────────────────────────────────────────╯\n")


def _ask_optional(label: str, prompt: str) -> str:
    """Опциональное поле: Enter — пусто."""
    return input(f"  {prompt}").strip()


def _ask_required(label: str, prompt: str) -> str:
    """Обязательное поле."""
    while True:
        value = input(f"  {prompt}").strip()
        if value:
            return value
        print("  ⚠ Обязательное поле. Введите значение.")


def collect_prompt() -> tuple[list[dict[str, str]], str]:
    """
    Собирает часть «Промпт»: роль, контекст, задача, формат ответа.
    Возвращает (messages для API, итоговая строка промпта для показа пользователю).
    """
    print("  ─── Промпт ───\n")
    role = _ask_optional("роль", "Роль (необязательно, Enter — пропустить): ")
    context = _ask_optional("контекст", "Контекст (необязательно, Enter — пропустить): ")
    task = _ask_required("задача", "Задача (обязательно): ")
    output_format = _ask_optional("формат", "Формат ответа (необязательно, Enter — пропустить): ")

    parts_display = []
    if role:
        parts_display.append(f"Роль: {role}")
    if context:
        parts_display.append(f"Контекст: {context}")
    parts_display.append(f"Задача: {task}")
    if output_format:
        parts_display.append(f"Формат ответа: {output_format}")
    display_prompt = "\n".join(parts_display)

    messages: list[dict[str, str]] = []
    if role:
        messages.append({"role": "system", "content": role})
    user_parts = []
    if context:
        user_parts.append(f"Контекст: {context}")
    user_parts.append(f"Задача: {task}")
    if output_format:
        user_parts.append(f"Формат ответа: {output_format}")
    messages.append({"role": "user", "content": "\n\n".join(user_parts)})

    return messages, display_prompt


def collect_parameters() -> tuple[float, int]:
    """Собирает часть «Параметры»: температура и макс. токенов."""
    print("\n  ─── Параметры ───\n")
    print(f"  Допустимый интервал температуры: {TEMPERATURE_MIN} — {TEMPERATURE_MAX}")
    hint_t = f"  Температура (Enter = {DEFAULT_TEMPERATURE}): "
    while True:
        raw = input(hint_t).strip()
        if not raw:
            temperature = DEFAULT_TEMPERATURE
            break
        try:
            temperature = float(raw)
            if TEMPERATURE_MIN <= temperature <= TEMPERATURE_MAX:
                break
        except ValueError:
            pass
        print(f"  ⚠ Введите число от {TEMPERATURE_MIN} до {TEMPERATURE_MAX}.")

    hint_m = f"  Максимальное количество токенов в ответе (Enter = {DEFAULT_MAX_TOKENS}): "
    while True:
        raw = input(hint_m).strip()
        if not raw:
            max_tokens = DEFAULT_MAX_TOKENS
            break
        try:
            max_tokens = int(raw)
            if max_tokens > 0:
                break
        except ValueError:
            pass
        print("  ⚠ Введите целое положительное число.")

    return temperature, max_tokens


def confirm_and_send(
    messages: list[dict[str, str]],
    display_prompt: str,
    temperature: float,
    max_tokens: int,
) -> tuple[str | None, dict[str, int] | None]:
    """
    Показывает итог, запрашивает согласие. При «да» — запрос к API.
    Возвращает (content, usage) или (None, None) если пользователь не согласен.
    """
    print("\n  ╭─────────────────────────────────────────╮")
    print("  │           Итоговый промпт                 │")
    print("  ╰─────────────────────────────────────────╯\n")
    print(display_prompt)
    print("\n  Параметры:")
    print(f"    Температура: {temperature}")
    print(f"    Макс. токенов: {max_tokens}")
    print()
    while True:
        answer = input("  Отправить запрос? (да/нет): ").strip().lower()
        if answer in ("да", "yes", "y", "д"):
            break
        if answer in ("нет", "no", "n", "н"):
            return None, None
        print("  Введите «да» или «нет».")

    print("\n  Отправляю запрос к модели...\n")
    try:
        content, usage = get_chat_response(
            messages,
            OPENAI_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return content, usage
    except Exception as e:
        print(f"Ошибка API: {e}", file=sys.stderr)
        return None, None


def print_response(content: str, temperature: float, usage: dict[str, int] | None) -> None:
    """Выводит ответ модели и сопутствующую информацию."""
    print("  ╭─────────────────────────────────────────╮")
    print("  │              Ответ модели                 │")
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


def run_dialog_cycle() -> bool:
    """
    Один цикл: сбор промпта → параметры → подтверждение → запрос (или отмена).
    После ответа: дозапрос, /new (начать заново) или /quit (выход).
    Возвращает True чтобы начать цикл заново, False для выхода из программы.
    """
    _print_header()

    messages, display_prompt = collect_prompt()
    temperature, max_tokens = collect_parameters()

    content, usage = confirm_and_send(messages, display_prompt, temperature, max_tokens)
    if content is None and usage is None:
        return True  # не согласен — начать заново

    print_response(content or "", temperature, usage)

    # Цикл дозапросов
    while True:
        print(f"  Дозапрос (текст), {COMMAND_NEW} — начать заново, {COMMAND_QUIT} — выход.")
        user_input = input("  > ").strip()
        if not user_input:
            continue
        if user_input.lower() in (COMMAND_QUIT, "выход", "quit", "q"):
            return False
        if user_input.lower() in (COMMAND_NEW, "заново", "new"):
            return True  # начать заново

        # Дозапрос: добавляем ответ ассистента и новое сообщение пользователя
        messages.append({"role": "assistant", "content": content or ""})
        messages.append({"role": "user", "content": user_input})

        print("\n  Отправляю запрос...\n")
        try:
            content, usage = get_chat_response(
                messages,
                OPENAI_MODEL,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            print(f"Ошибка API: {e}", file=sys.stderr)
            continue
        print_response(content or "", temperature, usage)


def main() -> None:
    if not OPENAI_API_KEY:
        print("Ошибка: задайте OPENAI_API_KEY в .env или в переменных окружения.", file=sys.stderr)
        sys.exit(1)

    while run_dialog_cycle():
        pass  # начать заново
    print("  До свидания.\n")


if __name__ == "__main__":
    main()
