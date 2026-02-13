"""
Telegram-бот с OpenAI (gpt-5-mini). Контекст в памяти, учёт токенов в SQLite.
"""

import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message

from config import BOT_OPENAI_MODEL, BOT_TOKEN, OPENAI_API_KEY
from context_manager import (
    append_to_context,
    clear_context,
    get_messages,
    get_user_token_stats,
    init_token_usage_db,
    log_token_usage,
)
from openai_client import get_chat_response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

CLEAR_PHRASES = ("очистить контекст", "очистить", "clear context", "clear")

# Стоимость токенов: запросы $0.25 / 1M, ответы $2 / 1M
PROMPT_COST_PER_1M = 0.25
COMPLETION_COST_PER_1M = 2.0

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


def _call_openai(messages: list[dict[str, str]]) -> tuple[str, dict[str, int] | None]:
    """Синхронный вызов OpenAI (для run_in_executor). Без temperature/max_tokens — для рассуждающих моделей."""
    return get_chat_response(messages, BOT_OPENAI_MODEL)


@dp.message(Command("start"))
async def cmd_start(message: Message) -> None:
    await message.answer(
        "Привет! Я бот с GPT. Пиши сообщения — я буду отвечать с учётом контекста.\n"
        "/clear или «очистить контекст» — сбросить историю.\n"
        "/stats — статистика токенов и стоимость."
    )


@dp.message(Command("stats"))
async def cmd_stats(message: Message) -> None:
    """Статистика токенов и стоимость в долларах."""
    user_id = message.from_user.id if message.from_user else 0
    prompt_tokens, completion_tokens = get_user_token_stats(user_id)
    cost_prompt = (prompt_tokens / 1_000_000) * PROMPT_COST_PER_1M
    cost_completion = (completion_tokens / 1_000_000) * COMPLETION_COST_PER_1M
    total_cost = cost_prompt + cost_completion
    text = (
        "📊 <b>Статистика токенов</b>\n\n"
        f"Токенов запросов: <b>{prompt_tokens:,}</b>\n"
        f"Токенов ответов: <b>{completion_tokens:,}</b>\n"
        f"Всего токенов: <b>{prompt_tokens + completion_tokens:,}</b>\n\n"
        "💰 <b>Стоимость</b>\n"
        f"Запросы ($0.25/1M): <b>${cost_prompt:.6f}</b>\n"
        f"Ответы ($2/1M): <b>${cost_completion:.6f}</b>\n"
        f"Итого: <b>${total_cost:.6f}</b>"
    )
    await message.answer(text, parse_mode="HTML")


@dp.message(Command("clear"))
async def cmd_clear(message: Message) -> None:
    user_id = message.from_user.id if message.from_user else 0
    clear_context(user_id)
    await message.answer("Контекст диалога очищен. Можете начать разговор заново.")


@dp.message(F.text)
async def handle_text(message: Message) -> None:
    if not message.text or not message.from_user:
        return

    user_id = message.from_user.id
    text = message.text.strip()

    if text.lower() in CLEAR_PHRASES or text == "/clear":
        clear_context(user_id)
        await message.answer("Контекст диалога очищен. Можете начать разговор заново.")
        return

    messages = get_messages(user_id) + [{"role": "user", "content": text}]

    await message.answer("Думаю…")
    loop = asyncio.get_event_loop()
    try:
        content, usage = await loop.run_in_executor(None, _call_openai, messages)
    except Exception as e:
        logger.exception("OpenAI error for user %s: %s", user_id, e)
        await message.answer(
            "Произошла ошибка при обращении к модели. Попробуйте позже или упростите запрос."
        )
        return

    append_to_context(user_id, text, content)

    if usage:
        log_token_usage(
            user_id,
            usage["prompt_tokens"],
            usage["completion_tokens"],
            usage["total_tokens"],
        )

    if len(content) > 4000:
        content = content[:3997] + "..."
    await message.answer(content)


async def main() -> None:
    if not BOT_TOKEN:
        logger.error("BOT_TOKEN не задан. Укажите его в .env")
        sys.exit(1)
    if not OPENAI_API_KEY:
        logger.error("OPENAI_API_KEY не задан. Укажите его в .env")
        sys.exit(1)

    init_token_usage_db()
    logger.info("Бот запущен (модель: %s)", BOT_OPENAI_MODEL)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
