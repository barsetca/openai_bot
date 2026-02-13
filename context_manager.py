"""
Управление контекстом диалога: оперативная память (dict) и учёт токенов в SQLite.
"""

import logging
import sqlite3
from pathlib import Path
from config import TOKEN_USAGE_DB_PATH

logger = logging.getLogger(__name__)

# Контекст по user_id: список сообщений для OpenAI
_contexts: dict[int, list[dict[str, str]]] = {}


def _get_connection() -> sqlite3.Connection:
    path = Path(TOKEN_USAGE_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(path), check_same_thread=False)


def init_token_usage_db() -> None:
    """Создаёт таблицу для учёта токенов по запросам/ответам."""
    try:
        conn = _get_connection()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS token_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
            """
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.exception("Ошибка инициализации БД токенов: %s", e)
        raise


def get_messages(user_id: int) -> list[dict[str, str]]:
    """Возвращает текущий контекст сообщений пользователя (копию списка)."""
    return list(_contexts.get(user_id, []))


def append_to_context(
    user_id: int,
    user_content: str,
    assistant_content: str,
) -> None:
    """Добавляет пару user/assistant в контекст пользователя."""
    if user_id not in _contexts:
        _contexts[user_id] = []
    _contexts[user_id].append({"role": "user", "content": user_content})
    _contexts[user_id].append({"role": "assistant", "content": assistant_content})


def clear_context(user_id: int) -> None:
    """Очищает контекст диалога для пользователя."""
    _contexts[user_id] = []
    logger.info("Контекст очищен для user_id=%s", user_id)


def log_token_usage(
    user_id: int,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
) -> None:
    """Сохраняет информацию о потраченных токенах в SQLite."""
    try:
        conn = _get_connection()
        conn.execute(
            "INSERT INTO token_usage (user_id, prompt_tokens, completion_tokens, total_tokens) VALUES (?, ?, ?, ?)",
            (user_id, prompt_tokens, completion_tokens, total_tokens),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.exception("Ошибка записи token_usage: %s", e)


def get_user_token_stats(user_id: int) -> tuple[int, int]:
    """
    Возвращает суммарные токены запросов и ответов для пользователя.

    Returns:
        (total_prompt_tokens, total_completion_tokens)
    """
    try:
        conn = _get_connection()
        row = conn.execute(
            "SELECT COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(completion_tokens), 0) FROM token_usage WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        conn.close()
        return (row[0], row[1]) if row else (0, 0)
    except Exception as e:
        logger.exception("Ошибка чтения статистики токенов: %s", e)
        return (0, 0)
