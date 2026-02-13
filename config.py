"""
Настройки и токены приложения (CLI + Telegram-бот).
"""

import os

from dotenv import load_dotenv

load_dotenv()

# OpenAI
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4.1")

# Telegram-бот
BOT_TOKEN: str = os.getenv("BOT_TOKEN", "")

# Модель для бота (отдельно от CLI)
BOT_OPENAI_MODEL: str = os.getenv("BOT_OPENAI_MODEL", "gpt-5-mini-2025-08-07")

# SQLite для учёта токенов
TOKEN_USAGE_DB_PATH: str = os.getenv("TOKEN_USAGE_DB_PATH", "token_usage.db")
