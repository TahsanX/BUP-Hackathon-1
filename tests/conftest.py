"""
tests/conftest.py — Load .env before any test module imports app.* code,
so API keys (GEMINI_API_KEY, GROQ_API_KEY, HF_API_TOKEN) are available.
"""
from dotenv import load_dotenv

load_dotenv()
