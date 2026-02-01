import os
from dotenv import load_dotenv

load_dotenv()

APP_NAME = "AI Journaling Companion"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = """
Privacy-first AI-powered journaling companion.

Features:
- Local sentiment analysis (no external APIs)
- Rule-based theme detection
- Empathetic, non-judgmental reflections
- In-memory storage only (no persistence)

Privacy guarantees:
- All processing happens locally
- No data leaves your system
- No user tracking or authentication
- Data cleared on restart
"""

API_PREFIX = ""
CORS_ORIGINS = ["*"]

# OpenAI Configuration (Optional)
# If enabled, enhances reflection wording for emotional resonance and engagement
# Local NLP (sentiment + themes) remains the authoritative source of truth
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
USE_OPENAI_REFINEMENT = os.getenv("USE_OPENAI_REFINEMENT", "false").lower() == "true"
OPENAI_MODEL = "gpt-4o-mini"
OPENAI_MAX_TOKENS = 300  # Increased to support 3-6 sentence reflections
OPENAI_TEMPERATURE = 0.8  # Slightly higher for more natural, warm responses
