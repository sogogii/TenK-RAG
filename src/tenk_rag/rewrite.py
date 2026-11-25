"""Rewrite questions into financial-statement terms before retrieval."""

import json
import anthropic

from tenk_rag.config import LLM_MODEL, PROJECT_ROOT

CACHE_PATH = PROJECT_ROOT / "eval" / "rewrites.json"

SYSTEM_PROMPT = """You rewrite questions about a company's 10-K filing so they use the standard
line-item names that appear in financial statements, such as "net income", "operating income",
or "research and development expense."

Rules:
- Keep the company name and the year exactly as given.
- Do not add facts, numbers, or extra questions.
- If the question already uses standard terms, return it unchanged.
- Reply with the rewritten question only."""


class Rewriter:
    """Rewrites questions with an LLM, caching results so every run uses the same rewrite."""

    def __init__(self):
        self.client = anthropic.Anthropic()
        self.cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}

    def rewrite(self, question):
        # Calls the API only for new questions, saving after each one
        if question not in self.cache:
            response = self.client.messages.create(
                model=LLM_MODEL,
                max_tokens=1000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": question}],
            )
            text = "".join(b.text for b in response.content if b.type == "text").strip()
            if response.stop_reason != "end_turn" or not text:
                raise RuntimeError(f"Rewrite failed (stop_reason={response.stop_reason}): {question}")
            self.cache[question] = text
            CACHE_PATH.write_text(json.dumps(self.cache, indent=2, ensure_ascii=False))
        return self.cache[question]