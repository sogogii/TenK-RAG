"""Answer questions about a 10-K using retrieved chunks and an LLM."""

import sys

import anthropic

from tenk_rag.config import LLM_MODEL
from tenk_rag.search import Searcher

SYSTEM_PROMPT = """You answer questions about a company's 10-K filing using onyl the provided sources.

Rules:
- Use only the information in the sources. Do not use the outside knowledge.
- Cite every fact with its source number, like [1] or [2].
- Check table headers and units: which column is which year, and whether values are in millions, thousands, or per share.
- If the sources do not contain the answer, say "The provided sources do not contain this information." Do not guess.
"""


def build_prompt(question, results):
	"""Number each retrieved chunk so the answer can cite it."""
	sources = "\n\n".join(
		f"[{i}] (id: {chunk['id']}, filed {chunk['filing_date']})\n{chunk['text']}"
		for i, (chunk, _) in enumerate(results, start=1)
	)
	return f"Sources:\n\n{sources}\n\nQuestion: {question}"


class Answerer:
	"""Retrieves chunks for a question and asks the LLM to answer from them."""

	def __init__(self, k=5):
		self.searcher = Searcher()
		self.client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from the environment
		self.k = k

	def answer(self, question, ticker):
		results = self.searcher.search(question, ticker, k=self.k)
		response = self.client.messages.create(
			model=LLM_MODEL,
			max_tokens=500,
			system=SYSTEM_PROMPT,
			messages=[{"role": "user", "content": build_prompt(question, results)}],
		)
		# The response may contain several blocks, so only the text blocks are joined
		text = "".join(block.text for block in response.content if block.type == "text")
		return text, results


def main():
	if len(sys.argv) < 3:
		print('Usage: python -m tenk_rag.answer TICKER "question"')
		return

	ticker, question = sys.argv[1].upper(), " ".join(sys.argv[2:])
	text, results = Answerer().answer(question, ticker)

	print(text)
	print("\nSources:")
	for i, (chunk, score) in enumerate(results, start=1):
		print(f"[{i}] {chunk['id']} ({chunk['type']}, score {score:.3f})")


if __name__=="__main__":
	main()