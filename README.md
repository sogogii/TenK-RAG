# TenK-RAG

Retrieval-augmented question answering over SEC 10-K filings, with a focus on table and numerical questions.

## Research Question

Where do RAG systems fail on table-heavy and numerical questions in 10-K filings, and which retrieval and preprocessing choices improve them?

## Motivation

10-K filings hold much of their key information in financial tables. Questions such as "What was the year-over-year revenue growth?" require locating the right table, extracting values, and often doing arithmetic. Standard RAG pipelines are built for prose and tend to struggle here. This project measures those failures and tests ways to fix them.

## Approach

- **Data:** Recent 10-K filings from 20 S&P 500 companies, collected via SEC EDGAR
- **Pipeline:** Parsing (text + tables) → chunking → embedding → vector search → LLM answer generation with source citations
- **Evaluation:**
  - [FinanceBench](https://github.com/patronus-ai/financebench) (primary benchmark)
  - A custom set of ~50 table and numerical questions (value lookup, growth calculation, year-over-year comparison, unanswerable questions)
  - Metrics: Recall@k, MRR, answer accuracy, citation accuracy
- **Experiments:**
  - General vs. finance-specific embedding models
  - Table handling: linearized text vs. preserved markdown structure
  - Chunk size and overlap
  - Top-k and reranking
  - Hybrid search (BM25 + dense)
  - Fine-tuning an embedding model on 10-K data

## Project Structure

```
TenK-RAG/
├── notebooks/      # Exploration and prototyping
├── src/tenk_rag/   # Core pipeline code
├── configs/        # Experiment configurations
├── data/           # Raw and processed filings (not tracked)
├── experiments/    # Experiment logs and results
└── tests/
```

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The SEC requires a User-Agent with your name and email:

```bash
export SEC_USER_AGENT="Your Name you@example.com"
```

## Usage

Download the latest 10-K for each company:

```bash
python -m tenk_rag.collect   # download the latest 10-K for each company
python -m tenk_rag.parse     # split filings into text and table blocks
python -m tenk_rag.chunk     # split blocks into retrieval-sized chunks
python -m tenk_rag.embed     # embed all chunks
python -m tenk_rag.search AAPL "What were Apple's total net sales in 2026?"
python -m tenk_rag.answer AAPL "What was diluted earnings per share in 2025?"
```

## Results

_In progress._

## Data Notes

- **XOM:** The ticker now maps to ExxonMobil Holdings Corp (CIK 2115436), which has not filed a 10-K yet. The latest 10-K is retrieved from the prior registrant, Exxon Mobil Corp (CIK 34088).

### Parser Limitations

- **Multi-level headers:** Headers spanning several columns (HTML `colspan`) are not aligned, e.g. PFE's revenue-by-geography table with Worldwide / U.S. / International groups.
- **Short table titles:** A title row shorter than 40 characters (e.g. "Costs and expenses follow:") is treated as a section label and prefixed to every row.
- **Unit labels in header rows:** A header row whose first cell is a unit, like "(MILLIONS) | 2025 | 2024", is parsed as a data row.
- **Section labels without a closing total:** A section label (e.g. "Denominator:") carries over to later rows when no "Total" row ends the section, producing labels like "Denominator - Diluted earnings per share".
- **Section labels inside the first row:** When a section label shares a cell with the first item (e.g. "Average shares: Basic"), later rows like "Diluted" don't get the label, so repeated row names stay ambiguous.
- **Hypothesis rejected:** Removing the company name from the BAC query removed name-matching debt chunks
  from the top 5, but the correct tables still ranked outside the top 10.
  The company name adds noise but is not the main cause of BAC misses.
- **Subtotals inside a section:** A "Total" row closes the current section, so rows after a subtotal (e.g. GS segment "Pre-tax earnings") lose their segment label.

After these fixes, at most 7% of tables per company lack a detected header (see `notebooks/05_parse_check.ipynb`).

## Roadmap

- [x] Week 1: Environment setup, data collection pipeline
- [ ] Week 2: End-to-end baseline RAG with citations
- [ ] Week 3: Evaluation framework and baseline scores
- [ ] Week 4: Comparative experiments
- [ ] Week 5: Embedding fine-tuning and final configuration
- [ ] Week 6: Demo deployment and technical report
