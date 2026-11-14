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
```

## Results

_In progress._

## Data Notes

- **XOM:** The ticker now maps to ExxonMobil Holdings Corp (CIK 2115436), which has not filed a 10-K yet. The latest 10-K is retrieved from the prior registrant, Exxon Mobil Corp (CIK 34088).

## Roadmap

- [ ] Week 1: Environment setup, data collection pipeline
- [ ] Week 2: End-to-end baseline RAG with citations
- [ ] Week 3: Evaluation framework and baseline scores
- [ ] Week 4: Comparative experiments
- [ ] Week 5: Embedding fine-tuning and final configuration
- [ ] Week 6: Demo deployment and technical report
