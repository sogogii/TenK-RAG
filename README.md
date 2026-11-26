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

### Preliminary Findings

_Based on 29 retrieval questions across 6 companies (AAPL, JPM, BAC, GS, XOM, CVX). The evaluation set is still growing, so these numbers will change._

Baseline: `all-MiniLM-L6-v2` embeddings, 150-word text chunks, token-limited table chunks, top-k cosine similarity within one company.

| Metric    | Score |
| --------- | ----- |
| Recall@5  | 0.31  |
| Recall@10 | 0.52  |
| Recall@50 | 0.86  |

**1. Retrieval, not generation, is the bottleneck.** When the correct chunk is retrieved, the LLM reads the right year column and cites it. When it is not, the LLM correctly answers that the sources lack the information instead of guessing.

**2. Tables lose to narrative text.** For table questions, the top results are often MD&A paragraphs about the same topic, while the table holding the exact number ranks lower.

**3. Paraphrased questions rarely find tables.** Questions reusing a table's row labels reach Recall@5 of about 50%, but paraphrased questions (e.g. "profit kept after all costs and taxes" instead of "net income") found the table in only 1 of 8 cases.

**4. Most misses are near misses.** The correct chunk is in the top 50 for 86% of questions, so reranking the top 50 could raise Recall@5 substantially.

**5. Some misses are deep.** For XOM revenue and R&D, the questions use the table's exact row labels, yet the correct chunks rank outside the top 50. These are candidates for keyword (BM25) retrieval.

**6. Rejected hypotheses.**

- _Truncation:_ Measuring table chunks in model tokens cut tables over the 256-token limit from 36% to 5%, but retrieval barely changed.
- _Company name in the query:_ Removing "Bank of America" from a BAC query removed name-matching chunks, but the correct tables still ranked outside the top 10.

### Experiment 1: Cross-Encoder Reranking (71 questions, 20 companies)

| Metric    | Baseline | + Rerank |
| --------- | -------- | -------- |
| Recall@1  | 0.14     | 0.18     |
| Recall@5  | 0.35     | 0.44     |
| Recall@10 | 0.55     | 0.59     |
| Recall@50 | 0.83     | 0.83     |
| MRR@50    | 0.26     | 0.31     |

Reranking improved 30 questions and hurt 19. The gain was uneven: on the first 41 questions Recall@5 rose from 10 to 16 hits, but on the 30 questions added later it stayed at 15, with 10 improved and 11 worse. An early 41-question run overstated the effect by almost half.

Paraphrased questions reached 4/22 in the top 5. Results depend on wording: "advertising and marketing" found Nike's "Demand creation expense" at rank 2, while "profit kept after all costs and taxes" failed across several companies.

### Experiment 2: Query Rewriting

Claude Haiku rewrites each question into the standard line-item terms used in
financial statements (e.g., "profit after all expenses and taxes" → "net income").
Rewrites are cached in `eval/rewrites.json` so runs are reproducible.
31 of 71 questions changed: all 22 paraphrases and 9 lookups.

| Setting                | R@1  | R@5  | R@10 | R@50 | MRR@50 | Lookup R@5 | Paraphrase R@5 |
| ---------------------- | ---- | ---- | ---- | ---- | ------ | ---------- | -------------- |
| Baseline               | 0.14 | 0.35 | 0.55 | 0.83 | 0.26   | 21/48      | 4/22           |
| Rewrite only           | 0.17 | 0.38 | 0.52 | 0.87 | 0.29   | 20/48      | 7/22           |
| Rerank only            | 0.18 | 0.44 | 0.59 | 0.83 | 0.31   | 26/48      | 4/22           |
| Rewrite at rerank only | 0.24 | 0.56 | 0.69 | 0.83 | 0.39   | 26/48      | 13/22          |
| Rewrite + rerank       | 0.24 | 0.56 | 0.70 | 0.87 | 0.39   | 25/48      | 14/22          |

**Findings**

- Rewriting and reranking work together. Rewriting alone adds 0.03 to Recall@5
  and reranking alone adds 0.09, but both together add 0.21. The cross-encoder
  reads the query and chunk side by side, so matching the table's exact
  line-item name helps it much more than it helps the bi-encoder.
- Rewriting at search time is what raises Recall@50 (0.83 → 0.87). Four
  questions entered the top 50, which reranking alone can never do, and two of
  them then reached the top 3 after reranking (aapl-12, ba-03).
- Paraphrase Recall@5 went from 4/22 to 14/22. The rewriter even used
  company-specific terms, such as Nike's "demand creation expense".
- The cost falls on lookups (26 → 25). gs-01 dropped from rank 1 to outside the
  top 50 because the rewrite changed "How much did Goldman Sachs earn from investment banking in 2024?" to "What were Goldman Sachs' investment banking fees in 2024?". jnj-01 ("sales to customers" → "net sales") kept rank 1.
- 7 of the 9 questions still outside the top 50 are lookups that already use
  standard terms, so rewriting cannot fix them. These motivate Experiment 3
  (hybrid BM25).
- The "rewrite at rerank only" row came from a bug where search used the
  original question. It turned out to be a useful variant: almost the same
  Recall@5, but without the Recall@50 gain.

  ### Experiment 3: Hybrid Search (BM25 + Dense)

BM25 keyword search is built per company with `rank-bm25`. Hybrid search takes
the top 50 from dense and BM25 search and merges them with Reciprocal Rank
Fusion (score = sum of 1 / (60 + rank)).

| Setting                       | R@1      | R@3      | R@5      | R@10     | R@50     | MRR@50   |
| ----------------------------- | -------- | -------- | -------- | -------- | -------- | -------- |
| Dense (baseline)              | 0.14     | 0.32     | 0.35     | 0.55     | 0.83     | 0.26     |
| BM25                          | 0.13     | 0.25     | 0.35     | 0.54     | 0.79     | 0.24     |
| Hybrid                        | 0.20     | 0.34     | 0.42     | 0.55     | 0.89     | 0.31     |
| Hybrid + rerank               | 0.20     | 0.35     | 0.46     | 0.62     | 0.89     | 0.33     |
| Dense + rewrite + rerank      | 0.24     | 0.46     | 0.56     | 0.70     | 0.87     | 0.39     |
| **Hybrid + rewrite + rerank** | **0.25** | **0.48** | **0.59** | **0.77** | **0.97** | **0.42** |

**Predictions (written before running)**

- BM25 will be weak on paraphrases. ✅ Without rewriting, hybrid hurt most
  paraphrases (e.g., cat-03 18 → 36, pfe-03 50 → miss).
- Hybrid will raise Recall@50 and recover exact-term lookups like amzn-02. ✅
- BM25 will not help aapl-06 ("net income"), since the term appears everywhere.
  ❌ It moved from outside the top 50 to rank 21.

**Findings**

- BM25 alone matches dense search on Recall@5 (0.35), but they find different
  questions: hybrid Recall@50 (0.89) is higher than either one.
- The three methods cover each other's weaknesses. Rewriting turns paraphrases
  into exact terms, BM25 finds tables by those terms, and reranking orders the
  larger pool. Adding hybrid to the best Experiment 2 setting gave 14 better and
  7 worse; every loss was 1–2 ranks.
- Questions outside the top 50 dropped from 9 to 2 (xom-02, msft-03). gs-01,
  lost in Experiment 2, came back (miss → 6).
- The answer is now in the top 50 for 69 of 71 questions, but in the top 5 for
  only 42. The remaining problem is mostly ranking, so a stronger reranker is
  the next step.

## Data Notes

- **XOM:** The ticker now maps to ExxonMobil Holdings Corp (CIK 2115436), which has not filed a 10-K yet. The latest 10-K is retrieved from the prior registrant, Exxon Mobil Corp (CIK 34088).

### Parser Limitations

- **Multi-level headers:** Headers spanning several columns (HTML `colspan`) are not aligned, e.g. PFE's revenue-by-geography table with Worldwide / U.S. / International groups.
- **Short table titles:** A title row shorter than 40 characters (e.g. "Costs and expenses follow:") is treated as a section label and prefixed to every row.
- **Unit labels in header rows:** A header row whose first cell is a unit, like "(MILLIONS) | 2025 | 2024", is parsed as a data row.
- **Section labels without a closing total:** A section label (e.g. "Denominator:") carries over to later rows when no "Total" row ends the section, producing labels like "Denominator - Diluted earnings per share".
- **Section labels inside the first row:** When a section label shares a cell with the first item (e.g. "Average shares: Basic"), later rows like "Diluted" don't get the label, so repeated row names stay ambiguous.
- **Subtotals inside a section:** A "Total" row closes the current section, so rows after a subtotal (e.g. GS segment "Pre-tax earnings") lose their segment label.
- **Header shifted by a split column:** When one header cell spans two data columns (e.g. UNH's "Change 2025 vs. 2024" over a $ and a % column), the header is padded from the left and every year label shifts one column, so values sit under the wrong year.
- **Rows with empty cells shift left:** Empty cells are dropped and short rows are padded on the right, so in tables with interleaved columns (e.g. NKE's "FISCAL 2025 | % CHANGE | FISCAL 2024"), a row without change values puts its 2024 figure under "% CHANGE".
- **Numeric footnotes and split negatives:** Numeric footnote markers (e.g. CAT's superscript 1, 2) are kept as values and shift later columns, and negative percentages split across cells (e.g. "(2" and "%)") are not rejoined.
- **Rows without a label:** A subtotal row with no label (e.g. BA's gross profit line) uses its first value as the label, shifting the remaining values one column left.

After these fixes, at most 7% of tables per company lack a detected header (see `notebooks/05_parse_check.ipynb`).

## Roadmap

- [x] Week 1: Environment setup, data collection pipeline
- [x] Week 2: End-to-end baseline RAG with citations
- [x] Week 3: Evaluation framework and baseline scores
- [x] Week 4: Comparative experiments
- [ ] Week 5: Embedding fine-tuning and final configuration
- [ ] Week 6: Demo deployment and technical report
