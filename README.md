# rag-from-scratch

A small search system, built by hand, that finds the passage of EU sustainability law answering a plain-English question, with a word-counting and an embedding version.

> **Part of a series on RAG over EU sustainability law:** **rag-from-scratch** (retrieval built by hand) → rag-eval (not public yet) (measured: three scores and a Claude judge) → rag-claude-citations (not public yet) (API-native: citations and caching)

**Result:** on 10 questions at k = 5, version A (TF-IDF word counts, written by hand in numpy) and version B (`BAAI/bge-small-en-v1.5` embedding model on the CPU) both score 6 / 10.

## Run it

Tested on Python 3.13 (Windows 11). In the repo folder:

```bash
python -m venv .venv
```

```bash
.venv/Scripts/activate        # Windows; on macOS or Linux: source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
python run.py
```

Downloads and checks: [docs/details.md](docs/details.md).

## How it works

Each text is cut into chunks of 200 words, each overlapping the next by 50 words. Chunks and questions become lists of numbers: version A counts words (TF-IDF), version B uses 384 numbers from a small trained model. The question is compared with every chunk by cosine similarity and the top k = 5 chunks are returned. A language model never sees a chunk outside the top k, so retrieval is measured first. See [docs/architecture.md](docs/architecture.md).

## Results

| Version | How it turns text into numbers | Hit rate at k = 5 |
|---|---|---|
| A | TF-IDF word counts, written by hand in numpy | **6 / 10** |
| B | `BAAI/bge-small-en-v1.5` embedding model on the CPU | **6 / 10** |

Other depths, same run:

| Version | k = 1 | k = 3 | k = 5 | k = 10 |
|---|---|---|---|---|
| A, TF-IDF | 5 | 6 | 6 | 7 |
| B, bge-small | 2 | 3 | 6 | 8 |

Per-question table, test set and failure analysis: [docs/details.md](docs/details.md).

## Limits

- 10 questions is a small test. One question is 10 percentage points; the A versus B tie
  says nothing about which is better in general.
- I wrote the questions knowing the texts, so some reuse the law's wording. That favours
  version A. Questions from people who have not read the law would likely favour B.
- Texts as originally published in the Official Journal. Later amendments are not
  included, so an answer such as q05 reflects the 2022 text, which may differ from the law
  in force today.
- The hit rates measure retrieval only. Answers from `ask.py` are not scored yet.

## Data and licence

| Key | Text | CELEX | Words |
|---|---|---|---|
| `taxonomy` | Regulation (EU) 2020/852, the EU Taxonomy Regulation | 32020R0852 | 18,379 |
| `csrd` | Directive (EU) 2022/2464, the Corporate Sustainability Reporting Directive | 32022L2464 | 39,408 |
| `esrs` | Commission Delegated Regulation (EU) 2023/2772, ESRS set 1 | 32023R2772 | 130,571 |

EU legal texts: © European Union, https://eur-lex.europa.eu. EUR-Lex's legal notice states
that, unless otherwise specified, its legal documents may be re-used for commercial or
non-commercial purposes, under the Commission's reuse policy (Decision 2011/833/EU):
https://eur-lex.europa.eu/content/legal-notice/legal-notice.html (checked 2026-11-19). Only
the official Official Journal texts are authentic; this tool is not legal advice.

Embedding model: `BAAI/bge-small-en-v1.5` by BAAI, run through
[fastembed](https://github.com/qdrant/fastembed). Model card:
https://huggingface.co/BAAI/bge-small-en-v1.5

Code: MIT licence, see `LICENSE`.
