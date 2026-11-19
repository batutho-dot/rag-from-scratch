# rag-from-scratch

Finds the passage of EU sustainability law that answers a plain-English question, with a word-count version and an embedding version.

> **Part of a series on RAG over EU sustainability law:** **rag-from-scratch** (retrieval built by hand) → [rag-eval](https://github.com/batutho-dot/rag-eval) (measured: three scores and a Claude judge) → rag-claude-citations (not public yet; API-native: citations and caching)

**Result:** on 10 questions at k = 5, both versions score 6 / 10.

## Run it

Tested on Python 3.13 (Windows 11).

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; on macOS or Linux: source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

## How it works

Texts are cut into 200-word chunks overlapping by 50. Version A turns chunks and questions into word counts (TF-IDF), version B into 384 numbers from a small trained model. Cosine similarity ranks the chunks and the top k = 5 come back. A language model never sees a chunk outside the top k, so I measure retrieval first. [Architecture](docs/architecture.md).

## Results

| Version | How it turns text into numbers | Hit rate at k = 5 |
|---|---|---|
| A | TF-IDF word counts, written by hand in numpy | **6 / 10** |
| B | `BAAI/bge-small-en-v1.5` embedding model on the CPU | **6 / 10** |

| Version | k = 1 | k = 3 | k = 5 | k = 10 |
|---|---|---|---|---|
| A, TF-IDF | 5 | 6 | 6 | 7 |
| B, bge-small | 2 | 3 | 6 | 8 |

Per-question table, failure analysis, downloads and checks: [docs/details.md](docs/details.md).

## Limits

- 10 questions is small: one question is 10 percentage points, so the tie says nothing general.
- I wrote the questions knowing the texts, which favours A. Questions from people who have not read the law would likely favour B.
- The texts have no later amendments. q05 reflects the 2022 text, which may differ from the law today.
- The hit rates cover retrieval only. `ask.py` answers are not scored.

## Data and licence

| Key | Text | CELEX | Words |
|---|---|---|---|
| `taxonomy` | Regulation (EU) 2020/852, the EU Taxonomy Regulation | 32020R0852 | 18,379 |
| `csrd` | Directive (EU) 2022/2464, the Corporate Sustainability Reporting Directive | 32022L2464 | 39,408 |
| `esrs` | Commission Delegated Regulation (EU) 2023/2772, ESRS set 1 | 32023R2772 | 130,571 |

EU legal texts: © European Union, https://eur-lex.europa.eu. Unless otherwise specified, EUR-Lex allows re-use for commercial and non-commercial purposes under the Commission's reuse policy (Decision 2011/833/EU): https://eur-lex.europa.eu/content/legal-notice/legal-notice.html (checked 2026-11-19). Only the Official Journal texts are authentic. This tool is not legal advice.

Embedding model: `BAAI/bge-small-en-v1.5` by BAAI, run through [fastembed](https://github.com/qdrant/fastembed). Model card: https://huggingface.co/BAAI/bge-small-en-v1.5

Code: MIT licence, see `LICENSE`.
