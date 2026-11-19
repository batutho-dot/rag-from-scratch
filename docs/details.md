# Details

Part 1 of the series, the starting point. Next: [rag-eval](https://github.com/batutho-dot/rag-eval) measures this pipeline end to end.

## Question and answer

**Question:** how often does a simple search engine return the passage of EU sustainability reporting law that answers a plain-English question? Does a small embedding model beat word counting written by hand in numpy?

**Answer (10 questions, k = 5):** both versions score 6 / 10 (README), on different questions (table below). The scores measure retrieval only. `ask.py` adds the answering half with Claude, tried on 3 questions and not scored.

## Terms

- **Chunk.** A 200-word piece of text, overlapping the next by 50, so a sentence cut at one edge is whole in the neighbour. The three texts give 1,257 chunks.
- **Embedding.** A list of numbers for each chunk and question. Version A: one number per vocabulary word (TF-IDF, high when a word is frequent in this chunk and rare elsewhere). Version B: 384 numbers from a small trained model that puts similar meanings close together, even with no shared words.
- **Top-k.** Compare the question with every chunk by cosine similarity (how closely two vectors point the same way) and return the k most similar. Here k = 5.
- **RAG.** These steps, then a language model that answers from the returned chunks only. A chunk outside the top k never reaches it, so I measure retrieval first.

Diagrams and the line of each box: [docs/architecture.md](architecture.md).

## Run it

`run.py` downloads the three texts from EUR-Lex into `data/` (about 6 MB of HTML), builds and scores both indexes, prints each question with the rank of its answer, and writes `results/latest.json`. The first run also downloads the embedding model (quantised ONNX, about 65 MB, cached by `fastembed`). No GPU, no API key, no cost.

- `python -m rag.evaluate --only a` runs version A alone in under a second.
- `python verify.py` checks the code after a change (about 1 second; `--full` adds version B, about 2 minutes) and ends with `verify: PASS` or `verify: FAIL at <step>`. `verify-features/` lists what each step covers.

On Windows the model download may print a symlink warning. It is harmless.

The texts are the Official Journal versions, fetched from `https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:<number>`. They are not stored in this repo.

## The test set

`questions.json` holds 10 questions: 3 on the Taxonomy, 4 on the CSRD, 3 on the ESRS. Each has the exact words of the law that answer it (`evidence`) and where they sit (`where`). A **hit** is any of the top 5 chunks containing the evidence phrase. The script first checks that every phrase exists in the downloaded text, so a broken test fails loudly instead of scoring zero.

| # | Question | A rank | B rank |
|---|---|---|---|
| q01 | What are the environmental objectives of the EU Taxonomy? | >10 | 3 |
| q02 | Which international guidelines do the Taxonomy's minimum safeguards refer to? | 1 | 1 |
| q03 | When is an economic activity considered to do significant harm to climate change mitigation? | 1 | 1 |
| q04 | What two kinds of information must a company's sustainability reporting cover under the CSRD? | >10 | >10 |
| q05 | From which financial year must large public-interest entities with more than 500 employees report under the CSRD? | 3 | 9 |
| q06 | By what date must the Commission adopt standards for reasonable assurance of sustainability reporting? | 1 | 5 |
| q07 | In what digital format does the management report with the sustainability statement have to be prepared? | >10 | >10 |
| q08 | How long is the medium-term time horizon in the ESRS? | 1 | 5 |
| q09 | Which information must a company always disclose, whatever its materiality assessment finds? | 1 | 8 |
| q10 | Which ESRS disclosure requirement asks for a company's Scope 1, 2 and 3 greenhouse gas emissions? | 6 | 4 |

Hit rate at other depths: [README](../README.md).

A is sharper when the question repeats the law's words ("minimum safeguards", "medium-term time horizon"): 5 of 10 at rank 1. B finds paraphrased questions A misses (q01, q10) but rarely ranks the answer first: 2 of 10 at rank 1.

## What it gets wrong

**q07 (digital format): both miss, partly the test's fault.** Version A's first result is chunk `csrd#0078`, a recital saying undertakings "should therefore be required to prepare their management report in the electronic reporting format specified in Article 3 of Commission Delegated Regulation (EU) 2019/815". That answers the question. The test accepts only the operative article (new Article 29d(1)), so it scores a miss.

**q04 (two kinds of information): both miss, a real failure.** The answer is the CSRD's "double materiality" sentence: the company's impacts on sustainability matters, and how sustainability matters affect the company. The question says "two kinds of information", which the law does not, and the ESRS discusses impacts and materiality in hundreds of places. ESRS is 871 of the 1,257 chunks (69 percent), so it crowds the top for both versions.

Other limits: one accepted passage per question (see q07). Fixed 200-word chunks ignore article boundaries, and the ESRS tables are flattened to plain lines.

## Ask it a question

`ask.py` is the full RAG loop. It retrieves the top 5 chunks (version B by default), sends only those to Claude with the instruction to answer from them alone, cite the chunk ids, and reply "not in the documents" if they do not hold the answer. It prints the answer, the chunks, and the tokens and cost, and appends each call to `results/ask_log.jsonl`.

It needs an Anthropic API key in `ANTHROPIC_API_KEY` (on Windows a saved user variable works). After `python run.py` has downloaded the texts:

```bash
python ask.py "What are the environmental objectives of the EU Taxonomy?"
python ask.py -i                     # ask several questions; empty line quits
python ask.py --model sonnet "..."   # Claude Sonnet 5.5 instead of Claude Haiku 5.5
python ask.py --retriever a "..."    # TF-IDF retrieval instead of embeddings
```

The first call with version B embeds all chunks (about 3 minutes on a laptop CPU) and caches them in `data/cache/`. Later calls start in about a second.

Tried on 3 of the 10 questions with Claude Haiku 5.5 (2026-11-19, effort `low`):

| Question | Retrieved the answer? | Answer | Tokens in / out | Cost |
|---|---|---|---|---|
| q01 environmental objectives | yes (`taxonomy#0058`, Article 9) | Right: all six objectives, cited | 2,380 / 162 | $0.00032 |
| q04 two kinds of information (CSRD) | no | Wrong: answered "forward-looking and retrospective, qualitative and quantitative" from a recital, instead of the double materiality pair; flagged that the passage only says "should" | 2,380 / 461 | $0.00047 |
| q07 digital format | the recital, not the article | Right: the format in Article 3 of Delegated Regulation (EU) 2019/815, cited | 2,421 / 124 | $0.00030 |

Total for the 3 calls: $0.00109, at Anthropic's list prices of $0.10 per million input tokens and $0.50 per million output tokens for Claude Haiku 5.5 (model table dated 2026-11-03). Output tokens include hidden reasoning.

q04 shows the main risk of RAG. When retrieval misses, the model does not say "not in the documents". It answers a nearby question from whatever passage it got. The next project measures that.

## Next

A test set with questions the texts cannot answer and several accepted passages per question, then cited answers from a language model.
