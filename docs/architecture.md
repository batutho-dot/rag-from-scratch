# How the code fits together

(a) shows which file uses which. (b) follows one question from text to answer. The tables give the line of each box.

## (a) Structure

Solid arrows are imports and calls, dotted arrows are files read or written. The three boxes on the right are outside this repo.

```mermaid
flowchart LR
    run["run.py<br/>download, then score"]
    ask["ask.py<br/>cited answer, cost"]
    verify["verify.py<br/>PASS or FAIL"]

    subgraph rag["rag/ package"]
        sources["sources.py<br/>3 CELEX numbers"]
        download["download.py<br/>HTML to plain text"]
        chunk["chunk.py<br/>200-word windows<br/>50-word overlap"]
        va["version_a.py<br/>TF-IDF, numpy"]
        vb["version_b.py<br/>bge-small, cache"]
        evaluate["evaluate.py<br/>hit rate at k"]
    end

    questions[("questions.json<br/>10 questions")]
    data[("data/<br/>HTML, text, cache")]
    results[("results/<br/>latest.json<br/>ask_log.jsonl")]

    eurlex["EUR-Lex<br/>HTTP download"]
    fastembed["fastembed<br/>BAAI/bge-small-en-v1.5<br/>CPU"]
    claude["Claude API<br/>Haiku 5.5 or Sonnet 5.5"]

    run --> download
    run --> evaluate
    download --> sources
    download --> eurlex
    download -.-> data
    evaluate --> chunk
    evaluate --> va
    evaluate --> vb
    evaluate -.-> questions
    evaluate -.-> results
    vb --> fastembed
    vb -.-> data
    chunk -.-> data
    ask --> chunk
    ask --> va
    ask --> vb
    ask --> claude
    ask -.-> results
    verify --> download
    verify --> chunk
    verify --> evaluate
    verify --> va
    verify --> vb
    verify -.-> questions
    verify -.-> results
```

| Box | Where | What it does |
|---|---|---|
| run.py | `run.py:9` | Calls `download.main()`, then `evaluate.main()`. |
| ask.py | `ask.py:126` | Builds a retriever, asks Claude, prints and logs each answer. |
| verify.py | `verify.py:120` | Runs the checks, stops at the first failure. |
| sources.py | `rag/sources.py:11` | Lists the three texts by CELEX number; `url()` at line 27 builds the EUR-Lex address. |
| download.py | `rag/download.py:46` | Downloads each text once into `data/raw/`, writes plain text to `data/text/`. |
| chunk.py | `rag/chunk.py:24` | Cuts a text into 200-word chunks overlapping by 50; `load_chunks` at line 33 does all three. |
| version_a.py | `rag/version_a.py:23` | Builds the TF-IDF matrix in numpy and searches it. |
| version_b.py | `rag/version_b.py:18` | Turns each chunk into 384 numbers with bge-small, saves and searches them. |
| evaluate.py | `rag/evaluate.py:50` | Builds both indexes, asks the 10 questions, writes `results/latest.json`. |
| questions.json | `questions.json` | The 10 questions, each with the exact words of the law that answer it. |
| EUR-Lex | `rag/download.py:53` | The only network call for the texts (`urllib.request.urlopen`). |
| fastembed | `rag/version_b.py:24` | Loads the embedding model, about 65 MB on the first run. |
| Claude API | `ask.py:88`, `ask.py:92` | Sonnet uses the beta call with fallback, Haiku the plain call. |

## (b) The path of one question

Only the split differs between version A and version B.

```mermaid
flowchart TD
    texts["3 EU texts<br/>188k words"]
    chunks["chunk_text<br/>200 words, 50-word overlap<br/>1,257 chunks"]
    question["your question"]

    subgraph A["Version A: word counts"]
        a1["tokenize<br/>drop stopwords"]
        a2["TF-IDF vector<br/>rare words weigh more"]
    end

    subgraph B["Version B: embedding model"]
        b1["add query prefix"]
        b2["bge-small vector<br/>384 numbers"]
    end

    cosine["cosine similarity<br/>one score per chunk"]
    topk["top-k<br/>keep best 5"]
    score["evaluate.py<br/>evidence phrase<br/>in the top 5?"]
    answer["ask.py, optional<br/>Claude reads 5 chunks<br/>cited answer or<br/>not in the documents"]

    texts --> chunks
    chunks --> a2
    chunks --> b2
    question --> a1
    question --> b1
    a1 --> a2
    b1 --> b2
    a2 --> cosine
    b2 --> cosine
    cosine --> topk
    topk --> score
    topk --> answer
```

| Box | Where | What it does |
|---|---|---|
| chunk_text | `rag/chunk.py:24` | Slides a 200-word window along the text, 150 words per step. |
| tokenize | `rag/version_a.py:19` | Lowercases, keeps letters and digits, drops a short stopword list. |
| TF-IDF vector | `rag/version_a.py:30`, `rag/version_a.py:42` | Weighs words by rarity across chunks, damps repeats with log, scales to length 1. |
| query prefix | `rag/version_b.py:41` | Puts the instruction sentence bge expects before the question. |
| bge-small vector | `rag/version_b.py:30` | 384 numbers per chunk, scaled to length 1. |
| cosine similarity | `rag/version_a.py:48`, `rag/version_b.py:42` | One matrix multiplication scores every chunk. |
| top-k | `rag/version_a.py:49`, `rag/version_b.py:43` | Sorts the scores, keeps the best k indexes. |
| evaluate.py | `rag/evaluate.py:30` | Records the rank of the first chunk holding the evidence phrase. |
| ask.py | `ask.py:75` | Sends the 5 chunks and the question to Claude with the answer-only-from-these rules (`ask.py:39`). |
