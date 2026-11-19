"""Ask Claude a question about the EU texts, answered only from the retrieved passages.

    python ask.py "What are the Taxonomy's environmental objectives?"
    python ask.py -i                      interactive: one question per line, empty line quits
    python ask.py --model sonnet "..."    Claude Sonnet 5.5 instead of Claude Haiku 5.5
    python ask.py --retriever a "..."     TF-IDF retrieval (version A) instead of embeddings (B)

Steps: retrieve the top-k chunks (k = 5), send only those to Claude with an instruction to
answer from them alone, print the answer, the chunks sent, and the tokens and cost of the
call. Every call is appended to results/ask_log.jsonl (no API key is ever written).

Needs an Anthropic API key in the ANTHROPIC_API_KEY environment variable. On Windows, a
key saved as a user environment variable is read from the registry if the variable is not
already in this process.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rag.chunk import load_chunks  # noqa: E402

MODELS = {
    # id, input USD per million tokens, output USD per million tokens.
    # Anthropic list prices from its model table dated 2026-11-03; check before relying on them.
    # The Haiku price holds for prompts up to 100K tokens; these prompts are about 2K.
    "haiku": ("claude-haiku-5-5", 0.10, 0.50),
    "sonnet": ("claude-sonnet-5-5", 2.00, 10.00),
}
K = 5
NOT_FOUND = "not in the documents"
SYSTEM = (
    "You answer questions about EU sustainability law using only the passages given in the "
    "user message. Each passage has an id. Rules:\n"
    "1. Use only facts stated in the passages. Do not use outside knowledge.\n"
    "2. After each claim, cite the passage id in square brackets, for example [csrd#0078].\n"
    f'3. If the passages do not contain the answer, reply exactly "{NOT_FOUND}" and nothing else.\n'
    "4. Keep the answer short: at most a few sentences or a short list."
)


def load_key() -> None:
    """Put the user-level ANTHROPIC_API_KEY into this process if it is missing. Never prints it."""
    if os.environ.get("ANTHROPIC_API_KEY") or sys.platform != "win32":
        return
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as reg:
            value, _ = winreg.QueryValueEx(reg, "ANTHROPIC_API_KEY")
    except OSError:
        return
    if value:
        os.environ["ANTHROPIC_API_KEY"] = value


def build_retriever(name: str, chunks):
    texts = [c.text for c in chunks]
    if name == "a":
        from rag.version_a import TfidfIndex

        return TfidfIndex(texts)
    from rag.version_b import EmbeddingIndex

    return EmbeddingIndex(texts, cache=ROOT / "data" / "cache" / "bge_small_embeddings.npy")


def ask(client, model_key: str, question: str, passages) -> dict:
    model, price_in, price_out = MODELS[model_key]
    context = "\n\n".join(f'<passage id="{c.id}">\n{c.text}\n</passage>' for c in passages)
    request = dict(
        model=model,
        max_tokens=2000,
        system=SYSTEM,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": f"{context}\n\nQuestion: {question}"}],
    )
    t = time.perf_counter()
    if model_key == "sonnet":
        # Sonnet 5.5: if a safety classifier declines, the API reruns on a fallback model.
        response = client.beta.messages.create(
            betas=["server-side-fallback-2026-07-01"], fallbacks="default", **request
        )
    else:
        response = client.messages.create(**request)
    seconds = time.perf_counter() - t
    if response.stop_reason == "refusal":
        answer = "(the model declined to answer)"
    else:
        answer = "".join(b.text for b in response.content if b.type == "text").strip()
    u = response.usage
    cost = u.input_tokens * price_in / 1e6 + u.output_tokens * price_out / 1e6
    return {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": response.model,
        "question": question,
        "chunks": [c.id for c in passages],
        "answer": answer,
        "stop_reason": response.stop_reason,
        "input_tokens": u.input_tokens,
        "output_tokens": u.output_tokens,
        "cost_usd": round(cost, 6),
        "seconds": round(seconds, 1),
    }


def show(record: dict, passages) -> None:
    print(f"\nAnswer ({record['model']}):\n{record['answer']}\n")
    print("Passages sent:")
    for c in passages:
        print(f"  [{c.id}] {c.text[:150]}...")
    print(
        f"\nTokens: {record['input_tokens']:,} in, {record['output_tokens']:,} out "
        f"(output includes the model's hidden reasoning). Cost: ${record['cost_usd']:.5f}. "
        f"Time: {record['seconds']} s."
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("question", nargs="?")
    p.add_argument("-i", "--interactive", action="store_true")
    p.add_argument("--model", choices=sorted(MODELS), default="haiku")
    p.add_argument("--retriever", choices=["a", "b"], default="b")
    p.add_argument("-k", type=int, default=K)
    args = p.parse_args()
    if not args.question and not args.interactive:
        p.error("give a question in quotes, or -i")

    text_dir = ROOT / "data" / "text"
    if not any(text_dir.glob("*.txt")):
        print("No texts yet: run `python run.py` (or `python -m rag.download`) first.")
        return 1
    load_key()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set. Set it as an environment variable first.")
        return 1

    import anthropic

    client = anthropic.Anthropic()
    chunks = load_chunks(text_dir)
    print(f"Loading retriever {args.retriever.upper()} over {len(chunks)} chunks...", flush=True)
    retriever = build_retriever(args.retriever, chunks)
    log = ROOT / "results" / "ask_log.jsonl"
    log.parent.mkdir(exist_ok=True)

    questions = [args.question] if args.question else []
    while True:
        if not questions:
            if not args.interactive:
                break
            try:
                line = input("\nQuestion (empty line to quit): ").strip()
            except EOFError:
                break
            if not line:
                break
            questions.append(line)
        question = questions.pop(0)
        passages = [chunks[i] for i, _ in retriever.search(question, args.k)]
        try:
            record = ask(client, args.model, question, passages)
        except anthropic.AuthenticationError:
            print("The API key was rejected (authentication error).")
            return 1
        except anthropic.APIStatusError as e:
            print(f"API error {e.status_code}: {e.message}")
            return 1
        except anthropic.APIConnectionError:
            print("Could not reach the Anthropic API (network error).")
            return 1
        record["retriever"] = args.retriever
        record["k"] = args.k
        show(record, passages)
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
