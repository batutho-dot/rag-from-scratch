"""One command: download the EU texts, build both indexes, score both on the 10 questions.

    python run.py
"""

from rag import download, evaluate

if __name__ == "__main__":
    download.main()
    evaluate.main()
