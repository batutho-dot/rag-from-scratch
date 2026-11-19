# Feature map

What this repo does for its user, and how `verify.py` drives each part.

| Feature | File | Verify step |
|---|---|---|
| Download and clean the EU texts | [download.md](download.md) | `units`, `real-run-A` |
| Version A: TF-IDF search in numpy | [version-a.md](version-a.md) | `units`, `real-run-A` |
| Version B: embedding search | [version-b.md](version-b.md) | `real-run-B` (`--full` only) |
| Scoring against the 10 questions | [evaluate.md](evaluate.md) | `questions`, `real-run-A`, `real-run-B` |

The one-command entry point `run.py` calls download, then evaluate; it is covered through
those two features.
