# Download and clean the EU texts

- **For the user:** fetches the Taxonomy Regulation, CSRD and ESRS set 1 from EUR-Lex and
  turns the HTML into plain text, one line per paragraph or table cell.
- **How to reach it:** `python -m rag.download`, or `python run.py`. Writes `data/raw/*.html`
  and `data/text/*.txt` (git-ignored).
- **How verify drives it:** `units` feeds `html_to_text` a fixed snippet. `real-run-A`
  copies `data/raw/` to a temporary folder and runs `download.main(<tmp>)` on it.
- **End state that proves it worked:** the toy snippet gives exactly `a b\nc`; the real
  texts give 1,257 chunks.
- **Gotchas:** with no `data/raw/`, verify downloads about 6 MB from EUR-Lex, so it needs the
  network. If EUR-Lex changes its HTML, the chunk count changes and `real-run-A` fails: that
  is the intended alarm, not a reason to loosen the check.
