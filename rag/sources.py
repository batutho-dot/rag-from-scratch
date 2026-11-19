"""The public EU texts this project searches, and where they come from.

All three are published on EUR-Lex. EUR-Lex's legal notice says the legal documents it
publishes may be re-used, for commercial or non-commercial purposes, unless otherwise
specified, under Commission Decision 2011/833/EU:
https://eur-lex.europa.eu/content/legal-notice/legal-notice.html
The repo still ships a download script rather than the documents, so every run uses the
official text.
"""

SOURCES = {
    "taxonomy": {
        "celex": "32020R0852",
        "title": "Regulation (EU) 2020/852 (EU Taxonomy Regulation)",
    },
    "csrd": {
        "celex": "32022L2464",
        "title": "Directive (EU) 2022/2464 (Corporate Sustainability Reporting Directive, CSRD)",
    },
    "esrs": {
        "celex": "32023R2772",
        "title": "Commission Delegated Regulation (EU) 2023/2772 (European Sustainability Reporting Standards, ESRS set 1)",
    },
}


def url(celex: str) -> str:
    return f"https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:{celex}"
