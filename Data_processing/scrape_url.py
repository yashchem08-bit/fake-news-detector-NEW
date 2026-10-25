"""
Member A — Data Engineer (bonus utility)
==========================================
Fetches an article's main text from a URL, so users can paste a link
instead of pasting raw article text. Imported by the backend's
/predict-url endpoint (Member C).

This is a best-effort extractor: it pulls all <p> tags after
stripping out obvious non-article elements (scripts, nav, footer...).
It will not work perfectly on every site, but covers most news pages.
"""

import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; FakeNewsDetectorBot/1.0)"}


def fetch_article_text(url: str, timeout: int = 10) -> str:
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()

    paragraphs = soup.find_all("p")
    text = " ".join(p.get_text(strip=True) for p in paragraphs)
    text = " ".join(text.split())  # collapse whitespace

    if len(text) < 200:
        raise ValueError(
            "Could not extract enough article text from this URL. "
            "Try pasting the article text directly instead."
        )

    return text


if __name__ == "__main__":
    # Quick manual test
    test_url = "https://example.com"
    print(fetch_article_text(test_url)[:500])
