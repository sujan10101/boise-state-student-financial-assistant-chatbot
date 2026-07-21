"""
Crawl Boise State Student Financial Services pages and save raw content.
"""
import os
import json
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

SEED_URLS = [
    "https://www.boisestate.edu/sfs/payments/",
    "https://www.boisestate.edu/sfs/",
    "https://www.boisestate.edu/sfs/tuition-fees/",
    "https://www.boisestate.edu/sfs/financial-wellness/",
    "https://www.boisestate.edu/sfs/faqs/",
]

ALLOWED_PREFIX = "https://www.boisestate.edu/sfs/"
OUTPUT_FILE = "data/raw_documents.json"


def fetch_page(url: str) -> str | None:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; RAG-Crawler/1.0)"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"  [WARN] Could not fetch {url}: {e}")
        return None


def extract_text_and_links(html: str, base_url: str) -> tuple[str, list[str]]:
    soup = BeautifulSoup(html, "html.parser")

    # Remove nav, footer, script, style noise
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    main = soup.find("main") or soup.find("div", class_="content") or soup.body
    text = main.get_text(separator="\n", strip=True) if main else soup.get_text(separator="\n", strip=True)

    # Collapse blank lines
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    clean_text = "\n".join(lines)

    # Discover internal links
    links = []
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        parsed = urlparse(href)
        href_clean = parsed.scheme + "://" + parsed.netloc + parsed.path
        if href_clean.startswith(ALLOWED_PREFIX) and href_clean not in links:
            links.append(href_clean)

    return clean_text, links


def crawl(seed_urls: list[str], max_pages: int = 30) -> list[dict]:
    visited: set[str] = set()
    queue = list(seed_urls)
    documents = []

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        print(f"Fetching: {url}")
        html = fetch_page(url)
        if not html:
            continue

        text, links = extract_text_and_links(html, url)
        if len(text) > 100:
            documents.append({"url": url, "content": text})
            print(f"  Saved {len(text)} chars, found {len(links)} links")

        for link in links:
            if link not in visited and link not in queue:
                queue.append(link)

        time.sleep(0.5)

    return documents


def main():
    os.makedirs("data", exist_ok=True)
    print("=== Crawling Boise State SFS ===")
    docs = crawl(SEED_URLS, max_pages=30)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)
    print(f"\nDone. Saved {len(docs)} pages to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
