"""Download the latest 10-K filing for each company from SEC EDGAR."""

import json
import os
import time
from pathlib import Path 

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "JPM", "BAC", "GS", "V",
    "JNJ", "PFE", "UNH",
    "KO", "PG", "WMT", "NKE",
    "XOM", "CVX", "CAT", "BA",
]

# Ticker now maps to a new holding company with no 10-K yet
# Use the prior registrant
CIK_OVERRIDES = {"XOM": 34088}

REQUEST_DELAY = 0.2   # seconds; SEC allows up to 10 requests per second


# Raises a clear, actionable error if the environment variable is missing
def get_headers():
    user_agent = os.environ.get("SEC_USER_AGENT")
    if not user_agent:
        raise RuntimeError(
            'SEC_USER_AGENT is not set. Example: export SEC_USER_AGENT="Your Name you@example.com"'
        )
    return {"User-Agent": user_agent}


# Every request now goes through this one function, so the delay and error check can't be forgotten
def get(url, headers):
    time.sleep(REQUEST_DELAY)
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp


def load_ticker_map(headers):
    data = get("https://www.sec.gov/files/company_tickers.json", headers).json()
    return {item["ticker"]: item["cik_str"] for item in data.values()}


def find_10k_in_batch(batch):
    if "10-K" not in batch["form"]:
        return None
    i = batch["form"].index("10-K")
    return {
        "filing_date": batch["filingDate"][i],
        "accession": batch["accessionNumber"][i],
        "primary_doc": batch["primaryDocument"][i],
    }


def find_latest_10k(cik, headers):
    url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
    filings = get(url, headers).json()["filings"]

    result = find_10k_in_batch(filings["recent"])
    if result:
        return result

    for extra in filings.get("files", []):
        batch = get(f"https://data.sec.gov/submissions/{extra['name']}", headers).json()
        result = find_10k_in_batch(batch)
        if result:
            return result

    return None

def download_latest_10k(ticker, cik, headers):
    filing = find_latest_10k(cik, headers)
    if filing is None:
        print(f"{ticker}: no 10-K found")
        return None

    out_path = DATA_DIR / f"{ticker}_10-K_{filing['filing_date']}.html"
    if not out_path.exists():
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{cik}/"
            f"{filing['accession'].replace('-', '')}/{filing['primary_doc']}"
        )
        out_path.write_text(get(url, headers).text, encoding="utf-8")

    return {
        "ticker": ticker,
        "cik": cik,
        **filing,
        "path": str(out_path.relative_to(PROJECT_ROOT)),
    }


def main():
    headers = get_headers()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ticker_to_cik = load_ticker_map(headers)

    manifest = []
    for ticker in TICKERS:
        cik = CIK_OVERRIDES.get(ticker, ticker_to_cik[ticker])
        info = download_latest_10k(ticker, cik, headers)
        if info:
            manifest.append(info)
            print(f"{ticker}: {info['filing_date']}")

    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n{len(manifest)}/{len(TICKERS)} downloaded")


if __name__=="__main__":
    main()