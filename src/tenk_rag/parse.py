"""Parse 10-K HTML filings into ordered text and table blocks."""

import json
import re
import warnings
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# iXBRL filings are XHTML; parsing them as HTML is intended
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

NUMBER = re.compile(r"^\(?-?[\d,]*\.?\d+\)?%?$")  # e.g. 6,617 / 39.3% / (321)
YEAR = re.compile(r"^(19|20)\d{2}$")              # e.g. 2026
TABLE_MARKER = re.compile(r"\[\[TABLE_(\d+)\]\]")  # placeholder left where a table was
BLOCK_TAGS = ["div", "p", "br", "li", "h1", "h2", "h3", "h4"]  # tags that end a paragraph
# (A|B) matches either phrase. New footer formats can be added with another |
PAGE_FOOTER = re.compile(
    r"(Form 10-K|Annual Report)\s+([ivxlc]+|\d{1,3})$", re.IGNORECASE
)  # e.g. "Pfizer Inc. 2025 Form 10-K 12", "2025 Annual Report 3"


### Table parsing 

def clean_row(cells):
	"""Drop empty and '$' cells, and attch stray '%' or ')' to the previous cell."""
	out = []
	for c in cells:
		c = re.sub(r"\(\s+", "(", c.strip())   # "( 321 )" -> "(321)"
		c = re.sub(r"\s+\)", ")", c)
		if c in ("", "$"):
			continue
		if c in ("%", ")", ")%") and out:
			out[-1] += c
			continue
		out.append(c)
	return out


def is_section(row):
	"""A single-cell row ending with ':' is a section title, e.g. 'Net sales:'."""
	return len(row) == 1 and row[0].endswith(":")


def has_amounts(row):
	"""True if any value cell (after the label) holds an amount; years don't count."""
	return any(NUMBER.match(c) and not YEAR.match(c) for c in row[1:])


def looks_like_header(row):
    """A row with no amounts can be a header, e.g. 'Periods | Shares | Price'."""
    return not has_amounts(row)


def table_rows(table):
	"""Extract cleaned rows from an HTML table."""
	rows = []
	for tr in table.find_all("tr"):
		cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
		row = clean_row(cells)
		if row:
			rows.append(row)
	return rows


def pop_caption(rows):
	"""Remove and return a long single-cell first row: a table title, not data."""
	if rows and len(rows[0]) == 1 and len(rows[0][0]) > 40:
		return rows.pop(0)[0]
	return ""


def table_to_markdown(rows):
	"""Convert cleaned table rows into a markdown table string."""
	width = max(len(r) for r in rows)   # data rows (label + values) are the widest
	n_cols = width - 1   # value columns, excluding the label column

	# Header rule 1: leading rows without a label cell are shorter than data rows
	header = [""] * n_cols
	corner = ""
	while rows and len(rows[0]) < width and not is_section(rows[0]):
		hr = rows.pop(0)
		if len(hr) == 1:   # a title spanning all columns, e.g. "Years Ended"
			hr = hr * n_cols
		hr = [""] * (n_cols - len(hr)) + hr   # right-align with the value columns
		header = [f"{h} {x}".strip() for h, x in zip(header, hr)]   # merge multi-row headers

	# Header rule 2: a full-width first row with no amounts, e.g."Periods | Shares | Price"
	if (
		not any(header)
		and len(rows) > 1
		and not is_section(rows[0])
		and looks_like_header(rows[0])
	):
		hr = rows.pop(0)
		corner = hr[0]
		header = hr[1:] + [""] * (n_cols - len(hr[1:]))

	# Body: prefix rows with their section so repeated labels stay distinct
	# e.g. "Products" -> "Cost of sales - Products"
	body = []
	section = ""
	for r in rows:
		if is_section(r):
			section = r[0].rstrip(":")
			continue
		label = r[0]
		if section and not label.startswith("Total"):
			label = f"{section} - {label}"
		body.append([label] + r[1:] + [""] * (width - len(r)))
		if r[0].startswith("Total"):   # a total row closes the section 
			section = ""

	lines = [f"| {corner} | " + " | ".join(header) + " |", "|" + " --- |" * width]
	lines += ["| " + " | ".join(r) + " |" for r in body]
	return "\n".join(lines)


### Document parsing

def parse_10k(html, context_lines=2):
	"""Split a 10-K into ordered text and table blocks.
	
	Each table block carries the text lines just before it as context,
	which often holds the years and units (e.g. "dollars in millions").
	"""
	soup = BeautifulSoup(html, "lxml")

	# Remove hidden XBRL data that readers never see
	for tag in soup.find_all(style=lambda s: s and "display:none" in s.replace(" ", "")):
		tag.decompose()

	# Classify each table: drop, turn into text, or keep as a markdown table
	tables = []   # (caption, markdown) paris
	for table in soup.find_all("table"):
		rows = table_rows(table)
		caption = pop_caption(rows)   # a title row, if any, is moved out first

		if not rows:
			if caption:   # the "table" was only a title
				table.replace_with(f"\n{caption}\n")
			else:   # layout-only table with no text
				table.decompose()
			continue

		if len(rows) == 1:   # one-row "tables" are often not data
			line = " ".join(rows[0])
			if PAGE_FOOTER.search(line):
				table.decompose()
				continue
			if not has_amounts(rows[0]):
				table.replace_with(f"\n{caption}\n{line}\n")
				continue

		table.replace_with(f"\n[[TABLE_{len(tables)}]]\n")
		tables.append((caption, table_to_markdown(rows)))

	# Break lines only at paragraph ends, so inline <span>s stay in one sentence
	for tag in soup.find_all(BLOCK_TAGS):
		tag.append("\n")
	body = soup.body or soup   # skip <head>, e.g. the <title> text
	lines = [re.sub(r"\s+", " ", line).strip() for line in body.get_text().splitlines()]
	lines = [line for line in lines if line]

	# Walk lines in order: text accumulates until a table marker appears
	blocks = []
	buffer = []   # text lines for the current text block
	recent = []   # all text lines so far, used for table context
	for line in lines:
		m = TABLE_MARKER.fullmatch(line)
		if m:
			if buffer:
				blocks.append({"type": "text", "content": "\n".join(buffer)})
				buffer = []
			caption, md = tables[int(m.group(1))]
			context = recent[-context_lines:] + ([caption] if caption else [])
			blocks.append({
				"type": "table",
				"context": " ".join(context),
				"content": md
			})
		else:
			buffer.append(line)
			recent.append(line)

	if buffer:
		blocks.append({"type": "text", "content": "\n".join(buffer)})
	return blocks


### Batch processing

def main():
	"""Parse every filing in the manifest and save one JSON file per company."""
	manifest = json.loads((RAW_DIR / "manifest.json").read_text())
	PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

	failed = []
	for filing in manifest:
		ticker = filing["ticker"]
		try:
			html = (PROJECT_ROOT / filing["path"]).read_text(encoding="utf-8")
			blocks = parse_10k(html)
		except Exception as e:   # keep going so one bad filing doesn't stop the batch
			failed.append(ticker)
			print(f"{ticker}: FAILED ({type(e).__name__}: {e})")
			continue

		# Keep filing metadata with the blocks so answer can cite their source
		doc = {
			"ticker": ticker,
			"filing_date": filing["filing_date"],
			"accession": filing["accession"],
			"source": filing["path"],
			"blocks": blocks,
		}
		out_path = PROCESSED_DIR / f"{ticker}_10-K_{filing['filing_date']}.json"
		out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")

		n_tables = sum(b["type"] == "table" for b in blocks)
		print(f"{ticker}: {len(blocks)} blocks ({n_tables} tables)")

	print(f"\n{len(manifest) - len(failed)}/{len(manifest)} parsed")
	if failed:
		print("Failed:", failed)


if __name__=="__main__":
	main()