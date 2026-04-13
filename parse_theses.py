"""
parse_theses.py

Parses thesis PDFs into paragraph-level CSVs and a JSON index file.
Uses pdfplumber for text extraction with position-based paragraph detection.

Handles:
- Running headers/footers (stripped)
- Section titles (dropped from paragraphs, but "Abstract" section is captured)
- Single-column and two-column page layouts
"""

import csv
import json
import re
import statistics
from pathlib import Path

import pdfplumber

OUTPUT_DIR = Path(__file__).parent

THESIS_CONFIGS = [
    {
        "pdf_path": Path("/Users/sha/Downloads/drive-download-20260412T184509Z-3-001/Tong_Sha.pdf"),
        "author": "Tong Sha",
        "thesis_title": "THE DISPROPORTIONATE INFLUENCE OF ONLINE REVIEWS",
        "output_csv": OUTPUT_DIR / "tong_sha_paragraphs.csv",
    },
    {
        "pdf_path": Path("/Users/sha/Downloads/drive-download-20260412T184509Z-3-001/UrvaJoshi_Thesis2020May.pdf"),
        "author": "Urva Joshi",
        "thesis_title": "THE PARADOXES OF FAST FASHION: CAN WE OVERCOME A WICKED PROBLEM?",
        "output_csv": OUTPUT_DIR / "urva_joshi_paragraphs.csv",
    },
]


# ── Header/footer detection ───────────────────────────────────────────────────

def detect_header_footer_bands(pdf):
    """
    Return (header_frac, footer_frac) — fractions of page height that are
    occupied by running headers/footers.  We sample the first several pages
    and look for text that sits in the top or bottom 10% of the page.
    """
    top_texts = {}   # text → count of pages it appears on near the top
    bot_texts = {}

    sample_pages = min(8, len(pdf.pages))
    for page in pdf.pages[:sample_pages]:
        h = page.height
        words = page.extract_words()
        for w in words:
            if w["top"] < h * 0.10:
                top_texts[w["text"]] = top_texts.get(w["text"], 0) + 1
            if w["bottom"] > h * 0.90:
                bot_texts[w["text"]] = bot_texts.get(w["text"], 0) + 1

    # Repeated text (≥3 pages) in those bands is a running header/footer
    header_words = {t for t, c in top_texts.items() if c >= 3}
    footer_words = {t for t, c in bot_texts.items() if c >= 3}
    return header_words, footer_words


def filter_words(words, page_height, header_words, footer_words,
                 header_frac=0.09, footer_frac=0.91):
    """Remove words that are in the header/footer bands."""
    kept = []
    for w in words:
        if w["top"] < page_height * header_frac:
            continue
        if w["bottom"] > page_height * footer_frac:
            continue
        # Also drop isolated header/footer repeated tokens wherever they appear
        if w["text"] in header_words or w["text"] in footer_words:
            # Only drop if they're lone occurrences in the band vicinity
            if w["top"] < page_height * 0.12 or w["bottom"] > page_height * 0.88:
                continue
        kept.append(w)
    return kept


# ── Column detection ──────────────────────────────────────────────────────────

def detect_columns(words, page_width):
    """
    Return (has_two_columns, split_x).
    Two-column layout is detected when there are substantial word groups on
    both sides of the page with a clear gap around the midpoint.
    """
    if not words:
        return False, page_width / 2

    mid = page_width / 2
    gap = 30  # minimum gap around midpoint to consider multi-column

    left_words  = [w for w in words if w["x1"] < mid - gap]
    right_words = [w for w in words if w["x0"] > mid + gap]

    # Need meaningful content on both sides
    if len(left_words) < 5 or len(right_words) < 5:
        return False, mid

    # Check that left words don't extend into the right zone
    max_left_x1 = max(w["x1"] for w in left_words)
    min_right_x0 = min(w["x0"] for w in right_words)

    if min_right_x0 - max_left_x1 > 20:
        split_x = (max_left_x1 + min_right_x0) / 2
        return True, split_x

    return False, mid


# ── Line grouping ─────────────────────────────────────────────────────────────

def group_into_lines(words, y_tolerance=3):
    """
    Group words into lines by clustering on their `top` y-coordinate.
    Returns a list of lines, each line being a list of word dicts sorted by x0.
    """
    if not words:
        return []

    # Sort by top, then x0
    words = sorted(words, key=lambda w: (round(w["top"] / y_tolerance), w["x0"]))

    lines = []
    current_line = [words[0]]
    current_top = words[0]["top"]

    for w in words[1:]:
        if abs(w["top"] - current_top) <= y_tolerance:
            current_line.append(w)
        else:
            lines.append(sorted(current_line, key=lambda w: w["x0"]))
            current_line = [w]
            current_top = w["top"]

    if current_line:
        lines.append(sorted(current_line, key=lambda w: w["x0"]))

    return lines


def line_text(line):
    return " ".join(w["text"] for w in line)


def median_word_height(line):
    heights = [w["height"] for w in line]
    return statistics.median(heights) if heights else 0


# ── Paragraph extraction ──────────────────────────────────────────────────────

def lines_to_paragraphs(lines, body_height_threshold, header_height_ratio=1.4):
    """
    Convert a list of lines into:
      - paragraphs: list of (text, is_section_header) tuples

    A line is a section header if its median word height exceeds
    body_height_threshold * header_height_ratio.

    Paragraph breaks are detected by gaps > 1.8× median line spacing.
    """
    if not lines:
        return []

    # Compute median line gap
    tops = [line[0]["top"] for line in lines]
    gaps = [tops[i + 1] - tops[i] for i in range(len(tops) - 1)]
    median_gap = statistics.median(gaps) if gaps else 12
    break_threshold = median_gap * 1.8

    paragraphs = []
    current_para_lines = []
    prev_top = None

    def flush_para():
        if current_para_lines:
            text = " ".join(current_para_lines)
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) >= 30:
                paragraphs.append((text, False))  # (text, is_header)
            current_para_lines.clear()

    for i, line in enumerate(lines):
        mh = median_word_height(line)
        is_header = mh >= body_height_threshold * header_height_ratio
        txt = line_text(line)

        # Detect paragraph break by gap from previous line
        if prev_top is not None:
            gap = line[0]["top"] - prev_top
            if gap > break_threshold:
                flush_para()

        if is_header:
            flush_para()
            paragraphs.append((txt.strip(), True))  # is_header=True
        else:
            current_para_lines.append(txt)

        prev_top = line[0]["top"]

    flush_para()
    return paragraphs


# ── Body height estimation ────────────────────────────────────────────────────

def estimate_body_height(pdf, header_words, footer_words):
    """
    Estimate the median word height for body text across the whole document.
    """
    all_heights = []
    for page in pdf.pages[:min(10, len(pdf.pages))]:
        words = filter_words(
            page.extract_words(), page.height, header_words, footer_words
        )
        all_heights.extend(w["height"] for w in words)

    if not all_heights:
        return 12.0

    # Body text is the most common height; ignore outliers (very large = headers)
    # Use mode of rounded values
    rounded = [round(h, 1) for h in all_heights]
    from collections import Counter
    most_common = Counter(rounded).most_common(1)[0][0]
    return most_common


# ── Main parsing function ─────────────────────────────────────────────────────

def parse_thesis(config):
    """
    Parse a single thesis PDF.

    Returns:
        paragraphs: list of body paragraph strings (abstract excluded)
        abstract:   the abstract text, or "" if not found
    """
    pdf_path = config["pdf_path"]
    paragraphs = []
    abstract_parts = []
    in_abstract = False
    abstract_captured = False

    with pdfplumber.open(pdf_path) as pdf:
        header_words, footer_words = detect_header_footer_bands(pdf)
        body_height = estimate_body_height(pdf, header_words, footer_words)

        for page in pdf.pages:
            pw = page.width
            ph = page.height

            words = filter_words(
                page.extract_words(), ph, header_words, footer_words
            )

            has_cols, split_x = detect_columns(words, pw)

            if has_cols:
                columns = [
                    [w for w in words if w["x1"] <= split_x],
                    [w for w in words if w["x0"] > split_x],
                ]
            else:
                columns = [words]

            for col_words in columns:
                if not col_words:
                    continue
                lines = group_into_lines(col_words)
                items = lines_to_paragraphs(lines, body_height)

                for text, is_header in items:
                    if is_header:
                        section = text.strip().lower()
                        if section == "abstract":
                            in_abstract = True
                            abstract_captured = False
                        elif in_abstract and not abstract_captured:
                            # Moving to the next section — abstract is done
                            in_abstract = False
                            abstract_captured = True
                        # Don't add section titles to paragraphs list
                    else:
                        if in_abstract:
                            abstract_parts.append(text)
                        else:
                            paragraphs.append(text)

    abstract = " ".join(abstract_parts).strip()
    # Normalize whitespace in abstract
    abstract = re.sub(r"\s+", " ", abstract)
    return paragraphs, abstract


# ── CSV / JSON output ─────────────────────────────────────────────────────────

def write_csv(paragraphs, author, thesis_title, output_path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["author", "thesis_title", "paragraph"])
        for para in paragraphs:
            writer.writerow([author, thesis_title, para])


def main():
    index = []

    for config in THESIS_CONFIGS:
        print(f"Parsing: {config['pdf_path'].name} ...")
        paragraphs, abstract = parse_thesis(config)

        write_csv(
            paragraphs,
            config["author"],
            config["thesis_title"],
            config["output_csv"],
        )
        print(f"  → {len(paragraphs)} paragraphs written to {config['output_csv'].name}")
        print(f"  → Abstract: {abstract[:80]!r}{'...' if len(abstract) > 80 else ''}")

        index.append({
            "author": config["author"],
            "thesis_title": config["thesis_title"],
            "abstract": abstract,
            "num_paragraphs": len(paragraphs),
            "csv_path": str(config["output_csv"]),
        })

    json_path = OUTPUT_DIR / "theses_index.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)
    print(f"\nIndex written to {json_path}")


if __name__ == "__main__":
    main()
