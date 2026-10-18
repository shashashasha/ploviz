#!/usr/bin/env python3
"""
Score thesis paragraphs against PLO manifestations using Pratt's pratt-jev decision model.

For each paragraph in a theses CSV (author, thesis_title, paragraph, ...), pratt-jev rates
every manifestation in a manifestations JSON on a 0-3 evidence scale:

    0 = no evidence, 1 = weak, 2 = moderate, 3 = strong

Outputs:
    <output>.csv        author, thesis_title, paragraph, PLO1_P1M01 ... (graded 0-3 counts)
    <output>_probs.csv  the same rows with raw scores and level probabilities, so the
                        counts can be recomputed at any threshold without re-scoring

A count is the highest level k where P(level >= k) >= threshold.

Usage:
    python3 thesis-scorer.py data/theses_counted.csv manifestations_new.json -o data/theses_scored.csv
    python3 thesis-scorer.py data/theses_counted.csv manifestations_new.json --limit 5
    python3 thesis-scorer.py data/theses_counted.csv manifestations_new.json --resume
    python3 thesis-scorer.py --from-probs data/theses_scored_probs.csv manifestations_new.json --threshold 0.7

The API key is read from settings.json (provider.Pratt.options.apiKey); PRATT_API_KEY overrides it.
"""

import argparse
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request

SYSTEMONE_URL = "https://llm.pratt.edu/v1/systemone"
MODEL = "pratt-jev"
ID_COLUMNS = ["author", "thesis_title", "paragraph"]
LEVELS = [
    "No evidence: the paragraph does not address this manifestation",
    "Weak evidence: a passing or implicit mention",
    "Moderate evidence: the paragraph clearly engages this manifestation",
    "Strong evidence: the paragraph explicitly and substantially demonstrates this manifestation",
]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

csv.field_size_limit(sys.maxsize)


def load_api_key(settings_path):
    key = os.environ.get("PRATT_API_KEY")
    if key:
        return key
    try:
        with open(settings_path) as f:
            settings = json.load(f)
        key = settings["provider"]["Pratt"]["options"]["apiKey"]
    except (OSError, json.JSONDecodeError, KeyError) as e:
        sys.exit(f"Error: could not read API key from {settings_path} ({e}). "
                 "Populate settings.json or set PRATT_API_KEY.")
    if not key or "YOUR_API_KEY" in key:
        sys.exit(f"Error: {settings_path} still contains the placeholder API key.")
    return key


def load_manifestations(path):
    """Return an ordered list of dicts: column, plo_key, plo_title, id, name, description."""
    with open(path) as f:
        data = json.load(f)
    items = []
    for plo_key, plo in data.items():
        for m in plo["manifestations"]:
            items.append({
                "column": f"{plo_key}_{m['id']}",
                "plo_key": plo_key,
                "plo_title": plo["title"],
                "plo_description": plo["description"],
                "id": m["id"],
                "name": m["manifestation"],
                "description": m["description"],
            })
    return items


def build_question(m):
    return {
        "type": "score",
        "instructions": (
            f"This paragraph comes from a graduate communications design thesis. "
            f"Program learning outcome '{m['plo_title']}': {m['plo_description']} "
            f"Manifestation '{m['name']}': {m['description']} "
            f"How much evidence of this manifestation does the paragraph show?"
        ),
        "criteria": LEVELS,
    }


def post(payload, api_key, max_retries=6):
    body = json.dumps(payload).encode("utf-8")
    delay = 2.0
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(
            SYSTEMONE_URL,
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries:
                sys.stderr.write(f"  HTTP {e.code}, retrying in {delay:.0f}s...\n")
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            raise RuntimeError(f"HTTP {e.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < max_retries:
                sys.stderr.write(f"  Network error ({e}), retrying in {delay:.0f}s...\n")
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            raise RuntimeError(f"Request failed: {e}") from None


def score_paragraph(paragraph, manifestations, api_key, batch_by_plo):
    """Return {column: {"score": float, "p": [p0, p1, p2, p3]}} for every manifestation."""
    if batch_by_plo:
        groups = {}
        for m in manifestations:
            groups.setdefault(m["plo_key"], []).append(m)
        batches = list(groups.values())
    else:
        batches = [manifestations]

    results = {}
    cost = 0.0
    for batch in batches:
        payload = {
            "model": MODEL,
            "state": paragraph,
            "questions": {m["column"]: build_question(m) for m in batch},
        }
        response = post(payload, api_key)
        answers = response.get("answers") or {}
        cost += (response.get("usage") or {}).get("cost") or 0.0
        for m in batch:
            ans = answers.get(m["column"])
            if not ans:
                raise RuntimeError(f"No answer returned for {m['column']}")
            probs = ans.get("probabilities") or {}
            p = [float(probs.get(str(i), 0.0)) for i in range(len(LEVELS))]
            results[m["column"]] = {"score": float(ans.get("score", 0.0)), "p": p}
    return results, cost


def to_count(p, threshold):
    """Highest level k such that P(level >= k) >= threshold."""
    count = 0
    for k in range(1, len(p)):
        if sum(p[k:]) >= threshold:
            count = k
    return count


def prob_columns(manifestations):
    cols = []
    for m in manifestations:
        cols.append(f"{m['column']}_score")
        cols.extend(f"{m['column']}_p{i}" for i in range(len(LEVELS)))
    return cols


def default_probs_path(output_path):
    root, ext = os.path.splitext(output_path)
    return f"{root}_probs{ext or '.csv'}"


def rebuild_from_probs(probs_path, manifestations, output_path, threshold):
    count_cols = [m["column"] for m in manifestations]
    with open(probs_path, newline="", encoding="utf-8") as fin, \
         open(output_path, "w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=ID_COLUMNS + count_cols)
        writer.writeheader()
        n = 0
        for row in reader:
            out = {c: row[c] for c in ID_COLUMNS}
            for m in manifestations:
                p = [float(row[f"{m['column']}_p{i}"]) for i in range(len(LEVELS))]
                out[m["column"]] = to_count(p, threshold)
            writer.writerow(out)
            n += 1
    print(f"Rebuilt {n} rows at threshold {threshold} -> {output_path}")


def count_existing_rows(path):
    if not os.path.exists(path):
        return 0
    with open(path, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="Score thesis paragraphs against PLO manifestations with pratt-jev.")
    parser.add_argument("input", help="Theses CSV (author, thesis_title, paragraph, ...), or the probs CSV with --from-probs")
    parser.add_argument("manifestations", help="Manifestations JSON (e.g. manifestations_new.json)")
    parser.add_argument("-o", "--output", default=os.path.join("data", "theses_scored.csv"), help="Counts CSV to write")
    parser.add_argument("--probs-output", help="Raw probabilities CSV (default: <output>_probs.csv)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Cutoff for P(level >= k) when assigning counts (default 0.5)")
    parser.add_argument("--limit", type=int, help="Only score the first N paragraphs")
    parser.add_argument("--resume", action="store_true", help="Skip rows already present in the output files")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds to pause between paragraphs (default 0.5)")
    parser.add_argument("--batch-by-plo", action="store_true", help="Send one request per PLO instead of one per paragraph")
    parser.add_argument("--settings", default=os.path.join(SCRIPT_DIR, "settings.json"), help="Path to settings.json")
    parser.add_argument("--from-probs", action="store_true", help="Rebuild counts from an existing probs CSV without calling the API")
    args = parser.parse_args()

    if not 0.0 < args.threshold <= 1.0:
        sys.exit("Error: --threshold must be in (0, 1].")

    manifestations = load_manifestations(args.manifestations)

    if args.from_probs:
        rebuild_from_probs(args.input, manifestations, args.output, args.threshold)
        return

    api_key = load_api_key(args.settings)
    probs_path = args.probs_output or default_probs_path(args.output)

    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in ID_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            sys.exit(f"Error: input CSV is missing columns: {missing}")
        rows = list(reader)
    if args.limit:
        rows = rows[:args.limit]

    count_cols = [m["column"] for m in manifestations]
    p_cols = prob_columns(manifestations)

    skip = 0
    if args.resume:
        skip = min(count_existing_rows(args.output), count_existing_rows(probs_path))
    mode = "a" if skip else "w"

    print(f"{len(rows)} paragraphs x {len(manifestations)} manifestations "
          f"({'resuming at row ' + str(skip + 1) if skip else 'starting fresh'})")

    total_cost = 0.0
    with open(args.output, mode, newline="", encoding="utf-8") as fc, \
         open(probs_path, mode, newline="", encoding="utf-8") as fp:
        count_writer = csv.DictWriter(fc, fieldnames=ID_COLUMNS + count_cols)
        prob_writer = csv.DictWriter(fp, fieldnames=ID_COLUMNS + p_cols)
        if not skip:
            count_writer.writeheader()
            prob_writer.writeheader()

        for i, row in enumerate(rows):
            if i < skip:
                continue
            paragraph = (row.get("paragraph") or "").strip()
            ids = {c: row.get(c, "") for c in ID_COLUMNS}
            label = f"[{i + 1}/{len(rows)}] {ids['author']}"

            if paragraph:
                try:
                    results, cost = score_paragraph(paragraph, manifestations, api_key, args.batch_by_plo)
                except RuntimeError as e:
                    sys.exit(f"{label}: failed - {e}\nRe-run with --resume to continue from this row.")
                total_cost += cost
            else:
                results = {m["column"]: {"score": 0.0, "p": [1.0, 0.0, 0.0, 0.0]} for m in manifestations}

            count_row = dict(ids)
            prob_row = dict(ids)
            for m in manifestations:
                r = results[m["column"]]
                count_row[m["column"]] = to_count(r["p"], args.threshold)
                prob_row[f"{m['column']}_score"] = round(r["score"], 4)
                for lvl, pv in enumerate(r["p"]):
                    prob_row[f"{m['column']}_p{lvl}"] = round(pv, 4)
            count_writer.writerow(count_row)
            prob_writer.writerow(prob_row)
            fc.flush()
            fp.flush()

            hits = sum(1 for c in count_cols if count_row[c] > 0)
            print(f"{label}: {hits} manifestations with evidence")
            if args.delay:
                time.sleep(args.delay)

    print(f"\nDone. Counts -> {args.output}\nProbabilities -> {probs_path}\nAPI cost this run: ${total_cost:.6f}")


if __name__ == "__main__":
    main()
