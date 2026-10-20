#!/usr/bin/env python3
"""
Replace student names in data/*.csv with stable 6-letter hashed IDs.

Each author + thesis title pair maps to an ID derived from an FNV-1a hash of their
name initials and thesis-title initials (the same algorithm index.html used to run in
the browser, so existing "Student QMRTXA" labels are preserved).

Outputs:
    student_hash.json   private, git-ignored map of full name -> {hash, thesis_title}
    data/*.csv          the author column is rewritten in place with the hashed ID

Re-running is safe: existing IDs in student_hash.json are reused, and author values
that are already hashes are left alone. Run this on any new CSV (e.g. from
parse_theses.py) before committing it.

Usage:
    python3 hash_students.py                 # hash every CSV in data/
    python3 hash_students.py data/new.csv    # hash specific files
    python3 hash_students.py --check         # fail if any name appears in a git-tracked file
"""

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HASH_PATH = os.path.join(SCRIPT_DIR, "student_hash.json")
MASK = 0xFFFFFFFF

csv.field_size_limit(sys.maxsize)


def initials_of(text):
    return "".join(w[0].upper() for w in re.split(r"[^A-Za-z0-9]+", text or "") if w)


def hash_to_letters(s, length=6):
    """FNV-1a 32-bit, then base-26 letters (port of the JS version, incl. Math.imul semantics)."""
    h = 0x811C9DC5
    units = s.encode("utf-16-le")
    for i in range(0, len(units), 2):
        h ^= units[i] | (units[i + 1] << 8)
        h = (h * 0x01000193) & MASK
    out = ""
    for _ in range(length):
        out += chr(65 + h % 26)
        h = ((h // 26) ^ ((h * 0x9E3779B1) & MASK)) & MASK
    return out


def load_mapping(path):
    """Return {(author, title): hash} from student_hash.json, if it exists."""
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    mapping = {}
    for name, entry in data.items():
        for e in entry if isinstance(entry, list) else [entry]:
            mapping[(name, e["thesis_title"])] = e["hash"]
    return mapping


def save_mapping(path, mapping):
    by_name = {}
    for (name, title), h in sorted(mapping.items()):
        by_name.setdefault(name, []).append({"hash": h, "thesis_title": title})
    out = {}
    for name, entries in by_name.items():
        if len(entries) > 1:
            sys.stderr.write(f"Warning: one student has {len(entries)} thesis titles "
                             f"({', '.join(e['hash'] for e in entries)})\n")
        out[name] = entries if len(entries) > 1 else entries[0]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")


def assign_ids(keys, mapping):
    """Give every new (author, title) key an unused ID, in sorted order like the JS version."""
    used = set(mapping.values())
    for author, title in sorted(k for k in keys if k not in mapping):
        seed = initials_of(author) + initials_of(title)
        counter = 0
        while True:
            h = hash_to_letters(f"{seed}#{counter}" if counter else seed)
            counter += 1
            if h not in used:
                break
        used.add(h)
        mapping[(author, title)] = h


def name_patterns(mapping):
    """Case-insensitive patterns for each student's full name (and first + last, if longer)."""
    patterns = []
    for name, h in {name: h for (name, _), h in mapping.items()}.items():
        tokens = name.split()
        variants = {name}
        if len(tokens) > 2:
            variants.add(f"{tokens[0]} {tokens[-1]}")
        for v in variants:
            body = r"\s+".join(re.escape(t) for t in v.split())
            patterns.append((re.compile(r"(?<!\w)" + body + r"(?!\w)", re.IGNORECASE), h))
    # Longest names first so a full name wins over its first + last variant
    return sorted(patterns, key=lambda p: -len(p[0].pattern))


def scrub_text(text, patterns):
    for pattern, h in patterns:
        text = pattern.sub(h, text)
    return text


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        first = f.readline()
        f.seek(0)
        reader = csv.DictReader(f)
        rows = list(reader)
        return reader.fieldnames or [], rows, "\r\n" if first.endswith("\r\n") else "\n"


def default_csvs():
    return sorted(glob.glob(os.path.join(SCRIPT_DIR, "data", "*.csv")))


def hash_files(paths):
    mapping = load_mapping(HASH_PATH)
    known_hashes = set(mapping.values())

    tables = []
    keys = set()
    for path in paths:
        fields, rows, eol = read_csv(path)
        if "author" not in fields or "thesis_title" not in fields:
            print(f"skip {os.path.relpath(path, SCRIPT_DIR)} (no author/thesis_title columns)")
            continue
        tables.append((path, fields, rows, eol))
        for r in rows:
            author = (r["author"] or "").strip()
            if author and author not in known_hashes:
                keys.add((author, (r["thesis_title"] or "").strip()))

    assign_ids(keys, mapping)
    patterns = name_patterns(mapping)

    for path, fields, rows, eol in tables:
        changed = scrubbed = 0
        for r in rows:
            author = (r["author"] or "").strip()
            if author and author not in known_hashes:
                r["author"] = mapping[(author, (r["thesis_title"] or "").strip())]
                changed += 1
            # Names also turn up inside text, e.g. title-page bylines in the paragraph column
            for c in fields:
                if c != "author" and r.get(c):
                    clean = scrub_text(r[c], patterns)
                    if clean != r[c]:
                        r[c] = clean
                        scrubbed += 1
        if changed or scrubbed:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields, lineterminator=eol)
                writer.writeheader()
                writer.writerows(rows)
        print(f"{os.path.relpath(path, SCRIPT_DIR)}: {changed} authors hashed, {scrubbed} names scrubbed from text")

    save_mapping(HASH_PATH, mapping)
    print(f"{len(mapping)} students -> {os.path.relpath(HASH_PATH, SCRIPT_DIR)}")


def check_tracked():
    mapping = load_mapping(HASH_PATH)
    if not mapping:
        sys.exit(f"Error: {HASH_PATH} is missing or empty.")
    names = {name for name, _ in mapping}
    patterns = name_patterns(mapping)
    files = subprocess.run(["git", "ls-files", "-z"], cwd=SCRIPT_DIR, capture_output=True,
                           check=True, text=True).stdout.split("\0")
    found = 0
    for rel in filter(None, files):
        path = os.path.join(SCRIPT_DIR, rel)
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as f:
            text = f.read().decode("utf-8", errors="ignore")
        hits = sorted({h for p, h in patterns if p.search(text)})
        if hits:
            found += len(hits)
            print(f"{rel}: names of {len(hits)} students (e.g. {', '.join(hits[:5])})")
    if found:
        sys.exit(f"Found student names in tracked files ({found} matches).")
    print(f"OK: none of {len(names)} student names appear in git-tracked files.")


def main():
    parser = argparse.ArgumentParser(description="Hash student names in data CSVs.")
    parser.add_argument("csvs", nargs="*", help="CSV files to hash (default: data/*.csv)")
    parser.add_argument("--check", action="store_true", help="Verify no student name appears in a git-tracked file")
    args = parser.parse_args()
    if args.check:
        check_tracked()
    else:
        hash_files([os.path.abspath(p) for p in args.csvs] or default_csvs())


if __name__ == "__main__":
    main()
