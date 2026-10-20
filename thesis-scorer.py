#!/usr/bin/env python3
"""
Score thesis paragraphs against PLO manifestations with several Pratt models in parallel.

For each paragraph in a theses CSV (author, thesis_title, paragraph, ...), every model rates
every manifestation in a manifestations JSON on a 0-5 scale that weighs the paragraph's
intent, not just shared vocabulary:

    0 = no evidence
    1 = low evidence, or only incidental to the paragraph's intent
    2 = some evidence, and some ancillary intent of the paragraph
    3 = some evidence, and clearly related to the paragraph's intent
    4 = multiple notes of evidence, and the core intent of the paragraph
    5 = very strong demonstration, and the core argument of the paragraph

Models (see MODELS): pratt-jev is a decision model called on /v1/systemone and returns level
probabilities directly; the others are chat models on /v1/chat/completions that are asked
for a JSON probability distribution over the six levels.

Outputs, per model slug:
    <prefix>_<slug>.csv        author, thesis_title, paragraph, PLO1_P1M01 ... (counts 0-5)
    <prefix>_<slug>_probs.csv  the same rows with paragraph_intent, expected scores and level
                               probabilities, so counts can be recomputed at any threshold

A count is the highest level k where P(level >= k) >= threshold.

Count method (--method count): every paragraph is split into sentences, and each sentence is
checked against every manifestation. A sentence whose confidence is at least --threshold
(default 0.8) adds 1 to that manifestation's count. Jev answers one yes/no question per
manifestation for each sentence; chat models rate numbered sentences in chunks of 40.

    <prefix>_<slug>.csv            author, thesis_title, paragraph, PLO1_P1M01 ... (sentence counts)
    <prefix>_<slug>_sentences.csv  one row per sentence with a confidence per manifestation

Usage:
    python3 thesis-scorer.py data/theses_counted.csv manifestations_new.json
    python3 thesis-scorer.py data/theses_counted.csv manifestations_new.json --models jev,gemini --limit 5
    python3 thesis-scorer.py data/theses_counted.csv manifestations_new.json --resume
    python3 thesis-scorer.py --from-probs data/theses_scored_jev_probs.csv manifestations_new.json --threshold 0.7
    python3 thesis-scorer.py data/theses_counted.csv manifestations_new.json --method count --models jev,gemini
    python3 thesis-scorer.py --from-sentences data/theses_count_jev_sentences.csv manifestations_new.json --threshold 0.6

The input CSV's author column should already be hashed (see hash_students.py).
The API key is read from settings.json (provider.Pratt.options.apiKey); PRATT_API_KEY overrides it.
"""

import argparse
import concurrent.futures
import csv
import http.client
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request

API_BASE = "https://llm.pratt.edu/v1"
SYSTEMONE_URL = f"{API_BASE}/systemone"
CHAT_URL = f"{API_BASE}/chat/completions"
MODELS = {
    "jev":          {"id": "pratt-jev",          "kind": "decision"},
    "medium":       {"id": "pratt-medium",       "kind": "chat"},
    "gemini":       {"id": "pratt-gemini",       "kind": "chat"},
    "deepseek-pro": {"id": "pratt-deepseek-pro", "kind": "chat"},
    "qwen":         {"id": "pratt-qwen",         "kind": "chat"},
}
ID_COLUMNS = ["author", "thesis_title", "paragraph"]
SENTENCE_ID_COLUMNS = ["author", "thesis_title", "paragraph_index", "sentence_index", "sentence"]
LEVELS = [
    "No evidence of this manifestation",
    "Low evidence, or only incidental to the paragraph's intent",
    "Some evidence, and part of the paragraph's ancillary intent",
    "Some evidence, and clearly related to the paragraph's intent",
    "Multiple notes of evidence, and the core intent of the paragraph",
    "Very strong demonstration, and the core argument of the paragraph",
]
READING_GUIDANCE = (
    "First work out what the paragraph is arguing or doing: its core argument and any secondary "
    "purposes. Then judge whether the meaning of the manifestation, as its description defines it, "
    "is present in that intent. Do not reward shared vocabulary: a paragraph can strongly "
    "demonstrate a manifestation without using any words from its description, and can use those "
    "words while demonstrating nothing."
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_THRESHOLD = {"score": 0.5, "count": 0.8}
CHAT_CHUNK_SENTENCES = 40
ABBREVIATIONS = {
    "e.g", "i.e", "al", "etc", "vs", "cf", "dr", "mr", "mrs", "ms", "st", "prof", "jr", "sr",
    "no", "vol", "pp", "p", "fig", "ed", "eds", "ch", "inc", "co", "ltd", "u.s", "u.k", "ca", "c",
}
# End punctuation, optional closing quotes/brackets, optional footnote number, then a sentence start
SENTENCE_BREAK = re.compile(r"([.!?])([”\"’')\]]*)(\d{0,3})\s+(?=[A-Z“\"‘(\[])")

csv.field_size_limit(sys.maxsize)
print_lock = threading.Lock()


def log(msg, err=False):
    with print_lock:
        (sys.stderr if err else sys.stdout).write(msg + "\n")
        (sys.stderr if err else sys.stdout).flush()


class FatalAPIError(RuntimeError):
    """A non-retryable HTTP error (bad key, model not in the key's access group, bad request)."""

    def __init__(self, code, detail):
        super().__init__(f"HTTP {code}: {detail}")
        self.code = code


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
    if not key or "YOUR_API_KEY" in key or "{{" in key:
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


def post(url, payload, api_key, max_retries=10):
    body = json.dumps(payload).encode("utf-8")
    delay = 2.0
    for attempt in range(max_retries + 1):
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries:
                log(f"  HTTP {e.code}, retrying in {delay:.0f}s...", err=True)
            else:
                raise FatalAPIError(e.code, detail[:500]) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError,
                http.client.HTTPException, json.JSONDecodeError) as e:
            if attempt >= max_retries:
                raise RuntimeError(f"Request failed: {e}") from None
            log(f"  Network error ({e}), retrying in {delay:.0f}s...", err=True)
        time.sleep(delay)
        delay = min(delay * 2, 60)


def usage_cost(response):
    return (response.get("usage") or {}).get("cost") or 0.0


def split_sentences(paragraph):
    """Split a paragraph into sentences, guarding against abbreviations, initials and decimals."""
    text = re.sub(r"\s+", " ", paragraph or "").strip()
    if not text:
        return []
    sentences, start = [], 0
    for m in SENTENCE_BREAK.finditer(text):
        if m.group(1) == ".":
            word = re.search(r"([\w.]+)$", text[start:m.start()])
            word = word.group(1).lower() if word else ""
            # Abbreviations ("et al.", "e.g.") and single-letter initials ("J. Smith") don't end sentences
            if word.rstrip(".") in ABBREVIATIONS or re.fullmatch(r"[a-z]", word):
                continue
        sentences.append(text[start:m.end(3)].strip())
        start = m.end()
    sentences.append(text[start:].strip())
    # Merge fragments under 3 words into the following sentence (or the previous one at the end)
    merged = []
    carry = ""
    for s in sentences:
        s = f"{carry} {s}".strip() if carry else s
        carry = ""
        if len(s.split()) < 3:
            carry = s
        else:
            merged.append(s)
    if carry:
        if merged:
            merged[-1] = f"{merged[-1]} {carry}"
        else:
            merged.append(carry)
    return merged


def normalize(p, fallback_score=0):
    """Clamp to six non-negative values summing to 1; one-hot on fallback_score if empty."""
    p = [max(0.0, float(x or 0.0)) for x in (p or [])][:len(LEVELS)]
    p += [0.0] * (len(LEVELS) - len(p))
    total = sum(p)
    if total <= 0:
        k = min(len(LEVELS) - 1, max(0, int(round(float(fallback_score or 0)))))
        return [1.0 if i == k else 0.0 for i in range(len(LEVELS))]
    return [x / total for x in p]


# ── Decision model (pratt-jev) ────────────────────────────────────────────────

def build_question(m):
    return {
        "type": "score",
        "instructions": (
            f"The state is a paragraph from a graduate communications design thesis. {READING_GUIDANCE} "
            f"Program learning outcome '{m['plo_title']}': {m['plo_description']} "
            f"Manifestation '{m['name']}': {m['description']} "
            f"How much evidence of this manifestation does the paragraph show, given its intent?"
        ),
        "criteria": LEVELS,
    }


def score_decision(model_id, title, paragraph, manifestations, api_key):
    payload = {
        "model": model_id,
        "state": {"thesis_title": title, "paragraph": paragraph},
        "questions": {m["column"]: build_question(m) for m in manifestations},
    }
    response = post(SYSTEMONE_URL, payload, api_key)
    answers = response.get("answers") or {}
    results = {}
    for m in manifestations:
        ans = answers.get(m["column"])
        if not ans:
            raise RuntimeError(f"No answer returned for {m['column']}")
        probs = ans.get("probabilities") or {}
        results[m["column"]] = normalize([probs.get(str(i), 0.0) for i in range(len(LEVELS))],
                                         ans.get("score", 0))
    return results, "", usage_cost(response)


# ── Chat models ───────────────────────────────────────────────────────────────

def build_system_prompt(manifestations):
    levels = "\n".join(f"{i} = {text}" for i, text in enumerate(LEVELS))
    plos = {}
    for m in manifestations:
        plos.setdefault(m["plo_key"], {"title": m["plo_title"], "description": m["plo_description"], "items": []})
        plos[m["plo_key"]]["items"].append(m)
    blocks = []
    for key, plo in plos.items():
        lines = [f"{key}: {plo['title']} - {plo['description']}"]
        lines += [f"  {m['column']}: {m['name']} - {m['description']}" for m in plo["items"]]
        blocks.append("\n".join(lines))
    example = ", ".join(f'"{m["column"]}": {{"probabilities": [p0, p1, p2, p3, p4, p5], "score": n}}'
                        for m in manifestations[:2])
    return (
        "You assess paragraphs from graduate communications design theses for evidence of a "
        "program's learning outcomes.\n\n"
        f"How to read: {READING_GUIDANCE}\n\n"
        f"Rate every manifestation below on this scale:\n{levels}\n\n"
        f"Manifestations, grouped by program learning outcome:\n\n" + "\n\n".join(blocks) + "\n\n"
        "Reply with JSON only, no prose or code fences, in this shape:\n"
        f'{{"intent": "<one or two sentences on the paragraph\'s core argument>", '
        f'"scores": {{{example}, ...}}}}\n'
        f"Include all {len(manifestations)} manifestation IDs. For each, \"probabilities\" is your "
        "probability for each level 0-5 (summing to 1) and \"score\" is the single most likely level."
    )


def parse_chat_json(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in reply")
    text = text[start:end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Repair common slips: missing commas between objects/values, trailing commas
        repaired = re.sub(r"([}\]\d\"])\s*\n?\s*([{\"])", r"\1,\2", text)
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        return json.loads(repaired)


def chat_json(model_id, messages, api_key, state, parse):
    """POST a chat request and return (parse(reply_json), cost), retrying once on an unparseable reply."""
    cost = 0.0
    last_error = None
    attempts = 0
    while attempts < 2:  # one retry if the reply can't be parsed
        # The retry runs slightly warmer so a malformed reply isn't simply repeated
        payload = {"model": model_id, "messages": messages, "temperature": 0.3 if attempts else 0,
                   "max_tokens": 8000}
        if state.get("json_mode", True):
            payload["response_format"] = {"type": "json_object"}
        try:
            response = post(CHAT_URL, payload, api_key)
        except FatalAPIError as e:
            if e.code == 400 and state.get("json_mode", True):
                state["json_mode"] = False  # model rejects response_format; ask in the prompt only
                continue
            raise
        attempts += 1
        cost += usage_cost(response)
        try:
            content = response["choices"][0]["message"]["content"]
            return parse(parse_chat_json(content)), cost
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as e:
            last_error = e
    raise RuntimeError(f"Could not parse reply from {model_id}: {last_error}")


def score_chat(model_id, title, paragraph, manifestations, api_key, system_prompt, state):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Thesis title: {title}\n\nParagraph:\n{paragraph}"},
    ]

    def parse(data):
        scores = data.get("scores") or {}
        missing = [m["column"] for m in manifestations if m["column"] not in scores]
        if missing:
            raise ValueError(f"missing {len(missing)} manifestations, e.g. {missing[0]}")
        results = {}
        for m in manifestations:
            s = scores[m["column"]] or {}
            p = s.get("probabilities")
            if isinstance(p, dict):
                p = [p.get(str(i), 0.0) for i in range(len(LEVELS))]
            results[m["column"]] = normalize(p, s.get("score", 0))
        return results, str(data.get("intent") or "").strip()

    (results, intent), cost = chat_json(model_id, messages, api_key, state, parse)
    return results, intent, cost


# ── Count method: sentence-level evidence ─────────────────────────────────────

COUNT_GUIDANCE = (
    "Judge the sentence by what it means within its paragraph, not by shared vocabulary: it is "
    "evidence only if it actually demonstrates the manifestation's meaning as its description "
    "defines it. A sentence can be evidence without using any words from the description, and "
    "can use those words without being evidence."
)


def clamp01(x):
    try:
        return min(1.0, max(0.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


def build_noul_question(m):
    return {
        "type": "noul",
        "instructions": (
            "The state is one sentence from a paragraph of a graduate communications design thesis, "
            f"with its neighbouring sentences for context. {COUNT_GUIDANCE} "
            f"Program learning outcome '{m['plo_title']}': {m['plo_description']} "
            f"Manifestation '{m['name']}': {m['description']} "
            "Is this sentence evidence of this manifestation?"
        ),
    }


def count_decision(model_id, title, sentences, manifestations, api_key, workers):
    """One /v1/systemone request per sentence; returns ([{column: confidence}] per sentence, cost)."""
    questions = {m["column"]: build_noul_question(m) for m in manifestations}

    def one(i):
        state = {
            "thesis_title": title,
            "previous_sentence": sentences[i - 1] if i else "",
            "sentence": sentences[i],
            "next_sentence": sentences[i + 1] if i + 1 < len(sentences) else "",
        }
        response = post(SYSTEMONE_URL, {"model": model_id, "state": state, "questions": questions}, api_key)
        answers = response.get("answers") or {}
        conf = {}
        for m in manifestations:
            ans = answers.get(m["column"])
            if not ans:
                raise RuntimeError(f"No answer returned for {m['column']}")
            conf[m["column"]] = clamp01(ans.get("noul"))
        return conf, usage_cost(response)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(workers, len(sentences)))) as pool:
        out = list(pool.map(one, range(len(sentences))))
    return [c for c, _ in out], sum(x for _, x in out)


def build_count_system_prompt(manifestations):
    plos = {}
    for m in manifestations:
        plos.setdefault(m["plo_key"], {"title": m["plo_title"], "description": m["plo_description"], "items": []})
        plos[m["plo_key"]]["items"].append(m)
    blocks = []
    for key, plo in plos.items():
        lines = [f"{key}: {plo['title']} - {plo['description']}"]
        lines += [f"  {m['column']}: {m['name']} - {m['description']}" for m in plo["items"]]
        blocks.append("\n".join(lines))
    return (
        "You find sentence-level evidence of a design program's learning outcomes in graduate "
        "communications design theses.\n\n"
        f"How to read: first work out what the paragraph is arguing. {COUNT_GUIDANCE}\n\n"
        "Manifestations, grouped by program learning outcome:\n\n" + "\n\n".join(blocks) + "\n\n"
        "For each numbered sentence, give your confidence (0 to 1) that it is evidence of each "
        "manifestation. List only manifestations with confidence of at least 0.3; leave out the rest.\n"
        "Reply with JSON only, no prose or code fences, in this shape:\n"
        '{"sentences": [{"n": 1, "evidence": {"PLO1_P1M01": 0.9, "PLO2_P2M03": 0.55}}, '
        '{"n": 2, "evidence": {}}, ...]}\n'
        "Include every sentence number, with an empty evidence object when there is none."
    )


def count_chat(model_id, title, sentences, manifestations, api_key, system_prompt, state):
    """One chat request per chunk of up to CHAT_CHUNK_SENTENCES sentences."""
    columns = [m["column"] for m in manifestations]
    n_chunks = (len(sentences) + CHAT_CHUNK_SENTENCES - 1) // CHAT_CHUNK_SENTENCES

    def rate(start, size, part_label):
        chunk = sentences[start:start + size]
        user = f"Thesis title: {title}\n\n"
        if part_label:
            user += f"{part_label} of a long paragraph ({len(sentences)} sentences).\n"
        context = sentences[max(0, start - 2):start]
        if context:
            user += "Preceding sentences, for context only (do not rate them):\n" + "\n".join(context) + "\n\n"
        user += "Sentences to rate:\n" + "\n".join(f"{i + 1}. {s}" for i, s in enumerate(chunk))
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]

        def parse(data):
            rows = data["sentences"]
            if not isinstance(rows, list):
                raise ValueError("'sentences' is not a list")
            by_n = {}
            for r in rows:
                by_n[int(r.get("n"))] = r.get("evidence") or {}
            # Sentences the model left out have no evidence
            return [{c: clamp01(by_n.get(i + 1, {}).get(c, 0.0)) for c in columns} for i in range(len(chunk))]

        try:
            return chat_json(model_id, messages, api_key, state, parse)
        except FatalAPIError:
            raise
        except RuntimeError as e:
            if size <= 5:
                raise
            # Malformed replies are sporadic; smaller chunks give shorter replies and a fresh prompt
            log(f"  {model_id}: {e}; splitting {size} sentences in half", err=True)
            half = size // 2
            first, c1 = rate(start, half, part_label)
            second, c2 = rate(start + half, size - half, part_label)
            return first + second, c1 + c2

    confs, cost = [], 0.0
    for k, start in enumerate(range(0, len(sentences), CHAT_CHUNK_SENTENCES)):
        size = min(CHAT_CHUNK_SENTENCES, len(sentences) - start)
        label = f"This is part {k + 1} of {n_chunks}" if n_chunks > 1 else ""
        chunk_confs, chunk_cost = rate(start, size, label)
        confs.extend(chunk_confs)
        cost += chunk_cost
    return confs, cost


# ── Output ────────────────────────────────────────────────────────────────────

def to_count(p, threshold):
    """Highest level k such that P(level >= k) >= threshold."""
    count = 0
    for k in range(1, len(p)):
        if sum(p[k:]) >= threshold:
            count = k
    return count


def expected_score(p):
    return sum(i * pv for i, pv in enumerate(p))


def prob_columns(manifestations):
    cols = ["paragraph_intent"]
    for m in manifestations:
        cols.append(f"{m['column']}_score")
        cols.extend(f"{m['column']}_p{i}" for i in range(len(LEVELS)))
    return cols


def output_paths(prefix, slug, method="score"):
    detail = "sentences" if method == "count" else "probs"
    return f"{prefix}_{slug}.csv", f"{prefix}_{slug}_{detail}.csv"


def sentence_columns(manifestations):
    return SENTENCE_ID_COLUMNS + [m["column"] for m in manifestations]


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


def rebuild_from_sentences(sentences_path, manifestations, output_path, threshold, ids_path):
    """Recount sentences at a new confidence threshold.

    ID columns come from the existing counts CSV, or from ids_path (the theses input) if it's missing.
    """
    count_cols = [m["column"] for m in manifestations]
    with open(sentences_path, newline="", encoding="utf-8") as f:
        sentence_rows = list(csv.DictReader(f))
    source = output_path if os.path.exists(output_path) else ids_path
    with open(source, newline="", encoding="utf-8") as f:
        id_rows = [{c: r[c] for c in ID_COLUMNS} for r in csv.DictReader(f)]
    if source == ids_path and sentence_rows:
        # Only the paragraphs the run reached
        id_rows = id_rows[:max(int(r["paragraph_index"]) for r in sentence_rows) + 1]
    counts = [{c: 0 for c in count_cols} for _ in id_rows]
    for r in sentence_rows:
        i = int(r["paragraph_index"])
        for c in count_cols:
            if float(r[c]) >= threshold:
                counts[i][c] += 1
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ID_COLUMNS + count_cols)
        writer.writeheader()
        for ids, c in zip(id_rows, counts):
            writer.writerow({**ids, **c})
    print(f"Rebuilt {len(id_rows)} rows at threshold {threshold} -> {output_path}")


def truncate_sentences(path, keep_paragraphs):
    """Drop sentence rows for paragraphs at or after keep_paragraphs, so resume appends cleanly."""
    if not os.path.exists(path):
        return
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        kept = [r for r in reader if int(r["paragraph_index"]) < keep_paragraphs]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)


def count_existing_rows(path):
    if not os.path.exists(path):
        return 0
    with open(path, newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.DictReader(f))


# ── Per-model run ─────────────────────────────────────────────────────────────

def run_model(slug, rows, manifestations, args, api_key):
    spec = MODELS[slug]
    counting = args.method == "count"
    counts_path, detail_path = output_paths(args.prefix, slug, args.method)
    count_cols = [m["column"] for m in manifestations]
    if spec["kind"] == "chat":
        system_prompt = (build_count_system_prompt if counting else build_system_prompt)(manifestations)
    chat_state = {}

    skip = 0
    if args.resume:
        if counting:
            skip = count_existing_rows(counts_path)
            truncate_sentences(detail_path, skip)
        else:
            skip = min(count_existing_rows(counts_path), count_existing_rows(detail_path))
    if skip >= len(rows):
        log(f"[{slug}] already complete ({skip} rows)")
        return {"slug": slug, "rows": 0, "cost": 0.0, "status": "complete"}
    log(f"[{slug}] {spec['id']} ({args.method}): {len(rows) - skip} paragraphs "
        f"({'resuming at row ' + str(skip + 1) if skip else 'starting fresh'})")

    def score_row(row):
        paragraph = (row.get("paragraph") or "").strip()
        title = (row.get("thesis_title") or "").strip()
        if counting:
            sentences = split_sentences(paragraph)
            if not sentences:
                return [], [], 0.0
            if spec["kind"] == "decision":
                confs, cost = count_decision(spec["id"], title, sentences, manifestations, api_key, args.sentence_workers)
            else:
                confs, cost = count_chat(spec["id"], title, sentences, manifestations, api_key, system_prompt, chat_state)
            return sentences, confs, cost
        if not paragraph:
            return {c: normalize([], 0) for c in count_cols}, "", 0.0
        if spec["kind"] == "decision":
            return score_decision(spec["id"], title, paragraph, manifestations, api_key)
        return score_chat(spec["id"], title, paragraph, manifestations, api_key, system_prompt, chat_state)

    def output_rows(i, ids, result):
        """Return (counts row, detail rows) for one paragraph."""
        count_row = dict(ids)
        if counting:
            sentences, confs, _ = result
            for c in count_cols:
                count_row[c] = sum(1 for conf in confs if conf[c] >= args.threshold)
            details = []
            for j, (sentence, conf) in enumerate(zip(sentences, confs)):
                details.append({"author": ids["author"], "thesis_title": ids["thesis_title"],
                                "paragraph_index": i, "sentence_index": j, "sentence": sentence,
                                **{c: round(conf[c], 4) for c in count_cols}})
            return count_row, details
        results, intent, _ = result
        prob_row = dict(ids, paragraph_intent=intent)
        for m in manifestations:
            p = results[m["column"]]
            count_row[m["column"]] = to_count(p, args.threshold)
            prob_row[f"{m['column']}_score"] = round(expected_score(p), 4)
            for lvl, pv in enumerate(p):
                prob_row[f"{m['column']}_p{lvl}"] = round(pv, 4)
        return count_row, [prob_row]

    detail_fields = sentence_columns(manifestations) if counting else ID_COLUMNS + prob_columns(manifestations)
    mode = "a" if skip else "w"
    done, cost, status = 0, 0.0, "complete"
    with open(counts_path, mode, newline="", encoding="utf-8") as fc, \
         open(detail_path, mode, newline="", encoding="utf-8") as fd, \
         concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        count_writer = csv.DictWriter(fc, fieldnames=ID_COLUMNS + count_cols)
        detail_writer = csv.DictWriter(fd, fieldnames=detail_fields)
        if not skip:
            count_writer.writeheader()
            detail_writer.writeheader()

        # map() yields in input order, so rows are written in order and --resume stays valid
        todo = rows[skip:]
        try:
            for offset, (row, result) in enumerate(zip(todo, pool.map(score_row, todo))):
                i = skip + offset
                ids = {c: row.get(c, "") for c in ID_COLUMNS}
                count_row, details = output_rows(i, ids, result)
                # Detail rows first: a paragraph is only "done" for --resume once its counts row exists
                detail_writer.writerows(details)
                fd.flush()
                count_writer.writerow(count_row)
                fc.flush()
                done += 1
                cost += result[-1]
                hits = sum(1 for c in count_cols if count_row[c] > 0)
                log(f"[{slug}] [{i + 1}/{len(rows)}] {ids['author']}: {hits} manifestations with evidence")
        except FatalAPIError as e:
            status = f"stopped: {e}"
            log(f"[{slug}] stopped - {e}", err=True)
            pool.shutdown(wait=False, cancel_futures=True)
        except RuntimeError as e:
            status = f"stopped at row {skip + done + 1}: {e}"
            log(f"[{slug}] {status}\n[{slug}] Re-run with --resume to continue.", err=True)
            pool.shutdown(wait=False, cancel_futures=True)

    if not skip and not done:
        # Don't leave header-only files behind; index.html treats a present CSV as available
        for path in (counts_path, detail_path):
            os.remove(path)
    return {"slug": slug, "rows": done, "cost": cost, "status": status,
            "counts": counts_path, "detail": detail_path}


def main():
    parser = argparse.ArgumentParser(description="Score thesis paragraphs against PLO manifestations with Pratt models.")
    parser.add_argument("input", help="Theses CSV (author, thesis_title, paragraph, ...), or the probs CSV with --from-probs")
    parser.add_argument("manifestations", help="Manifestations JSON (e.g. manifestations_new.json)")
    parser.add_argument("--models", default=",".join(MODELS),
                        help=f"Comma-separated model slugs to run in parallel (default: all of {', '.join(MODELS)})")
    parser.add_argument("--method", choices=["score", "count"], default="score",
                        help="score: one 0-5 rating per paragraph; count: sentences with evidence per paragraph")
    parser.add_argument("--prefix", help="Output path prefix (default data/theses_scored for score, data/theses_count "
                                         "for count); files are <prefix>_<model>.csv plus _probs or _sentences")
    parser.add_argument("--workers", type=int, default=4, help="Paragraphs in flight per model (default 4)")
    parser.add_argument("--sentence-workers", type=int, default=4,
                        help="Count method with Jev: sentence requests in flight per paragraph (default 4)")
    parser.add_argument("--threshold", type=float,
                        help="score: cutoff for P(level >= k) (default 0.5); count: sentence confidence cutoff (default 0.8)")
    parser.add_argument("--limit", type=int, help="Only score the first N paragraphs")
    parser.add_argument("--resume", action="store_true", help="Skip rows already present in each model's output files")
    parser.add_argument("--settings", default=os.path.join(SCRIPT_DIR, "settings.json"), help="Path to settings.json")
    parser.add_argument("--from-probs", action="store_true", help="Rebuild counts from an existing probs CSV without calling the API")
    parser.add_argument("--from-sentences", action="store_true",
                        help="Recount an existing sentences CSV at --threshold without calling the API")
    parser.add_argument("--ids", default=os.path.join("data", "theses_counted.csv"),
                        help="With --from-sentences: theses CSV for author/title/paragraph when the counts CSV is missing")
    parser.add_argument("-o", "--output", help="With --from-probs/--from-sentences: counts CSV to write "
                                               "(default: the input path without _probs/_sentences)")
    args = parser.parse_args()
    if args.from_sentences:
        args.method = "count"
    if args.threshold is None:
        args.threshold = DEFAULT_THRESHOLD[args.method]
    if args.prefix is None:
        args.prefix = os.path.join("data", "theses_count" if args.method == "count" else "theses_scored")

    if not 0.0 < args.threshold <= 1.0:
        sys.exit("Error: --threshold must be in (0, 1].")

    manifestations = load_manifestations(args.manifestations)

    if args.from_probs:
        output = args.output or re.sub(r"_probs(\.csv)?$", r"\1", args.input)
        if output == args.input:
            sys.exit("Error: pass -o; could not derive a counts path from the probs path.")
        rebuild_from_probs(args.input, manifestations, output, args.threshold)
        return
    if args.from_sentences:
        output = args.output or re.sub(r"_sentences(\.csv)?$", r"\1", args.input)
        if output == args.input:
            sys.exit("Error: pass -o; could not derive a counts path from the sentences path.")
        rebuild_from_sentences(args.input, manifestations, output, args.threshold, args.ids)
        return

    slugs = [s.strip() for s in args.models.split(",") if s.strip()]
    unknown = [s for s in slugs if s not in MODELS]
    if unknown:
        sys.exit(f"Error: unknown model(s) {unknown}; choose from {', '.join(MODELS)}")
    api_key = load_api_key(args.settings)

    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [c for c in ID_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            sys.exit(f"Error: input CSV is missing columns: {missing}")
        rows = list(reader)
    if args.limit:
        rows = rows[:args.limit]

    print(f"{len(rows)} paragraphs x {len(manifestations)} manifestations x {len(slugs)} models ({args.method})")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(slugs)) as pool:
        summaries = list(pool.map(lambda s: run_model(s, rows, manifestations, args, api_key), slugs))

    print("\nSummary:")
    for s in summaries:
        print(f"  {s['slug']:<13} {s['rows']:>4} rows  ${s['cost']:.4f}  {s['status']}")
        if s.get("counts") and s["rows"]:
            print(f"  {'':<13} -> {s['counts']}, {s['detail']}")


if __name__ == "__main__":
    main()
