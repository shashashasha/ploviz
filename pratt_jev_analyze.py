#!/usr/bin/env python3
"""
Analyze text with Pratt's pratt-jev decision model.

pratt-jev (Typesafe Jev 1.13) is a structured decision model, not a chat model.
It is invoked via POST https://llm.pratt.edu/v1/systemone with a { state, questions }
body. It returns typed answers with probabilities rather than prose.

Usage:
    export PRATT_API_KEY=sk-pratt-...
    python3 pratt_jev_analyze.py "The checkout page goes blank after clicking Pay."

    # Or pipe text in:
    cat ticket.txt | python3 pratt_jev_analyze.py

Question types supported:
    noul   - yes/no probability (probability the answer is "yes")
    choice - pick one option from named criteria
    score  - position on an ordered scale
"""

import json
import os
import sys
import urllib.error
import urllib.request

API_BASE = "https://llm.pratt.edu/v1"
SYSTEMONE_URL = f"{API_BASE}/systemone"
MODEL = "pratt-jev"


def load_api_key() -> str:
    """Resolve the Pratt API key from env, settings.json, or fail with a clear message."""
    key = os.environ.get("PRATT_API_KEY")
    if key:
        return key

    # Fall back to the local settings.json (git-ignored) if present.
    settings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "settings.json")
    if os.path.exists(settings_path):
        try:
            with open(settings_path, "r") as f:
                settings = json.load(f)
            key = settings.get("provider", {}).get("Pratt", {}).get("options", {}).get("apiKey")
            if key and not key.startswith("sk-pratt-YOUR_API_KEY_HERE"):
                return key
        except (json.JSONDecodeError, OSError):
            pass

    sys.stderr.write(
        "Error: No Pratt API key found.\n"
        "  Set PRATT_API_KEY=sk-pratt-... in your environment, or\n"
        "  populate settings.json (see settings.json.template).\n"
    )
    sys.exit(1)


def read_text() -> str:
    """Read text from command-line argument or stdin."""
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
    if not sys.stdin.isatty():
        return sys.stdin.read().strip()
    sys.stderr.write(
        "Usage: python3 pratt_jev_analyze.py \"<text to analyze>\"\n"
        "       cat file.txt | python3 pratt_jev_analyze.py\n"
    )
    sys.exit(1)


def build_payload(state_text: str) -> dict:
    """Build the /v1/systemone request body for pratt-jev."""
    return {
        "model": MODEL,
        "state": state_text,
        "questions": {
            "is_bug": {
                "type": "noul",
                "instructions": "Is the customer reporting a software defect or bug?",
            },
            "owner_team": {
                "type": "choice",
                "instructions": "Which team should own this issue?",
                "criteria": {
                    "payments": "Checkout, billing, or payment processing issues.",
                    "frontend": "Rendering, layout, or browser compatibility issues.",
                    "account": "Login, permissions, or profile issues.",
                    "other": "Anything not covered above.",
                },
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this issue?",
                "criteria": [
                    "Can wait for the next release",
                    "Should be fixed this week",
                    "Blocking revenue right now",
                ],
            },
        },
    }


def invoke_jev(payload: dict) -> dict:
    """POST the payload to /v1/systemone and return the parsed JSON response."""
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        SYSTEMONE_URL,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {load_api_key()}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        sys.stderr.write(f"HTTP {e.code} from {SYSTEMONE_URL}: {detail}\n")
        sys.exit(1)
    except urllib.error.URLError as e:
        sys.stderr.write(f"Request failed: {e}\n")
        sys.exit(1)


def format_answer(name: str, answer: dict) -> str:
    """Format a single decision-model answer for display."""
    atype = answer.get("type")
    lines = [f"  {name} ({atype}):"]

    if atype == "noul":
        p_yes = answer.get("noul")
        conf = answer.get("confidence")
        lines.append(f"    P(yes) = {p_yes}" + (f", confidence = {conf}" if conf is not None else ""))
        probs = answer.get("probabilities") or {}
        if probs:
            lines.append(f"    probabilities: yes={probs.get('yes')}, no={probs.get('no')}")

    elif atype == "choice":
        choice = answer.get("choice")
        conf = answer.get("confidence")
        lines.append(f"    choice = {choice}" + (f", confidence = {conf}" if conf is not None else ""))
        probs = answer.get("probabilities") or {}
        if probs:
            ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
            details = ", ".join(f"{k}={v}" for k, v in ranked)
            lines.append(f"    probabilities: {details}")

    elif atype == "score":
        score = answer.get("score")
        conf = answer.get("confidence")
        lines.append(f"    score = {score}" + (f", confidence = {conf}" if conf is not None else ""))
        probs = answer.get("probabilities") or {}
        if probs:
            details = ", ".join(f"{idx}={p}" for idx, p in sorted(probs.items(), key=lambda kv: float(kv[0])))
            lines.append(f"    probabilities by level index: {details}")
        legend = answer.get("legend") or {}
        if legend:
            lines.append(f"    legend: {legend}")

    else:
        lines.append(f"    {answer}")

    return "\n".join(lines)


def main() -> None:
    text = read_text()
    payload = build_payload(text)
    response = invoke_jev(payload)

    answers = response.get("answers") or {}
    usage = response.get("usage") or {}

    print(f"Model: {response.get('model')} | ID: {response.get('id')}")
    print(f"Input text: {text}\n")
    print("Decisions:")
    for name, answer in answers.items():
        print(format_answer(name, answer))
        print()

    if usage:
        print(f"Usage: input_tokens={usage.get('input_tokens')}, "
              f"output_tokens={usage.get('output_tokens')}, cost={usage.get('cost')}")


if __name__ == "__main__":
    main()
