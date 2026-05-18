# =============================================================================
# Dataset generation helpers (Tab 1) — same JSON format as before
# =============================================================================

import json
import random
import re
import time

import requests

OLLAMA_BASE = "http://localhost:11434".rstrip("/")
QUESTION_GENERATOR = "llama3.1"
BATCH_SIZE = 5
MIN_QUESTION_LEN = 12
MIN_ANSWER_LEN = 40
GENERATE_TIMEOUT = 600
MAX_RETRIES = 2


def _resolve_model(name):
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=10)
        resp.raise_for_status()
        names = [m.get("name", "") for m in resp.json().get("models", [])]
    except requests.RequestException:
        return None
    if name in names:
        return name
    tagged = f"{name}:latest"
    if tagged in names:
        return tagged
    for n in names:
        if n.split(":")[0] == name:
            return n
    return None


def _ollama_chat_json(prompt, model, read_timeout=GENERATE_TIMEOUT):
    resolved = _resolve_model(model) or model
    resp = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json={
            "model": resolved,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "format": "json",
            "options": {"num_predict": 4096, "temperature": 0.8},
        },
        timeout=(15, read_timeout),
    )
    resp.raise_for_status()
    content = (resp.json().get("message") or {}).get("content") or ""
    if not content.strip():
        raise ValueError("Llama 3.1 returned an empty response.")
    return content.strip()


def _parse_dataset_payload(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text).strip()
    parsed = json.loads(text)
    if isinstance(parsed, dict):
        for key in ("data", "examples", "items", "dataset"):
            if isinstance(parsed.get(key), list):
                return parsed[key]
        if "messages" in parsed:
            return [parsed]
    if isinstance(parsed, list):
        return parsed
    raise ValueError(f"Expected JSON array of examples, got {type(parsed).__name__}")


def normalize_example(item):
    """Convert one item to standard messages format, or None if invalid."""
    if not isinstance(item, dict):
        return None
    if "messages" in item:
        msgs = item["messages"]
    elif "question" in item and "answer" in item:
        msgs = [
            {"role": "user", "content": str(item["question"]).strip()},
            {"role": "assistant", "content": str(item["answer"]).strip()},
        ]
    else:
        return None

    if not isinstance(msgs, list) or len(msgs) < 2:
        return None

    user = next((m for m in msgs if m.get("role") == "user"), None)
    asst = next((m for m in msgs if m.get("role") == "assistant"), None)
    if not user or not asst:
        return None

    q = (user.get("content") or "").strip()
    a = (asst.get("content") or "").strip()
    if not q or not a:
        return None
    if "[" in q and "Question" in q and "]" in q:
        return None
    if len(q) < MIN_QUESTION_LEN or len(a) < MIN_ANSWER_LEN:
        return None

    return {
        "messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]
    }


def clean_dataset(examples):
    """Deduplicate and drop weak or malformed rows."""
    out = []
    seen_q = set()
    dropped = 0
    for raw in examples:
        ex = normalize_example(raw)
        if not ex:
            dropped += 1
            continue
        key = ex["messages"][0]["content"].lower()
        if key in seen_q:
            dropped += 1
            continue
        seen_q.add(key)
        out.append(ex)
    return out, dropped


GENERIC_ASPECTS = [
    "Foundations & Core Definitions (basic principles, terminology, fundamental concepts, and primary goals)",
    "Key Components & Anatomy (main elements, structure, rules, features, or parts that make it up)",
    "Processes & Workflows (how it works step-by-step, methodologies, procedures, or operations)",
    "Advanced Concepts & Variations (complex details, variations, advanced techniques, or related sub-types)",
    "Practical Application & Real-world Examples (best practices, typical use cases, scenarios, and actual implementation)",
    "Common Challenges, Troubleshooting & Solutions (errors, risks, limitations, pitfalls, and how to resolve or prevent them)",
    "Comparisons & Alternatives (contrast with similar concepts, pros and cons, when to use this vs. others)",
    "Evaluation, Diagnostics & Quality Control (how to measure success, inspect quality, test, or evaluate performance)",
    "Optimization & Advanced Tuning (how to improve efficiency, speed up, polish, or fine-tune)",
    "Guidelines, Safety & Professional Standards (best practices, safety guidelines, compliance, and outlook)"
]


def _batch_prompt(topic, batch_size, context, avoid_questions, batch_num=1):
    ctx = ""
    if context.strip():
        ctx = f"""
Reference material (use for accurate answers when relevant):
{context.strip()}
"""
    block = ""
    if avoid_questions:
        sample = avoid_questions[:25]
        block = f"""
Do NOT repeat or lightly rephrase these existing questions:
{json.dumps(sample, ensure_ascii=False)}
"""

    aspect = GENERIC_ASPECTS[(batch_num - 1) % len(GENERIC_ASPECTS)]

    return f"""You are building a training dataset for a domain-specific chatbot.

Topic: {topic}
{ctx}
For this specific batch (Batch #{batch_num}), focus heavily on the following aspect of "{topic}":
Aspect Focus: {aspect}

Create exactly {batch_size} high-quality, highly specific, and unique question-answer pairs that explore this aspect.

Quality rules:
- Questions must be specific to "{topic}" and focus on the aspect above (clinical, technical, or factual as appropriate).
- Mix question types: definitions, procedures, diagnostics, risks, indications, follow-up, comparisons.
- Answers must be 2-5 complete sentences with concrete detail (names, steps, criteria, values where sensible).
- No placeholders like [Question 1] or generic filler.
- No duplicate or near-duplicate questions.
{block}
Return ONLY a JSON object with an "examples" key containing a list of exactly {batch_size} Q&A pairs. Each pair in the list must have a "messages" list.
Example structure:
{{
  "examples": [
    {{
      "messages": [
        {{"role": "user", "content": "..."}},
        {{"role": "assistant", "content": "..."}}
      ]
    }}
  ]
}}
"""


def _generate_batch(topic, batch_size, context, avoid, batch_num=1, log=None):
    log = log or (lambda _=None: None)
    prompt = _batch_prompt(topic, batch_size, context, avoid, batch_num)
    last_err = None
    for attempt in range(1, MAX_RETRIES + 2):
        try:
            body = _ollama_chat_json(prompt, QUESTION_GENERATOR)
            raw = _parse_dataset_payload(body)
            cleaned, dropped = clean_dataset(raw)
            if not cleaned:
                raise ValueError("Batch produced no valid examples after filtering.")
            if dropped:
                log(f"    Filtered {dropped} weak/duplicate row(s) in batch.\n")
            return cleaned
        except Exception as e:
            last_err = e
            if attempt <= MAX_RETRIES:
                wait = 3 * attempt
                log(f"    Batch retry {attempt}/{MAX_RETRIES} in {wait}s ({e})\n")
                time.sleep(wait)
    raise last_err


def generate_dataset(topic, num_pairs, context="", log=None):
    """Generate num_pairs examples using Llama 3.1 in small batches."""
    log = log or (lambda _: None)
    if not _resolve_model(QUESTION_GENERATOR):
        raise RuntimeError(
            f"{QUESTION_GENERATOR} is not installed. Run: ollama pull {QUESTION_GENERATOR}"
        )

    collected = []
    seen = set()
    batch_num = 0
    consecutive_empty_batches = 0
    while len(collected) < num_pairs:
        need = min(BATCH_SIZE, num_pairs - len(collected))
        batch_num += 1
        log(f"  Batch {batch_num}: requesting {need} pair(s)...\n")
        batch = _generate_batch(
            topic,
            need,
            context,
            [ex["messages"][0]["content"] for ex in collected],
            batch_num=batch_num,
            log=log,
        )
        added = 0
        for ex in batch:
            key = ex["messages"][0]["content"].lower().strip()
            # Fuzzy match to drop punctuation and redundant whitespace
            normalized_q = re.sub(r'[^\w\s]', '', key).strip()
            if normalized_q in seen:
                continue
            seen.add(normalized_q)
            collected.append(ex)
            added += 1
            if len(collected) >= num_pairs:
                break
        if added == 0:
            consecutive_empty_batches += 1
            log(f"    Warning: No new unique examples added in Batch {batch_num}.\n")
            if consecutive_empty_batches >= 3:
                raise RuntimeError(
                    "Failed to generate new unique examples after multiple consecutive attempts. "
                    "Please try a broader topic or provide more context notes."
                )
        else:
            consecutive_empty_batches = 0

    return collected[:num_pairs]


def split_train_val(examples, train_ratio=0.8):
    """Shuffle then split; never duplicate the same row in both files."""
    data = list(examples)
    random.shuffle(data)
    if len(data) == 1:
        return data, [data[0]]
    split_idx = max(1, int(len(data) * train_ratio))
    if split_idx >= len(data):
        split_idx = len(data) - 1
    return data[:split_idx], data[split_idx:]
