# =============================================================================
# Domain-specific model testing (dynamic questions via Ollama)
# =============================================================================
#
# Unlike 39_test_finetuned.py (hardcoded questions + GPU Unsloth load), this script:
#   1. Uses a strong Ollama model (default: llama3.1) to generate NEW questions
#      about YOUR domain/topic — not random trivia.
#   2. Asks those questions to your fine-tuned Ollama model (default: from 46.py).
#
# Prerequisites:
#   - Ollama running (ollama serve or Ollama app)
#   - llama3.1 pulled (for question generation)
#   - Fine-tuned model imported (Tab 4 / ollama create …)
#
# Examples:
#   python 47_test_domain.py --topic "your domain from Tab 1"
#   python 47_test_domain.py --topic "Mars colonization" --num 8 --include-val
#   python 47_test_domain.py --topic "AI fundamentals" --model AHQ_quantize
# =============================================================================

import argparse
import json
import os
import re
import sys
import time

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PERSONAL_DATA_PATH = os.path.join(SCRIPT_DIR, "personal_data.json")
PERSONAL_VAL_PATH = os.path.join(SCRIPT_DIR, "personal_data_val.json")
QUESTIONS_CACHE_PATH = os.path.join(SCRIPT_DIR, "test_questions_generated.json")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "test_results_domain.json")

OLLAMA_BASE = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
# Question generation always uses Llama 3.1 (Tab 1 dataset gen uses the same model).
QUESTION_GENERATOR = "llama3.1"
DEFAULT_TEST_MODEL = "AHQ_quantize"
# Small batches + streaming avoid 5-minute read timeouts on one huge JSON response.
GENERATE_BATCH_SIZE = 3
GENERATE_READ_TIMEOUT = 600  # seconds between stream chunks (per batch)
GENERATE_MAX_RETRIES = 2
ANSWER_READ_TIMEOUT = 300


def _ollama_ok():
    try:
        requests.get(f"{OLLAMA_BASE}/", timeout=3)
        return True
    except requests.RequestException:
        return False


def _list_ollama_models():
    try:
        resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=10)
        resp.raise_for_status()
        names = []
        for item in resp.json().get("models", []):
            name = item.get("name", "")
            if name:
                names.append(name)
        return names
    except requests.RequestException:
        return []


def _resolve_model_name(requested):
    """Match 'AHQ_quantize' to installed 'AHQ_quantize:latest'."""
    installed = _list_ollama_models()
    if not installed:
        return None, installed
    if requested in installed:
        return requested, installed
    prefixed = f"{requested}:latest"
    if prefixed in installed:
        return prefixed, installed
    base = requested.split(":")[0]
    for name in installed:
        if name.split(":")[0] == base:
            return name, installed
    return None, installed


def _ensure_test_model(test_model):
    resolved, installed = _resolve_model_name(test_model)
    if resolved:
        return resolved

    modelfile = os.path.join(SCRIPT_DIR, "Modelfile")
    merged = os.path.join(SCRIPT_DIR, "personal_model_merged")
    print("ERROR: Fine-tuned Ollama model is not installed.")
    print(f"  Requested: {test_model}")
    print(f"  Installed: {', '.join(installed) if installed else '(none)'}")
    print()
    if os.path.isfile(modelfile) and os.path.isdir(merged):
        print("  Your export files exist. Import the model with:")
        print(f'    ollama create {test_model} --quantize q4_K_M -f "{modelfile}"')
        print("  Or use Tab 4 (Export) in the desktop app.")
    else:
        print("  Run Tab 4 (Export / 46.py) first to build Modelfile and personal_model_merged/.")
    print()
    print(f"  Then: ollama run {test_model}")
    sys.exit(1)


def _read_ollama_model_name_from_46():
    path = os.path.join(SCRIPT_DIR, "46.py")
    if not os.path.isfile(path):
        return DEFAULT_TEST_MODEL
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("OLLAMA_MODEL_NAME"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except OSError:
        pass
    return DEFAULT_TEST_MODEL


def _extract_user_questions(path):
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for item in data:
        for msg in item.get("messages", []):
            if msg.get("role") == "user":
                q = (msg.get("content") or "").strip()
                if q:
                    out.append(q)
    return out


def _parse_json_array(text):
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty model response")

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text)
        if not match:
            raise
        parsed = json.loads(match.group(0))

    if isinstance(parsed, list):
        return [str(q).strip() for q in parsed if str(q).strip()]

    if isinstance(parsed, dict):
        for key in ("questions", "items", "list"):
            val = parsed.get(key)
            if isinstance(val, list):
                return [str(q).strip() for q in val if str(q).strip()]
        q = parsed.get("question")
        if isinstance(q, str) and q.strip():
            return [q.strip()]

    raise ValueError(f"Expected JSON array of questions, got: {text[:200]!r}")


def _ensure_llama31_generator():
    resolved, installed = _resolve_model_name(QUESTION_GENERATOR)
    if not resolved:
        print("ERROR: Llama 3.1 is required for domain question generation.")
        print(f"  Install: ollama pull {QUESTION_GENERATOR}")
        print(f"  Installed models: {', '.join(installed) if installed else '(none)'}")
        sys.exit(1)
    return resolved


def _ollama_chat(prompt, model, read_timeout):
    """Use /api/chat with Llama 3.1."""
    resolved, installed = _resolve_model_name(model)
    if not resolved:
        raise ValueError(
            f"Llama 3.1 not installed. Run: ollama pull {QUESTION_GENERATOR}\n"
            f"  Installed: {', '.join(installed) or '(none)'}"
        )

    resp = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json={
            "model": resolved,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "format": "json",
            "options": {"num_predict": 2048, "temperature": 0.6},
        },
        timeout=(15, read_timeout),
    )
    resp.raise_for_status()
    msg = resp.json().get("message") or {}
    content = (msg.get("content") or "").strip()
    if content:
        return content

    thinking = (msg.get("thinking") or "").strip()
    if thinking:
        raise ValueError(f"Llama 3.1 returned only a reasoning trace (no answer).")
    raise ValueError("Llama 3.1 returned an empty reply.")


def _generate_question_batch(topic, batch_size, generator_model, avoid, read_timeout):
    avoid_sample = avoid[:40]
    extra = ""
    if len(avoid) > 40:
        extra = f"\n(Plus {len(avoid) - 40} more — do not copy.)"

    prompt = f"""Create exactly {batch_size} test questions for a fine-tuned medical/domain chat model.

Topic (every question MUST be about this):
{topic}

Rules:
- Domain-specific only; no unrelated trivia.
- Diverse phrasing.
- Do NOT copy or lightly rephrase:
{json.dumps(avoid_sample, indent=2, ensure_ascii=False)}{extra}

Return ONLY a JSON array of {batch_size} question strings, e.g. ["Q1?", "Q2?"]
"""

    last_err = None
    for attempt in range(1, GENERATE_MAX_RETRIES + 2):
        try:
            body = _ollama_chat(prompt, generator_model, read_timeout)
            return _parse_json_array(body)
        except (requests.RequestException, json.JSONDecodeError, ValueError) as e:
            last_err = e
            if attempt <= GENERATE_MAX_RETRIES:
                wait = 3 * attempt
                print(f"  Batch retry {attempt}/{GENERATE_MAX_RETRIES} in {wait}s ({e})")
                time.sleep(wait)
    raise last_err


def generate_domain_questions(
    topic,
    num_questions,
    generator_model,
    avoid=None,
    *,
    batch_size=GENERATE_BATCH_SIZE,
    read_timeout=GENERATE_READ_TIMEOUT,
):
    avoid = list(avoid or [])
    collected = []
    seen = {q.lower() for q in avoid}

    while len(collected) < num_questions:
        need = min(batch_size, num_questions - len(collected))
        batch_num = (len(collected) // batch_size) + 1
        total_batches = (num_questions + batch_size - 1) // batch_size
        print(f"  Batch {batch_num}/{total_batches}: generating {need} question(s)...", flush=True)

        batch = _generate_question_batch(
            topic, need, generator_model, avoid + collected, read_timeout
        )
        added = 0
        for q in batch:
            key = q.lower()
            if key in seen:
                continue
            seen.add(key)
            collected.append(q)
            added += 1
            if len(collected) >= num_questions:
                break
        if added == 0:
            raise ValueError("Model returned no new unique questions in this batch.")

    return collected[:num_questions]


def ask_model(model_name, question):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": question}],
        "stream": False,
    }
    resp = requests.post(f"{OLLAMA_BASE}/api/chat", json=payload, timeout=ANSWER_READ_TIMEOUT)
    if resp.status_code == 404:
        raise RuntimeError(
            f"Model '{model_name}' not found in Ollama. "
            f"Run: ollama create {model_name.split(':')[0]} --quantize q4_K_M -f Modelfile"
        )
    resp.raise_for_status()
    data = resp.json()
    msg = data.get("message") or {}
    text = (msg.get("content") or "").strip()
    if text:
        return text

    # Fallback for older/custom Modelfiles
    gen = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={"model": model_name, "prompt": question, "stream": False},
        timeout=ANSWER_READ_TIMEOUT,
    )
    gen.raise_for_status()
    return (gen.json().get("response") or "").strip()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Generate domain questions with Ollama and test your fine-tuned model."
    )
    parser.add_argument(
        "--topic",
        required=True,
        help="Same domain you used in Tab 1 / training (e.g. 'Abdurrahman Tekin research').",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=10,
        help="How many new questions to generate (default: 10).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Fine-tuned Ollama model to test (default: OLLAMA_MODEL_NAME from 46.py).",
    )
    parser.add_argument(
        "--include-val",
        action="store_true",
        help="Also run questions from personal_data_val.json (held-out from training split).",
    )
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Skip generation; only run questions already in test_questions_generated.json.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=GENERATE_BATCH_SIZE,
        help=f"Questions per Ollama call (default: {GENERATE_BATCH_SIZE}; smaller = faster per call).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=GENERATE_READ_TIMEOUT,
        help=f"Seconds to wait per batch while streaming (default: {GENERATE_READ_TIMEOUT}).",
    )
    args = parser.parse_args()

    test_model = args.model or _read_ollama_model_name_from_46()

    if not _ollama_ok():
        print("ERROR: Ollama is not reachable at", OLLAMA_BASE)
        print("Start the Ollama app or run: ollama serve")
        sys.exit(1)

    test_model = _ensure_test_model(test_model)
    generator_model = _ensure_llama31_generator()

    train_questions = _extract_user_questions(PERSONAL_DATA_PATH)
    val_questions = _extract_user_questions(PERSONAL_VAL_PATH) if args.include_val else []
    blocklist = list(dict.fromkeys(train_questions + val_questions))

    print("=" * 60)
    print("Domain test — generate questions, then query fine-tuned model")
    print("=" * 60)
    print(f"  Topic:           {args.topic}")
    print(f"  Question gen:    {generator_model} (Llama 3.1 only)")
    print(f"  Model under test:{test_model}")
    print(f"  Blocklist size:  {len(blocklist)} training/val questions (avoid duplicates)")
    print()

    generated = []
    if not args.no_generate:
        print(
            f"Generating {args.num} new domain questions "
            f"(batches of {args.batch_size}, timeout {args.timeout}s per batch)...\n"
        )
        try:
            generated = generate_domain_questions(
                args.topic,
                args.num,
                generator_model,
                avoid=blocklist,
                batch_size=max(1, args.batch_size),
                read_timeout=args.timeout,
            )
        except Exception as e:
            print(f"\nERROR generating questions: {e}")
            if os.path.isfile(QUESTIONS_CACHE_PATH):
                print(
                    "\nTip: Re-run with --no-generate to use cached questions in\n"
                    f"  {QUESTIONS_CACHE_PATH}"
                )
            print(
                "\nIf timeouts continue:\n"
                "  --batch-size 2 --num 5\n"
                f"  Ensure Llama 3.1 is installed: ollama pull {QUESTION_GENERATOR}"
            )
            sys.exit(1)
        if len(generated) < args.num:
            print(f"  Warning: got {len(generated)} questions, asked for {args.num}.")
        with open(QUESTIONS_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {"topic": args.topic, "questions": generated},
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"  Saved generated questions to: {QUESTIONS_CACHE_PATH}\n")
    elif os.path.isfile(QUESTIONS_CACHE_PATH):
        with open(QUESTIONS_CACHE_PATH, "r", encoding="utf-8") as f:
            cached = json.load(f)
        generated = cached.get("questions", [])
        print(f"Loaded {len(generated)} questions from cache.\n")
    else:
        print("ERROR: --no-generate set but test_questions_generated.json is missing.")
        sys.exit(1)

    all_questions = []
    seen = set()
    for label, q in [("val", q) for q in val_questions] + [("generated", q) for q in generated]:
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        all_questions.append((label, q))

    if not all_questions:
        print("No questions to run.")
        sys.exit(1)

    results = []
    for i, (source, question) in enumerate(all_questions, 1):
        print("-" * 60)
        print(f"[{i}/{len(all_questions)}] ({source}) {question}\n")
        try:
            answer = ask_model(test_model, question)
        except Exception as e:
            answer = f"<ERROR: {e}>"
        print(f"Answer:\n{answer}\n")
        results.append(
            {"source": source, "question": question, "answer": answer, "model": test_model}
        )

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"topic": args.topic, "model": test_model, "results": results},
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("=" * 60)
    print(f"Done. Full results saved to: {RESULTS_PATH}")
    print("=" * 60)
    print("\nTips:")
    print("  - GPU compare uses questions from personal_data.json (39_test_finetuned.py).")
    print("  - Re-run with --no-generate to reuse test_questions_generated.json.")
    print(f"  - Chat manually: ollama run {test_model.split(':')[0]}")


if __name__ == "__main__":
    main()
