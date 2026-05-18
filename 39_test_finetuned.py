# =============================================================================
# Test the fine-tuned SmolLM2 model (base + LoRA adapters)
# =============================================================================
#
# WHAT THIS SCRIPT DOES:
#   1. Loads the ORIGINAL (base) SmolLM2-1.7B-Instruct model — the one that
#      has never seen your personal data. Asks it a set of test questions.
#   2. Unloads the base model from GPU memory to free space.
#   3. Loads the FINE-TUNED model — same base model but with the LoRA adapters
#      (trained by 38_finetune.py) applied on top. Asks the same questions.
#   4. Prints both sets of answers side by side so you can directly compare
#      what the model knew before and after fine-tuning.
#
# WHY TWO SEPARATE LOADS?
#   The base model and fine-tuned model cannot be in GPU memory at the same
#   time (each takes ~2-4 GB in 4-bit). So we load one, get answers, delete
#   it, then load the other. This "delete and reload" approach is standard
#   when comparing models on a single GPU.
#
# Run:
#   python 39_test_finetuned.py
# =============================================================================


# --- IMPORTS ----------------------------------------------------------------
# os: for building file paths (to find the saved LoRA adapters).
# torch: PyTorch — runs the model on the GPU. Also used for torch.no_grad()
#   (disables gradient computation during inference to save memory and speed).
# FastLanguageModel: Unsloth's optimized loader that handles both base models
#   and models with LoRA adapters. It auto-detects adapter files and merges them.
import json
import os
import sys
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from unsloth import FastLanguageModel


# =============================================================================
# CONFIG
# =============================================================================

# MODEL_NAME: The same base model used during fine-tuning. This is the
# "untouched" model we compare against. It knows language and general facts
# but nothing about your personal data.
MODEL_NAME = "unsloth/SmolLM2-1.7B-Instruct"

# ADAPTER_DIR: Path to the folder where 38_finetune.py saved the LoRA adapters.
# This folder contains:
#   - adapter_model.safetensors: the trained LoRA weight matrices
#   - adapter_config.json: LoRA settings (rank, alpha, which layers)
#   - tokenizer files (tokenizer.json, special_tokens_map.json, etc.)
# When we load from this directory, Unsloth reads the adapter_config.json,
# downloads the base model (if not cached), and applies the adapters on top.
ADAPTER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "personal_model")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PERSONAL_DATA_PATH = os.path.join(SCRIPT_DIR, "personal_data.json")
PERSONAL_VAL_PATH = os.path.join(SCRIPT_DIR, "personal_data_val.json")

# MAX_SEQ_LENGTH: Maximum token length for the input prompt + generated answer.
# Must match or exceed what was used during training (2048 in 38_finetune.py).
MAX_SEQ_LENGTH = 2048

# LOAD_IN_4BIT: Load the base model weights in 4-bit precision (same as training).
# This keeps VRAM usage low and ensures the model behaves the same way it did
# during training (different precision = slightly different outputs).
LOAD_IN_4BIT = True  # match 38_finetune.py LOW_RESOURCE / 8 GB GPUs

# MAX_NEW_TOKENS: Maximum number of tokens the model can generate in its answer.
# 256 tokens ≈ roughly 200 words. If the model's answer is shorter, it will
# stop early (when it outputs the end-of-sequence token). If the answer would
# be longer, it gets cut off at 256 tokens.
MAX_NEW_TOKENS = 256

def load_test_questions():
    """Use only user questions from your dataset (train + val). No hardcoded or trivia."""
    questions = []
    seen = set()
    for path in (PERSONAL_DATA_PATH, PERSONAL_VAL_PATH):
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            for msg in item.get("messages", []):
                if msg.get("role") != "user":
                    continue
                q = (msg.get("content") or "").strip()
                if not q:
                    continue
                key = q.lower()
                if key in seen:
                    continue
                seen.add(key)
                questions.append(q)
    if not questions:
        raise FileNotFoundError(
            "No test questions found.\n"
            f"  Add Q&A pairs to {PERSONAL_DATA_PATH} (Tab 1), then train.\n"
            "  For extra domain-only questions, use 47_test_domain.py after Ollama export."
        )
    return questions


# =============================================================================
# ANSWER GENERATION
# =============================================================================

def generate_answer(model, tokenizer, question):
    """Generate an answer for a single question.

    Steps:
    1. Wrap the question in the model's chat template format.
       This adds special tokens like <|im_start|>user\n...<|im_end|>
       that the model was trained to recognize as conversation structure.
       add_generation_prompt=True adds the "<|im_start|>assistant\n" prefix
       so the model knows it should start generating the assistant's reply.

    2. Tokenize the formatted prompt into token IDs and move them to GPU.

    3. Run model.generate() to produce new tokens one by one (autoregressive):
       - temperature=None, top_p=None, do_sample=False → GREEDY decoding.
         The model always picks the single most probable next token.
         This makes outputs deterministic (same question → same answer every time).
       - pad_token_id: tells the model which token ID means "padding" so it
         doesn't crash if the tokenizer doesn't have one defined.

    4. Extract only the NEW tokens (the model's answer), ignoring the prompt
       tokens that were part of the input.

    5. Decode the token IDs back into human-readable text.
    """
    # Step 1: Format the question using the model's chat template
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Step 2: Convert the formatted text into token IDs on the GPU
    # return_tensors="pt" → return PyTorch tensors (required for model input)
    # .to("cuda") → move tensors to GPU memory where the model lives
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    # Step 3: Generate the answer
    # torch.no_grad() disables gradient tracking — we're not training here,
    # just generating. This saves ~50% memory and is faster.
    with torch.no_grad():
        outputs = model.generate(
            **inputs,                    # Pass input_ids and attention_mask
            max_new_tokens=MAX_NEW_TOKENS,  # Stop after 256 new tokens
            temperature=None,            # No temperature scaling (greedy)
            top_p=None,                  # No nucleus sampling (greedy)
            do_sample=False,             # Greedy: always pick the top token
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Step 4: The output contains [prompt_tokens + answer_tokens].
    # We only want the answer tokens, so we slice off the prompt part.
    prompt_length = inputs["input_ids"].shape[1]  # Number of tokens in the prompt
    response_ids = outputs[0][prompt_length:]      # Everything after the prompt

    # Step 5: Convert token IDs back to text
    # skip_special_tokens=True removes control tokens like <|im_end|> from output
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip()


# =============================================================================
# MAIN
# =============================================================================

def main():
    # --- Safety checks ---
    if not torch.cuda.is_available():
        raise RuntimeError("This script needs a CUDA GPU.")

    # Make sure the adapter directory exists (user must run 38_finetune.py first)
    if not os.path.isdir(ADAPTER_DIR):
        raise FileNotFoundError(
            f"Adapter directory not found: {ADAPTER_DIR}\n"
            f"Run 38_finetune.py first to train and save the LoRA adapters."
        )

    test_questions = load_test_questions()
    print(f"Testing {len(test_questions)} question(s) from your dataset only.\n")

    # =========================================================================
    # PHASE 1: Load and test the BASE model (no fine-tuning)
    # =========================================================================
    # This gives us a "before" snapshot — what does the model say about you
    # when it has never been trained on your data?
    print("=" * 70)
    print("Loading BASE model (no fine-tuning)...")
    print("=" * 70)
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,       # Download/load the original model
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,                  # Auto-detect best dtype (bfloat16)
        load_in_4bit=LOAD_IN_4BIT,   # 4-bit quantization for low VRAM
    )
    # for_inference(): switches the model from training mode to inference mode.
    # This disables dropout, enables optimized attention kernels, and makes
    # generation faster. Always call this before using model.generate().
    FastLanguageModel.for_inference(base_model)

    # Ask every test question to the base model and store the answers
    print("\nGenerating BASE model answers...\n")
    base_answers = {}
    for q in test_questions:
        base_answers[q] = generate_answer(base_model, tokenizer, q)

    # Free the base model from GPU memory. The GPU has limited VRAM and
    # cannot hold two copies of a 1.7B model simultaneously.
    # del: removes the Python reference to the model object.
    # torch.cuda.empty_cache(): tells PyTorch to release the freed GPU memory
    # back to the CUDA allocator so it's available for the next model load.
    del base_model
    torch.cuda.empty_cache()

    # =========================================================================
    # PHASE 2: Load and test the FINE-TUNED model (base + LoRA adapters)
    # =========================================================================
    # When model_name points to the adapter directory, Unsloth automatically:
    #   1. Reads adapter_config.json to find the base model name
    #   2. Loads the base model (from cache, no re-download)
    #   3. Applies the LoRA adapter weights on top
    # The result is a model that behaves like the base model but with your
    # personal knowledge "patched in" through the adapter weights.
    print("=" * 70)
    print("Loading FINE-TUNED model (base + LoRA adapters)...")
    print("=" * 70)
    ft_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_DIR,      # Points to the LoRA adapter folder
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(ft_model)

    # Ask the same questions to the fine-tuned model
    print("\nGenerating FINE-TUNED model answers...\n")
    ft_answers = {}
    for q in test_questions:
        ft_answers[q] = generate_answer(ft_model, tokenizer, q)

    # =========================================================================
    # PHASE 3: Print side-by-side comparison
    # =========================================================================
    # Compare answers on your dataset questions only (same domain as training).
    print("\n" + "=" * 70)
    print("COMPARISON: BASE vs FINE-TUNED")
    print("=" * 70)

    for q in test_questions:
        print(f"\n{'─' * 70}")
        print(f"  QUESTION: {q}")
        print(f"{'─' * 70}")
        print(f"  BASE MODEL:")
        for line in base_answers[q].split("\n"):
            print(f"    {line}")
        print()
        print(f"  FINE-TUNED:")
        for line in ft_answers[q].split("\n"):
            print(f"    {line}")

    print(f"\n{'=' * 70}")
    print("Done! If the fine-tuned answers are weak, add more examples to")
    print("personal_data.json and re-run 38_finetune.py.")
    print("=" * 70)


if __name__ == "__main__":
    main()
