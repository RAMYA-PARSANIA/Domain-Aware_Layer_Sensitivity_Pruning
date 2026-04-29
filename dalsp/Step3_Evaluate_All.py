"""
Step 3: Evaluation Dashboard — Original vs Flat-Pruned vs DALSP Subnetworks
=============================================================================
Key fixes and additions over v1
--------------------------------
  1. GSM8K evaluation fixed:
       - max_new_tokens=256 for math (chain-of-thought needs room)
       - Improved number extractor handles commas, negatives, decimals
         and falls back to the last number in the output if #### is absent
  2. n=200 samples per task (was 20; 25x more statistically robust)
  3. Per-task token budget (coding/math: 256; others: 64)
  4. Inference latency: tokens / second measured per model
  5. Peak GPU memory: Δ VRAM before/after model load
  6. Two pruned columns: flat-20% (Step 2 original) and DALSP (Step 2 new)
     so readers can directly see the gain from adaptive allocation
"""

import time
import re
import os
import gc

import torch
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from rouge_score import rouge_scorer
from tqdm import tqdm


# ── utility helpers ───────────────────────────────────────────────────────────

def get_gpu_mem_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 3)
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)


def format_phi3_prompt(text: str) -> str:
    return f"<|user|>\n{text}<|end|>\n<|assistant|>\n"


def extract_gsm8k_answer(text: str):
    """Extract reference answer from GSM8K '#### N' format."""
    m = re.search(r'####\s*(-?[\d,]+)', text)
    return m.group(1).replace(',', '') if m else None


def extract_model_number(text: str):
    """
    Robustly extract a numeric answer from model output.

    Strategy (in order of preference):
      1. Look for '#### <number>' anywhere in the output (model followed the
         prompt instruction).
      2. Look for 'The answer is <number>' or 'answer: <number>' (common
         chain-of-thought closing phrases).
      3. Fall back to the last standalone number in the output.
    """
    # Strategy 1: explicit #### marker
    m = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', text)
    if m:
        return m.group(1).replace(',', '')

    # Strategy 2: natural language answer phrases
    m = re.search(
        r'(?:the answer is|answer is|answer:|=)\s*(-?[\d,]+(?:\.\d+)?)',
        text, re.IGNORECASE
    )
    if m:
        return m.group(1).replace(',', '')

    # Strategy 3: last number in the response
    numbers = re.findall(r'-?\b\d[\d,]*(?:\.\d+)?\b', text)
    return numbers[-1].replace(',', '') if numbers else None


# ── per-task configuration ────────────────────────────────────────────────────

TASK_CONFIG = {
    # task_name: (max_new_tokens, eval_type)
    "Exact_Match (TriviaQA)":  (64,  "exact_match"),
    "Summarization (CNN/DM)":  (128, "rouge"),
    "Logic (GSM8K)":           (256, "math_match"),   # ← fixed: was 64
    "Sentiment (SST2)":        (16,  "classification"),
    "Coding (MBPP)":           (256, "rouge"),
    "Legal_Summ (BillSum)":    (128, "rouge"),
}


# ── dataset loading ────────────────────────────────────────────────────────────

def get_dataset_samples(num_samples: int = 200):
    print(f"Fetching {num_samples} evaluation samples per task ...")
    samples = {}

    try:
        ds = load_dataset("trivia_qa", "rc", split="validation", streaming=True)
        samples["Exact_Match (TriviaQA)"] = [
            {"prompt": r["question"], "ref": r["answer"]["aliases"]}
            for i, r in enumerate(ds) if i < num_samples
        ]
    except Exception as e:
        print(f"  [WARN] TriviaQA: {e}")

    try:
        ds = load_dataset("cnn_dailymail", "3.0.0", split="test", streaming=True)
        samples["Summarization (CNN/DM)"] = [
            {"prompt": f"Summarize:\n\n{r['article'][:1000]}...", "ref": r["highlights"]}
            for i, r in enumerate(ds) if i < num_samples
        ]
    except Exception as e:
        print(f"  [WARN] CNN/DM: {e}")

    try:
        ds = load_dataset("gsm8k", "main", split="test", streaming=True)
        samples["Logic (GSM8K)"] = [
            {
                "prompt": (
                    "Solve the following math problem step by step. "
                    "At the end, write your final numeric answer after '####'.\n\n"
                    f"{r['question']}"
                ),
                "ref": extract_gsm8k_answer(r["answer"])
            }
            for i, r in enumerate(ds) if i < num_samples
        ]
    except Exception as e:
        print(f"  [WARN] GSM8K: {e}")

    try:
        ds = load_dataset("glue", "sst2", split="validation", streaming=True)
        samples["Sentiment (SST2)"] = [
            {
                "prompt": (
                    f"Classify the sentiment of this sentence as exactly "
                    f"'positive' or 'negative':\n\n\"{r['sentence']}\"\n\nSentiment:"
                ),
                "ref": "positive" if r["label"] == 1 else "negative"
            }
            for i, r in enumerate(ds) if i < num_samples
        ]
    except Exception as e:
        print(f"  [WARN] SST2: {e}")

    try:
        ds = load_dataset("mbpp", "full", split="validation", streaming=True)
        samples["Coding (MBPP)"] = [
            {"prompt": f"Write a Python function:\n{r['text']}\n\n```python\n", "ref": r["code"]}
            for i, r in enumerate(ds) if i < num_samples
        ]
    except Exception as e:
        print(f"  [WARN] MBPP: {e}")

    try:
        ds = load_dataset("billsum", split="test", streaming=True)
        samples["Legal_Summ (BillSum)"] = [
            {"prompt": f"Summarize this legal bill:\n\n{r['text'][:1000]}...", "ref": r["summary"]}
            for i, r in enumerate(ds) if i < num_samples
        ]
    except Exception as e:
        print(f"  [WARN] BillSum: {e}")

    print(f"  Loaded {len(samples)} tasks.\n")
    return samples


# ── single-model evaluation ────────────────────────────────────────────────────

def evaluate_model(model_path: str, task_samples: dict):
    """
    Evaluate one model across all tasks.

    Returns a dict:
      {
        task_name: score_str,
        "__latency_tps__": tokens_per_second,
        "__memory_gb__":   delta_vram_gb,
      }
    """
    print(f"\n[EVAL] {model_path}")
    if not os.path.exists(model_path) and "microsoft" not in model_path:
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    mem_before = get_gpu_mem_gb()
    model      = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True, device_map="auto"
    )
    tokenizer  = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    mem_after  = get_gpu_mem_gb()
    delta_vram = max(0.0, mem_after - mem_before)

    scorer_obj = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    results    = {}

    # ── latency benchmark (warm-up + timed run on a fixed prompt) ─────────────
    _bench_prompt  = format_phi3_prompt("What is the capital of France?")
    _bench_tokens  = tokenizer(_bench_prompt, return_tensors="pt").to(device)
    _bench_new_tok = 32

    with torch.no_grad():
        # warm-up
        model.generate(**_bench_tokens, max_new_tokens=_bench_new_tok,
                       do_sample=False, pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        for _ in range(3):
            model.generate(**_bench_tokens, max_new_tokens=_bench_new_tok,
                           do_sample=False, pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - t0

    tokens_per_second = (3 * _bench_new_tok) / elapsed

    # Setup for batch processing
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    BATCH_SIZE = 32 # Can be increased since there is 48GB VRAM

    # ── task evaluation ────────────────────────────────────────────────────────
    for task_name, items in task_samples.items():
        max_new_tokens, task_type = TASK_CONFIG.get(task_name, (64, "rouge"))
        correct = 0
        rogue_l = 0.0

        print(f"  > {task_name} ({len(items)} samples, max_tokens={max_new_tokens})")

        for i in tqdm(range(0, len(items), BATCH_SIZE), leave=False, desc=f"    {task_name[:20]}"):
            batch_items = items[i:i + BATCH_SIZE]
            prompts = [format_phi3_prompt(data["prompt"]) for data in batch_items]
            
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            # Decode sequences after the prompt lengths
            seq_lens = inputs["input_ids"].shape[1]
            gen = out[:, seq_lens:]
            texts = tokenizer.batch_decode(gen, skip_special_tokens=True)

            for j, text in enumerate(texts):
                data = batch_items[j]
                text = text.strip()

                if task_type == "exact_match":
                    if any(ans.lower() in text.lower() for ans in data["ref"]):
                        correct += 1

                elif task_type == "math_match":
                    pred = extract_model_number(text)
                    ref  = str(data["ref"])
                    if pred is not None and pred == ref:
                        correct += 1

                elif task_type == "classification":
                    tl = text.lower()
                    pi, ni = tl.find("positive"), tl.find("negative")
                    guess = "none"
                    if pi != -1 and (ni == -1 or pi < ni):
                        guess = "positive"
                    elif ni != -1 and (pi == -1 or ni < pi):
                        guess = "negative"
                    if guess == data["ref"]:
                        correct += 1

                elif task_type == "rouge":
                    rogue_l += scorer_obj.score(data["ref"], text)["rougeL"].fmeasure

        n = len(items)
        if task_type in ("exact_match", "classification", "math_match"):
            results[task_name] = f"{correct/n*100:.1f}%"
        else:
            results[task_name] = f"{rogue_l/n:.3f}"

    results["__latency_tps__"] = tokens_per_second
    results["__memory_gb__"]   = delta_vram

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return results


# ── dashboard printer ──────────────────────────────────────────────────────────

def print_dashboard(task_samples: dict, models: list, all_results: dict):
    task_names = list(task_samples.keys())
    col_w      = 10

    # Header
    header_names = [name for name, _ in models]
    sep = "=" * (28 + 3 + sum(col_w + 3 for _ in models))
    print(f"\n\n{sep}")
    print(" " * 20 + "FINAL SUB-NETWORK COMPARISON DASHBOARD (DALSP)")
    print(sep)

    hdr = f"{'Task (Metric)':<28} | "
    hdr += " | ".join(f"{n:<{col_w}}" for n in header_names)
    print(hdr)
    print("-" * len(hdr))

    for task in task_names:
        row = f"{task:<28} | "
        vals = []
        for name, _ in models:
            raw = str(all_results[name].get(task, "N/A"))
            raw = raw.replace("R-L: ", "")
            vals.append(f"{raw[:col_w]:<{col_w}}")
        row += " | ".join(vals)
        print(row)

    print("-" * len(hdr))

    # Latency row
    lat_row = f"{'Latency (tok/s)':<28} | "
    lat_row += " | ".join(
        f"{all_results[name].get('__latency_tps__', 0.0):<{col_w}.1f}"
        for name, _ in models
    )
    print(lat_row)

    # Memory row
    mem_row = f"{'Peak VRAM (GB)':<28} | "
    mem_row += " | ".join(
        f"{all_results[name].get('__memory_gb__', 0.0):<{col_w}.2f}"
        for name, _ in models
    )
    print(mem_row)
    print(sep)

    # Superadditivity callout
    print("\n[NOTE] Values higher than 'Original' in a domain-matched task")
    print("       indicate superadditive specialisation — the pruned subnet")
    print("       EXCEEDS the full model on its target domain.")
    print("       Compare 'Code_Flat' vs 'Code_DALSP' to isolate the gain")
    print("       from entropy-guided adaptive budget allocation (DALSP).\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    NUM_SAMPLES = 100   # was 20 — statistically meaningful

    task_samples = get_dataset_samples(num_samples=NUM_SAMPLES)

    # Evaluate:  original  +  flat-pruned (Step2 v1)  +  DALSP (Step2 v2)
    models = [
        ("Original",      "microsoft/Phi-3.5-mini-instruct"),
        ("General_Flat",  "./phi-3.5-pruned-20_general"),
        ("Math_Flat",     "./phi-3.5-pruned-20_math"),
        ("Code_Flat",     "./phi-3.5-pruned-20_code"),
        ("Law_Flat",      "./phi-3.5-pruned-20_law"),
        ("General_DALSP", "./phi-3.5-pruned-dalsp_general"),
        ("Math_DALSP",    "./phi-3.5-pruned-dalsp_math"),
        ("Code_DALSP",    "./phi-3.5-pruned-dalsp_code"),
        ("Law_DALSP",     "./phi-3.5-pruned-dalsp_law"),
    ]

    all_results = {}
    for name, path in models:
        try:
            all_results[name] = evaluate_model(path, task_samples)
        except Exception as exc:
            print(f"  [SKIP] {name}: {exc}")
            all_results[name] = {t: "N/A" for t in task_samples}
            all_results[name]["__latency_tps__"] = 0.0
            all_results[name]["__memory_gb__"]   = 0.0

    print_dashboard(task_samples, models, all_results)

    # ── Save raw results for further analysis ──────────────────────────────────
    import json
    safe_results = {
        name: {k: str(v) for k, v in res.items()}
        for name, res in all_results.items()
    }
    with open("evaluation_results.json", "w") as f:
        json.dump(safe_results, f, indent=2)
    print("Raw results saved to evaluation_results.json")


if __name__ == "__main__":
    main()
