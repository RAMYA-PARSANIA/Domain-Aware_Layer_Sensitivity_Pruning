import time
import torch
import psutil
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from rouge_score import rouge_scorer
from tqdm import tqdm
import gc

def get_memory_usage(device="cuda"):
    if torch.cuda.is_available(): return torch.cuda.memory_allocated() / (1024**3)
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)

def format_phi3_prompt(text): return f"<|user|>\n{text}<|end|>\n<|assistant|>\n"
def extract_gsm8k_answer(text):
    match = re.search(r'####\s*(-?[0-9.,]+)', text)
    return match.group(1).replace(',', '') if match else None
def extract_model_number(text):
    numbers = re.findall(r'-?[0-9.,]+', text)
    return numbers[-1].replace(',', '') if numbers else None

def get_dataset_samples(num_samples=25):
    print(f"Fetching {num_samples} evaluation samples from each task...")
    samples = {}
    try:
        ds_trivia = load_dataset("trivia_qa", "rc", split="validation", streaming=True)
        samples["Exact_Match (TriviaQA)"] = {"items": [{"prompt": r["question"], "ref": r["answer"]["aliases"]} for i, r in enumerate(iter(ds_trivia)) if i < num_samples], "type": "exact_match"}

        ds_cnn = load_dataset("cnn_dailymail", "3.0.0", split="test", streaming=True)
        samples["Summarization (CNN/DM)"] = {"items": [{"prompt": f"Summarize:\n\n{r['article'][:1000]}...", "ref": r["highlights"]} for i, r in enumerate(iter(ds_cnn)) if i < num_samples], "type": "rouge"}

        ds_gsm8k = load_dataset("gsm8k", "main", split="test", streaming=True)
        samples["Logic (GSM8K)"] = {"items": [{"prompt": f"Answer math problem. End with '#### [number]'.\n\n{r['question']}", "ref": extract_gsm8k_answer(r["answer"])} for i, r in enumerate(iter(ds_gsm8k)) if i < num_samples], "type": "math_match"}
        
        ds_sst = load_dataset("glue", "sst2", split="validation", streaming=True)
        samples["Sentiment (SST2)"] = {"items": [{"prompt": f"Classify sentiment 'positive' or 'negative':\n\n\"{r['sentence']}\"\n\nSentiment:", "ref": "positive" if r["label"]==1 else "negative"} for i, r in enumerate(iter(ds_sst)) if i < num_samples], "type": "classification"}

        ds_code = load_dataset("mbpp", "full", split="validation", streaming=True)
        samples["Coding (MBPP)"] = {"items": [{"prompt": f"Write a Python function:\n{r['text']}\n\n```python\n", "ref": r["code"]} for i, r in enumerate(iter(ds_code)) if i < num_samples], "type": "rouge"}

        ds_law = load_dataset("billsum", split="test", streaming=True)
        samples["Legal_Summ (BillSum)"] = {"items": [{"prompt": f"Summarize legal bill:\n\n{r['text'][:1000]}...", "ref": r["summary"]} for i, r in enumerate(iter(ds_law)) if i < num_samples], "type": "rouge"}
    except Exception as e: print(f"Warning fetching: {e}")
    return samples

def evaluate_model(model_name_or_path, tasks):
    print(f"\n[EVALUATING] {model_name_or_path.upper()}...")
    if not os.path.exists(model_name_or_path) and "microsoft" not in model_name_or_path:
        raise Exception("Model folder not found.")
        
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=dtype, trust_remote_code=True, device_map="cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    results = {}
    
    for task_name, info in tasks.items():
        print(f" > Task: {task_name}")
        items, task_type = info["items"], info["type"]
        correct = 0
        rL = 0.0
        
        for i, data in enumerate(tqdm(items, leave=False)):
            inputs = tokenizer(format_phi3_prompt(data["prompt"]), return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_new_tokens=64, temperature=0.1, do_sample=True, pad_token_id=tokenizer.eos_token_id)
            gen_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            
            if task_type == "exact_match":
                if any(ans.lower() in text.lower() for ans in data["ref"]): correct += 1
            elif task_type == "math_match":
                if str(extract_model_number(text)) == str(data["ref"]): correct += 1
            elif task_type == "classification":
                p_idx, n_idx = text.lower().find('positive'), text.lower().find('negative')
                guess = "none"
                if p_idx != -1 and (n_idx == -1 or p_idx < n_idx): guess = "positive"
                elif n_idx != -1 and (p_idx == -1 or n_idx < p_idx): guess = "negative"
                if guess == data["ref"]: correct += 1
            elif task_type == "rouge":
                rL += scorer.score(data["ref"], text)['rougeL'].fmeasure

        if task_type in ["exact_match", "classification", "math_match"]:
            results[task_name] = f"{(correct / len(items) * 100):.1f}%"
        elif task_type == "rouge":
            results[task_name] = f"R-L: {(rL / len(items)):.3f}"
            
    # Cleanup memory
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return results

def main():
    tasks = get_dataset_samples(num_samples=20)
    models = [
        ("Original", "microsoft/Phi-3.5-mini-instruct"),
        ("General_Subnet", "./phi-3.5-pruned-20_general"),
        ("Math_Subnet", "./phi-3.5-pruned-20_math"),
        ("Code_Subnet", "./phi-3.5-pruned-20_code"),
        ("Law_Subnet", "./phi-3.5-pruned-20_law")
    ]
    
    all_results = {}
    for name, path in models:
        try: all_results[name] = evaluate_model(path, tasks)
        except Exception as e: 
            print(f"Skipped {name} ({e})")
            all_results[name] = {task: "N/A" for task in tasks.keys()}
            
    print("\n\n" + "="*115)
    print(" "*35 + "FINAL SUB-NETWORK COMPARISON DASHBOARD")
    print("="*115)
    print(f"{'Task (Metric)':<28} | {'Orig':<7} | {'General':<9} | {'Math(W)':<9} | {'Code(W)':<9} | {'Law(W)':<9}")
    print("-" * 115)
    
    for task in tasks.keys():
        row = [f"{task:<28}"]
        for name, _ in models:
            val = str(all_results[name].get(task, "N/A")).replace("R-L: ", "").replace("%", "")[:7]
            pad = 7 if name == "Original" else 9
            row.append(f"{val:<{pad}}")
        print(" | ".join(row))

if __name__ == "__main__":
    main()