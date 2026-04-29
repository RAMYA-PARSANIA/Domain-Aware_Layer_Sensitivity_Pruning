import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os

def main():
    model_id = "microsoft/Phi-3.5-mini-instruct"
    print(f"Loading {model_id} for Multi-Domain Profiling...")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True, device_map="cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    # We will profile 4 distinct domains: General, Math, Code, Law
    domains = [
        {
            "name": "general",
            "ds_path": "wikitext", "ds_name": "wikitext-2-raw-v1", "split": "train",
            "format_fn": lambda row: row["text"] if row["text"].strip() else ""
        },
        {
            "name": "math",
            "ds_path": "gsm8k", "ds_name": "main", "split": "train",
            "format_fn": lambda row: f"Question: {row['question']}\nAnswer: {row['answer']}"
        },
        {
            "name": "code",
            "ds_path": "mbpp", "ds_name": "full", "split": "train",
            "format_fn": lambda row: f"Problem: {row['text']}\nCode: {row['code']}"
        },
        {
            "name": "law",
            "ds_path": "billsum", "ds_name": "default", "split": "train",
            "format_fn": lambda row: f"Bill Text: {row['text'][:1000]}\nSummary: {row['summary']}"
        }
    ]

    num_batches = 32
    seq_len = 512
    model.eval()

    for domain in domains:
        print(f"\n\n{'='*50}\nProfiling Domain: {domain['name'].upper()}\n{'='*50}")
        
        # Reset activation stats for each domain
        activation_stats = {i: {"mean": 0.0, "norm": 0.0, "count": 0} for i in range(len(model.model.layers))}
        
        # Register hooks
        hooks = []
        def get_hook(layer_idx):
            def hook(module, input, output):
                x = input[0].detach()
                batch_seq_len = x.shape[0] * x.shape[1]
                batch_mean = x.mean(dim=(0, 1)).to(torch.float32)
                batch_norm = torch.sqrt((x ** 2).mean(dim=(0, 1))).to(torch.float32)
                
                curr_count = activation_stats[layer_idx]["count"]
                if curr_count == 0:
                    activation_stats[layer_idx]["mean"] = batch_mean
                    activation_stats[layer_idx]["norm"] = batch_norm
                else:
                    w = batch_seq_len / (curr_count + batch_seq_len)
                    activation_stats[layer_idx]["mean"] = (1 - w) * activation_stats[layer_idx]["mean"] + w * batch_mean
                    activation_stats[layer_idx]["norm"] = (1 - w) * activation_stats[layer_idx]["norm"] + w * batch_norm
                activation_stats[layer_idx]["count"] += batch_seq_len
            return hook

        for layer_idx, layer in enumerate(model.model.layers):
            hooks.append(layer.mlp.down_proj.register_forward_hook(get_hook(layer_idx)))

        # Stream dataset and calibrate
        print(f"Streaming {domain['ds_path']} dataset...")
        dataset = load_dataset(domain["ds_path"], domain.get("ds_name"), split=domain["split"], streaming=True)
        
        with torch.no_grad():
            batch_text = ""
            batches_processed = 0
            pbar = tqdm(total=num_batches, desc=f"Calibrating {domain['name']}")
            
            for row in dataset:
                try:
                    text_chunk = domain["format_fn"](row)
                    if not text_chunk: continue
                    batch_text += text_chunk + "\n\n"
                except Exception: continue
                
                tokens = tokenizer(batch_text, return_tensors="pt")
                if tokens.input_ids.shape[1] >= seq_len:
                    model(tokens.input_ids[:, :seq_len].to(device))
                    batch_text = ""
                    batches_processed += 1
                    pbar.update(1)
                    if batches_processed >= num_batches: break

        # Remove hooks after domain completion
        for h in hooks: h.remove()
            
        print(f"Calculating Wanda scores for {domain['name']}...")
        profiles = {}
        for layer_idx, layer in enumerate(model.model.layers):
            act_norm = activation_stats[layer_idx]["norm"].to(device)
            act_mean = activation_stats[layer_idx]["mean"].to(device)
            weight_norm = layer.mlp.down_proj.weight.data.norm(p=2, dim=0)
            
            wanda_score = weight_norm * act_norm
            profiles[layer_idx] = {"score": wanda_score.cpu(), "mean": act_mean.cpu()}
            
        output_file = f"profile_{domain['name']}.pt"
        torch.save(profiles, output_file)
        print(f"Saved '{output_file}'!")

    print("\nAll 4 Domain Profiles have been generated successfully! Proceed to Step 2.")

if __name__ == "__main__":
    main()