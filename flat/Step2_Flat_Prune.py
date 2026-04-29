import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import shutil
import os
import gc

def prune_linear_layer(layer, keep_indices, dim=0):
    with torch.no_grad():
        if dim == 0:
            layer.weight.data = layer.weight.data[keep_indices, :]
            if layer.bias is not None:
                layer.bias.data = layer.bias.data[keep_indices]
            layer.out_features = len(keep_indices)
        else:
            layer.weight.data = layer.weight.data[:, keep_indices]
            layer.in_features = len(keep_indices)
    return layer

def main():
    model_id = "microsoft/Phi-3.5-mini-instruct"
    domains = ["general", "math", "code", "law"]
    PRUNE_PERCENTAGE = 0.20
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    for domain in domains:
        profile_file = f"profile_{domain}.pt"
        if not os.path.exists(profile_file):
            print(f"Skipping {domain} (Profile file {profile_file} not found. Run Step1 first).")
            continue
            
        print(f"\n\n{'='*50}\nPruning '{domain}' Subnetwork ({PRUNE_PERCENTAGE*100}% of MLP)\n{'='*50}")
        
        # Load fresh original model
        print("Loading Original Unpruned Model...")
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True, device_map="cuda:0")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        neuron_profiles = torch.load(profile_file, map_location=device)

        print("Executing Subnetwork Surgery & Zero-Shot Bias Compensation...")
        
        keep_count = 0
        for layer_idx, layer in enumerate(model.model.layers):
            profile = neuron_profiles[layer_idx]
            scores = profile["score"].to(device)
            means = profile["mean"].to(device)
            
            sorted_indices = torch.argsort(scores, descending=True)
            total_neurons = scores.size(0)
            keep_count = int(total_neurons * (1.0 - PRUNE_PERCENTAGE))
            
            keep_indices, _ = torch.sort(sorted_indices[:keep_count])
            prune_indices, _ = torch.sort(sorted_indices[keep_count:])

            with torch.no_grad():
                down_proj = layer.mlp.down_proj
                
                pruned_weights = down_proj.weight.data[:, prune_indices].float()
                pruned_means = means[prune_indices].float()
                
                pruned_means = torch.clamp(pruned_means, min=-5, max=5)
                
                compensation_bias = torch.matmul(pruned_weights, pruned_means)
                compensation_bias = compensation_bias.to(down_proj.weight.dtype)
            
                if down_proj.bias is None:
                    down_proj.bias = torch.nn.Parameter(
                        torch.zeros(down_proj.out_features, device=device, dtype=dtype)
                    )
            
                down_proj.bias.data += compensation_bias

            keep_indices_gate_up = torch.cat([keep_indices, keep_indices + total_neurons])
            prune_linear_layer(layer.mlp.gate_up_proj, keep_indices_gate_up, dim=0)
            prune_linear_layer(layer.mlp.down_proj, keep_indices, dim=1)
            
        model.config.intermediate_size = keep_count
        
        for m in model.modules():
            if hasattr(m, "_tied_weights_keys") and isinstance(m._tied_weights_keys, list):
                m._tied_weights_keys = {k: None for k in m._tied_weights_keys}
                
        save_directory = f"./phi-3.5-pruned-20_{domain}"
        print(f"Saving '{domain}' subnetwork to {save_directory}...")
        
        if os.path.exists(save_directory):
            shutil.rmtree(save_directory)
            
        model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        
        snapshot_download(repo_id=model_id, allow_patterns=["*.py"], local_dir=save_directory)
        print(f"Finished {domain}!")
        
        # Free memory for next loop
        del model
        torch.cuda.empty_cache()
        gc.collect()

    print("\nAll 4 Subnetworks have been successfully pruned! Proceed to Step 3.")

if __name__ == "__main__":
    main()