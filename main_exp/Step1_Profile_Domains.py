"""
Step 1: Multi-Domain Profiling with Per-Layer Entropy Computation
=================================================================
Novel contribution: Beyond saving Wanda scores, we compute the Shannon entropy
of each layer's normalized score distribution. This entropy is the signal used
in Step 2 to allocate non-uniform pruning budgets per layer (DALSP).

Low entropy  → scores are concentrated on a few neurons → layer is "prunable"
High entropy → scores are uniform, all neurons matter   → layer is "sensitive"

After all domains are profiled, we also compute the cross-domain Jaccard
similarity of top-k neuron sets, providing mechanistic evidence that different
domains occupy structurally distinct MLP subnetworks.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os
import itertools


# ── helpers ──────────────────────────────────────────────────────────────────

def compute_score_entropy(scores: torch.Tensor) -> float:
    """
    Compute Shannon entropy of the Wanda score distribution for one layer.
    Scores are treated as an un-normalized probability distribution via softmax.
    Returns entropy in nats. Higher entropy = more uniform (sensitive layer).
    """
    probs = F.softmax(scores.float(), dim=0)
    # clamp for numerical safety before log
    entropy = -(probs * probs.clamp(min=1e-12).log()).sum().item()
    return entropy


def compute_jaccard(set_a: set, set_b: set) -> float:
    if len(set_a | set_b) == 0:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    model_id = "microsoft/Phi-3.5-mini-instruct"
    print(f"Loading {model_id} for Multi-Domain Profiling (DALSP)...")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, trust_remote_code=True, device_map="cuda:0"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # ── domain definitions ────────────────────────────────────────────────────
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

    NUM_BATCHES = 32
    SEQ_LEN     = 512
    # Keep-fraction used for cross-domain Jaccard (top-80% = bottom-20% pruned)
    JACCARD_KEEP_FRACTION = 0.80

    num_layers = len(model.model.layers)
    model.eval()

    # ── per-domain profiling loop ─────────────────────────────────────────────
    for domain in domains:
        print(f"\n\n{'='*60}\nProfiling Domain: {domain['name'].upper()}\n{'='*60}")

        activation_stats = {
            i: {"mean": 0.0, "norm": 0.0, "count": 0}
            for i in range(num_layers)
        }

        # Register forward hooks on each MLP down_proj
        hooks = []

        def get_hook(layer_idx):
            def hook(module, input, output):
                x = input[0].detach()
                batch_seq_len = x.shape[0] * x.shape[1]
                batch_mean    = x.mean(dim=(0, 1)).to(torch.float32)
                batch_norm    = torch.sqrt((x ** 2).mean(dim=(0, 1))).to(torch.float32)

                stats = activation_stats[layer_idx]
                if stats["count"] == 0:
                    stats["mean"] = batch_mean
                    stats["norm"] = batch_norm
                else:
                    w = batch_seq_len / (stats["count"] + batch_seq_len)
                    stats["mean"] = (1 - w) * stats["mean"] + w * batch_mean
                    stats["norm"] = (1 - w) * stats["norm"] + w * batch_norm
                stats["count"] += batch_seq_len
            return hook

        for layer_idx, layer in enumerate(model.model.layers):
            hooks.append(
                layer.mlp.down_proj.register_forward_hook(get_hook(layer_idx))
            )

        # Stream calibration data
        print(f"  Streaming '{domain['ds_path']}' ...")
        dataset = load_dataset(
            domain["ds_path"], domain.get("ds_name"),
            split=domain["split"], streaming=True
        )

        with torch.no_grad():
            batch_text        = ""
            batches_processed = 0
            pbar = tqdm(total=NUM_BATCHES, desc=f"  Calibrating [{domain['name']}]")

            for row in dataset:
                try:
                    chunk = domain["format_fn"](row)
                    if not chunk:
                        continue
                    batch_text += chunk + "\n\n"
                except Exception:
                    continue

                tokens = tokenizer(batch_text, return_tensors="pt")
                if tokens.input_ids.shape[1] >= SEQ_LEN:
                    model(tokens.input_ids[:, :SEQ_LEN].to(device))
                    batch_text         = ""
                    batches_processed += 1
                    pbar.update(1)
                    if batches_processed >= NUM_BATCHES:
                        break

            pbar.close()

        for h in hooks:
            h.remove()

        # ── Compute Wanda scores + per-layer entropy ──────────────────────────
        print(f"  Computing Wanda scores + layer entropy for '{domain['name']}' ...")
        profiles = {}
        entropies = []

        for layer_idx, layer in enumerate(model.model.layers):
            act_norm   = activation_stats[layer_idx]["norm"].to(device)
            act_mean   = activation_stats[layer_idx]["mean"].to(device)
            weight_norm = layer.mlp.down_proj.weight.data.norm(p=2, dim=0)

            wanda_score = weight_norm * act_norm

            # Novel: compute Shannon entropy of this layer's score distribution
            layer_entropy = compute_score_entropy(wanda_score)
            entropies.append(layer_entropy)

            profiles[layer_idx] = {
                "score":   wanda_score.cpu(),
                "mean":    act_mean.cpu(),
                "entropy": layer_entropy,        # ← new field used by Step 2
            }

        # Print per-layer entropy summary (first 8 layers for brevity)
        print(f"\n  Layer entropy summary (nats)  [high=sensitive, low=prunable]:")
        for i in range(min(8, num_layers)):
            bar = "█" * int(30 * entropies[i] / max(entropies))
            print(f"    Layer {i:02d}: {entropies[i]:.4f}  {bar}")
        print(f"    ...  (mean={sum(entropies)/len(entropies):.4f},"
              f" min={min(entropies):.4f}, max={max(entropies):.4f})")

        output_file = f"profile_{domain['name']}.pt"
        torch.save(profiles, output_file)
        print(f"\n  ✔ Saved '{output_file}' (includes per-layer entropy)")

    # ── Cross-domain neuron overlap analysis ──────────────────────────────────
    print(f"\n\n{'='*60}")
    print("Cross-Domain Neuron Overlap Analysis (Jaccard Similarity)")
    print(f"{'='*60}")
    print("Measuring structural separability of domain subnetworks.")
    print("Low overlap → domains use distinct MLP circuits (desired).\n")

    domain_names  = [d["name"] for d in domains]
    domain_top_k  = {}   # domain → list of frozensets (one per layer)

    for name in domain_names:
        pfile = f"profile_{name}.pt"
        if not os.path.exists(pfile):
            print(f"  Warning: {pfile} not found, skipping.")
            continue
        profiles = torch.load(pfile, map_location="cpu")
        top_k_per_layer = []
        for layer_idx in range(num_layers):
            scores = profiles[layer_idx]["score"]
            k      = int(len(scores) * JACCARD_KEEP_FRACTION)
            _, top_idx = torch.topk(scores, k)
            top_k_per_layer.append(frozenset(top_idx.tolist()))
        domain_top_k[name] = top_k_per_layer

    # Pairwise Jaccard, averaged across layers
    pairs = list(itertools.combinations(domain_names, 2))
    print(f"  {'Domain Pair':<22} | {'Avg Jaccard':>11} | {'Overlap verdict'}")
    print(f"  {'-'*22}-+-{'-'*11}-+-{'-'*25}")

    jaccard_matrix = {}
    for d1, d2 in pairs:
        if d1 not in domain_top_k or d2 not in domain_top_k:
            continue
        jacc_per_layer = [
            compute_jaccard(domain_top_k[d1][i], domain_top_k[d2][i])
            for i in range(num_layers)
        ]
        avg_jacc = sum(jacc_per_layer) / len(jacc_per_layer)
        jaccard_matrix[(d1, d2)] = avg_jacc

        if avg_jacc < 0.50:
            verdict = "✔ Highly distinct subnetworks"
        elif avg_jacc < 0.70:
            verdict = "~ Moderate overlap"
        else:
            verdict = "✗ High overlap (expected for general)"

        print(f"  {d1+' ↔ '+d2:<22} | {avg_jacc:>11.4f} | {verdict}")

    print(f"\n  Interpretation: Jaccard < 0.5 between specialised domains")
    print(f"  (math, code, law) supports the DALSP hypothesis that domain")
    print(f"  knowledge is structurally localised in distinct MLP circuits.")

    print("\n\nAll 4 domain profiles generated. Proceed to Step 2 (DALSP pruning).")


if __name__ == "__main__":
    main()
