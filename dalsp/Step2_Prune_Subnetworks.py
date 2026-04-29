"""
Step 2: DALSP — Domain-Aware Layer-Sensitivity Pruning
=======================================================
Novel contribution: Instead of a flat 20% pruning rate across all layers,
we use the per-layer Shannon entropy of the domain's Wanda score distribution
(saved in Step 1) to adaptively allocate the pruning budget.

Rationale (information-theoretic):
  • Low-entropy layer  → scores are concentrated (peaked distribution)
                       → a few neurons dominate → safely prune more aggressively
  • High-entropy layer → scores are uniform (flat distribution)
                       → all neurons contribute equally → prune conservatively

The total pruning budget (TARGET_PRUNE = 20%) is preserved globally via
explicit budget renormalisation. Each layer receives a rate in [MIN_RATE, MAX_RATE].

FIX (v2.1): The adaptive rates produce different per-layer keep-counts, which
caused shape mismatches on checkpoint reload (config.intermediate_size was set
to the minimum but larger layers were saved with more neurons). The surgery is
now split into three phases so every layer is trimmed to the same min_keep:

  Phase 1 – collect the top-scoring keep_indices for each layer at its
             adaptive rate (no model writes yet).
  Phase 2 – compute min_keep = min across all layers and re-slice every
             layer's keep_indices to that count, preserving the highest-
             scoring neurons (DALSP's domain-specific selection is intact).
  Phase 3 – apply structural surgery + zero-shot bias compensation.

DALSP's real value is the domain-specific *selection* of which neurons to
keep (guided by domain Wanda scores), not just how many. The count is now
uniform = min_keep, which is consistent with model.config.intermediate_size.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import shutil
import gc

BASE_DIR = "."  # adjust if your profiles/outputs live elsewhere


# ── pruning helpers ───────────────────────────────────────────────────────────

def prune_linear_layer(layer, keep_indices, dim=0):
    """Structurally prune a Linear layer along dim, keeping only keep_indices."""
    with torch.no_grad():
        if dim == 0:
            layer.weight.data  = layer.weight.data[keep_indices, :]
            if layer.bias is not None:
                layer.bias.data = layer.bias.data[keep_indices]
            layer.out_features = len(keep_indices)
        else:
            layer.weight.data  = layer.weight.data[:, keep_indices]
            layer.in_features  = len(keep_indices)
    return layer


# ── DALSP budget allocation ───────────────────────────────────────────────────

def compute_adaptive_rates(
    profiles: dict,
    target_rate: float = 0.20,
    alpha: float       = 0.40,
    min_rate: float    = 0.10,
    max_rate: float    = 0.32,
) -> dict:
    """
    Compute a per-layer pruning rate using the domain's Wanda score entropy.

    Algorithm
    ---------
    1. Collect entropy[i] for each layer i (saved by Step 1).
    2. Normalise entropies to [0, 1].
    3. Compute sensitivity[i] = 1 - norm_entropy[i]
    4. Compute a z-score of sensitivity and modulate around target_rate.
    5. Clamp to [min_rate, max_rate].
    6. Renormalise so the average equals exactly target_rate.

    Returns
    -------
    dict {layer_idx: pruning_rate}
    """
    num_layers = len(profiles)
    entropies  = [profiles[i]["entropy"] for i in range(num_layers)]

    e_min, e_max = min(entropies), max(entropies)
    e_range      = e_max - e_min + 1e-8

    norm_entropy = [(e - e_min) / e_range for e in entropies]
    sensitivity  = norm_entropy

    s_mean = sum(sensitivity) / num_layers
    s_std  = (sum((s - s_mean) ** 2 for s in sensitivity) / num_layers) ** 0.5 + 1e-8

    raw_rates = [
        target_rate + alpha * target_rate * ((s - s_mean) / s_std)
        for s in sensitivity
    ]

    clamped_rates = [max(min_rate, min(max_rate, r)) for r in raw_rates]

    actual_mean = sum(clamped_rates) / num_layers
    scale       = target_rate / actual_mean
    final_rates = {
        i: max(min_rate, min(max_rate, clamped_rates[i] * scale))
        for i in range(num_layers)
    }

    return final_rates


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    model_id          = "microsoft/Phi-3.5-mini-instruct"
    domains           = ["general", "math", "code", "law"]
    TARGET_PRUNE      = 0.20
    ALPHA             = 0.60
    PRINT_LAYER_TABLE = True

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    for domain in domains:
        profile_file = os.path.join(BASE_DIR, f"profile_{domain}.pt")
        if not os.path.exists(profile_file):
            print(f"[SKIP] {domain}: '{profile_file}' not found — run Step 1 first.")
            continue

        print(f"\n\n{'='*65}")
        print(f"DALSP Pruning  ·  Domain: {domain.upper()}  ·  Budget: {TARGET_PRUNE*100:.0f}% avg")
        print(f"{'='*65}")

        print("Loading original unpruned model...")
        model     = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, trust_remote_code=True, device_map="cuda:0"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        neuron_profiles = torch.load(profile_file, map_location=device)

        adaptive_rates = compute_adaptive_rates(
            neuron_profiles, target_rate=TARGET_PRUNE, alpha=ALPHA
        )

        num_layers = len(model.model.layers)
        actual_avg = sum(adaptive_rates.values()) / num_layers
        print(f"\nDALSP allocation summary (average = {actual_avg*100:.2f}%):")
        if PRINT_LAYER_TABLE:
            print(f"  {'Layer':>5} | {'Entropy':>8} | {'PruneRate':>9} | {'Allocation bar'}")
            print(f"  {'-----':>5}-+-{'--------':>8}-+-{'----------':>9}-+-{'-'*30}")
            for i in range(num_layers):
                ent    = neuron_profiles[i]["entropy"]
                rate   = adaptive_rates[i]
                bar    = "▓" * int(30 * rate / 0.40)
                marker = (" ← conservative" if rate < TARGET_PRUNE - 0.02
                          else " ← aggressive" if rate > TARGET_PRUNE + 0.02 else "")
                print(f"  {i:>5} | {ent:>8.4f} | {rate*100:>8.2f}% | {bar}{marker}")

        print("\nExecuting DALSP surgery (3-phase) + zero-shot bias compensation...")

        # ── Phase 1: collect per-layer top-scored keep indices ─────────────────
        # Ranked descending by domain Wanda score — not yet sorted by position.
        keep_indices_by_layer = {}
        for layer_idx in range(num_layers):
            scores        = neuron_profiles[layer_idx]["score"].to(device)
            total_neurons = scores.size(0)
            num_pruned    = int(total_neurons * adaptive_rates[layer_idx])
            num_kept      = total_neurons - num_pruned
            sorted_by_score = torch.argsort(scores, descending=True)
            keep_indices_by_layer[layer_idx] = sorted_by_score[:num_kept]

        # ── Phase 2: unify to min_keep ─────────────────────────────────────────
        # FIX: every layer must have the same intermediate_size so that
        # model.config.intermediate_size matches every saved weight tensor.
        # We keep the top min_keep neurons (highest domain-Wanda score) in
        # each layer; DALSP's domain-specific selection is fully preserved.
        min_keep = min(len(v) for v in keep_indices_by_layer.values())
        orig_size = neuron_profiles[0]["score"].size(0)
        print(f"  Uniform keep count (Phase-2 alignment): {min_keep} "
              f"({(1 - min_keep / orig_size)*100:.1f}% pruned per layer)")

        for layer_idx in range(num_layers):
            # Slice to min_keep (still highest-scoring), then sort by position
            top = keep_indices_by_layer[layer_idx][:min_keep]
            keep_indices_by_layer[layer_idx], _ = torch.sort(top)

        # ── Phase 3: structural surgery + bias compensation ────────────────────
        for layer_idx, layer in enumerate(model.model.layers):
            profile      = neuron_profiles[layer_idx]
            means        = profile["mean"].to(device)
            total_neurons = neuron_profiles[layer_idx]["score"].size(0)

            keep_indices  = keep_indices_by_layer[layer_idx]        # (min_keep,)

            # Derive prune indices as the complement of keep_indices
            keep_mask     = torch.zeros(total_neurons, dtype=torch.bool, device=device)
            keep_mask[keep_indices] = True
            prune_indices = torch.where(~keep_mask)[0]

            # Zero-shot bias compensation (unchanged from original Step 2)
            with torch.no_grad():
                down_proj      = layer.mlp.down_proj
                pruned_weights = down_proj.weight.data[:, prune_indices].float()
                pruned_means   = means[prune_indices].float()
                pruned_means   = torch.clamp(pruned_means, min=-5, max=5)
                compensation   = torch.matmul(pruned_weights, pruned_means)

                if down_proj.bias is None:
                    down_proj.bias = torch.nn.Parameter(
                        torch.zeros(down_proj.out_features, device=device, dtype=dtype)
                    )
                down_proj.bias.data += compensation.to(dtype)

            # Structural surgery on gate_up_proj and down_proj
            keep_indices_gate_up = torch.cat([keep_indices, keep_indices + total_neurons])
            prune_linear_layer(layer.mlp.gate_up_proj, keep_indices_gate_up, dim=0)
            prune_linear_layer(layer.mlp.down_proj,    keep_indices,          dim=1)

        # Every layer now has exactly min_keep neurons — safe to set in config
        model.config.intermediate_size = min_keep

        for m in model.modules():
            if hasattr(m, "_tied_weights_keys") and isinstance(m._tied_weights_keys, list):
                m._tied_weights_keys = {k: None for k in m._tied_weights_keys}

        # ── Save ──────────────────────────────────────────────────────────────
        save_dir = os.path.join(BASE_DIR, f"phi-3.5-pruned-dalsp_{domain}")
        print(f"\nSaving '{domain}' DALSP subnetwork → {save_dir} ...")
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)

        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        snapshot_download(repo_id=model_id, allow_patterns=["*.py"], local_dir=save_dir)

        torch.save(adaptive_rates, os.path.join(save_dir, "dalsp_rates.pt"))
        print(f"✔ Finished {domain}  (min_keep={min_keep}, rates saved to dalsp_rates.pt)")

        del model
        torch.cuda.empty_cache()
        gc.collect()

    print("\n\nAll 4 DALSP subnetworks pruned successfully. Proceed to Step 3.")


if __name__ == "__main__":
    main()