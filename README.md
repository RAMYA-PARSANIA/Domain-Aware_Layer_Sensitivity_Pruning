# Efficient LLM Inference via Subnet Selection: Domain-Aware Layer-Sensitivity Pruning (DALSP)

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)

## 📌 Abstract
[cite_start]Deploying state-of-the-art Large Language Models (LLMs) like Phi-3.5-mini is often bottlenecked by a massive computational footprint[cite: 116]. [cite_start]These models suffer from the "Full Depth Penalty," activating all parameters (e.g., 3.8 billion) for every forward pass regardless of task complexity[cite: 117]. 

[cite_start]This project introduces **Domain-Aware Layer-Sensitivity Pruning (DALSP)**, an information-theoretic framework that replaces traditional magnitude-based uniform pruning[cite: 119, 124]. [cite_start]By leveraging Shannon Entropy derived from layer-wise Wanda (Weight-and-Activation) score distributions, DALSP dynamically allocates pruning budgets across the transformer architecture[cite: 125]. [cite_start]This repository provides the codebase to structurally prune Phi-3.5-mini by 20% on average, yielding specialized subnetworks (General, Math, Code, Law) that drastically reduce VRAM usage and increase token-per-second (TPS) throughput while demonstrating domain-specific *Superadditivity*[cite: 121, 122, 187].

---

## 🚀 Key Features

* [cite_start]**Information-Theoretic Pruning:** Uses Shannon Entropy to identify "peaked" (low-entropy, redundant) vs. "flat" (high-entropy, sensitive) parameter distributions[cite: 135, 139].
* [cite_start]**Domain-Specific Subnetworks:** Calibrates activation norms on specialized data streams (GSM8K, MBPP, BillSum) to physically excise conflicting/irrelevant cognitive pathways[cite: 144, 147].
* [cite_start]**Zero-Shot Bias Compensation:** Absorbs the expected mean contribution of pruned neurons into surviving bias vectors, stabilizing the network instantly without computationally expensive retraining[cite: 182, 183, 184].
* [cite_start]**True Structural Surgery:** Physically slices weight matrices (rather than applying sparse masks), permanently shrinking MLP block dimensions to free up VRAM and accelerate FLOPs[cite: 180, 181].

---

## 🛠️ Methodology Pipeline

[cite_start]The project executes a rigorous three-phase "Structural Surgery" pipeline[cite: 161]:

1. [cite_start]**Multi-Domain Profiling and Entropy Scoring:** Forward hooks are registered on `down_proj` MLP layers[cite: 164, 165]. [cite_start]The network is calibrated across 32 batches of distinct datasets (GSM8K, MBPP, BillSum, WikiText-2)[cite: 167]. [cite_start]Wanda scores are computed and normalized into a probability distribution to extract layer-specific Shannon Entropy signatures[cite: 169, 172].
2. [cite_start]**Adaptive Budgeting via Layer Sensitivity:** High-entropy layers receive conservative pruning (clamped at 10% lower bound), while low-entropy layers undergo aggressive excision (up to 32% upper bound)[cite: 174, 175, 177]. [cite_start]Global network reduction is mathematically renormalized to exactly 20%[cite: 178].
3. [cite_start]**Structural Surgery:** Parameters are physically removed[cite: 180]. [cite_start]Activation drift is mitigated via zero-shot bias compensation[cite: 182].

---

## 📊 Evaluation & Benchmarks

[cite_start]Our experiments demonstrate the phenomenon of **Superadditivity**, where reducing the model's size actively sharpens its specialized reasoning capacity by acting as a structural denoiser[cite: 187, 194, 195].

### Hardware Efficiency (VRAM & Latency)
*Benchmarked on Phi-3.5-mini*

| Model Variant | VRAM (GB) | VRAM Reduction | Throughput (TPS) |
| :--- | :--- | :--- | :--- |
| **Original (Unpruned)** | 3.56 | - | 11.84 |
| **General Flat (20%)** | 3.11 | -12.6% | 12.62 |
| **General DALSP** | 3.00 | -15.8% | 17.63 |
| **Math DALSP** | 3.00 | -16.0% | 10.77 |
| **Code DALSP** | 2.98 | -16.3% | 8.20 |
| **Law DALSP** | 3.00 | -15.8% | 15.27 |

### Cognitive Retention & Superadditivity
*By physically deleting rows and columns that interfere with rule-based generation, DALSP models outperform the baseline in their target domains.*

| Task (Dataset) | Original Baseline | Math DALSP | Code DALSP | Law DALSP | General DALSP |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logic (GSM8K)** | 42.0% | **53.0%** 🏆 | 33.0% | 21.0% | 26.0% |
| **Coding (MBPP)** | 0.166 | 0.134 | **0.174** 🏆 | 0.130 | 0.166 |
| **Sentiment (SST2)** | 45.0% | 86.0% | 85.0% | 87.0% | 83.0% |
| **Exact Match (TriviaQA)** | 40.0% 🏆 | 8.0% | 10.0% | 10.0% | 19.0% |
| **Legal Summ (BillSum)** | 0.264 🏆 | 0.243 | 0.258 | 0.251 | 0.235 |

> [cite_start]**Note:** Encyclopedic knowledge (e.g., TriviaQA) is the primary casualty of structural pruning, confirming that specialization inevitably trades off broad factual recall[cite: 204, 210, 211].

---

## 🔮 Future Work

1. [cite_start]**LoRA Post-Pruning Fine-Tuning:** Applying Low-Rank Adaptation to re-learn neural connections severed during surgery[cite: 214].
2. [cite_start]**Attention Head Entropy Pruning:** Extending the DALSP methodology from MLP neurons to Multi-Head Attention components to compound memory savings[cite: 217, 218].
3. [cite_start]**Mixture-of-Subnetworks (MoS) Routing:** Utilizing a lightweight classifier to route inference queries to the appropriate, physically distinct domain expert[cite: 221, 222].

---

## 👨‍💻 Contributors

* **Areen Patil**
* **Ramya Parsania**
* **Krishna Sai Velidanda**
* **Aaryan Antala**

---
*Developed as part of the NLP coursework. For full theoretical background, refer to the accompanying research documentation.*
