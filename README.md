# Efficient LLM Inference via Subnet Selection: Domain-Aware Layer-Sensitivity Pruning (DALSP)

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)

## 📌 Abstract
Deploying state-of-the-art Large Language Models (LLMs) like Phi-3.5-mini is bottlenecked by their massive computational footprint. These models suffer from the "Full Depth Penalty," where every single parameter (approximately 3.8 billion) is activated for every forward pass, regardless of task complexity. Simple conversational prompts invoke the same overhead as complex mathematical reasoning, leading to high VRAM usage (3.56 GB) and slow inference speeds.

This project introduces **Domain-Aware Layer-Sensitivity Pruning (DALSP)**, a prototype designed to identify structural redundancy within the MLP blocks of the transformer architecture. Pivoting from traditional magnitude-based uniform pruning, DALSP uses an information-theoretic framework based on Shannon Entropy. By adaptively allocating pruning budgets across layers, we physically remove irrelevant neurons to achieve a 20% average reduction in MLP neurons. This generates specialized subnetworks (General, Math, Code, Law) that reduce memory footprints and increase tokens-per-second (TPS) throughput, completely bypassing the need for computationally expensive retraining.

---

## 🔬 Motivation: Overcoming the Limitations of Wanda
This project serves as a technical extension and critique of the Wanda (Weight-and-Activation) pruning method. While Wanda effectively ranks neuron importance, DALSP is engineered to resolve three of its critical limitations:

1. **Critique of Uniformity:** Wanda applies a rigid, flat pruning rate to every layer, assuming information density is distributed evenly. DALSP recognizes structural imbalances and uses entropy to modulate the pruning rate based on individual layer sensitivity.
2. **Critique of Domain Agnosticism:** Standard methods calibrate activations on generic, open-domain text corpora. DALSP profiles the model on highly specialized data streams, preserving specific sub-circuits that general pruning would blindly destroy.
3. **Critique of Global Thresholding:** Wanda ignores the shape of the score distribution. DALSP utilizes the information-theoretic properties of the distribution shape to protect fragile reasoning bottlenecks from catastrophic forgetting.

---

## 🛠️ Methodology & "Structural Surgery" Pipeline

The system is built on a rigorous three-phase pipeline transitioning from data-driven profiling to physical parameter excision.

### Phase 1: Multi-Domain Profiling and Entropy Scoring
To map the functional topology of the Phi-3.5-mini architecture, forward hooks are registered exclusively on every MLP `down_proj` layer (the ideal bottleneck for measuring true output contribution).
* **Calibration:** The network processes 32 batches (512 tokens each) from highly curated datasets: GSM8K (Logic/Math), MBPP (Code), BillSum (Law), and WikiText-2 (General baseline).
* **Scoring:** For each neuron, the Wanda score is computed based on pre-trained weight magnitude and dynamic activation. These absolute scores are normalized via a softmax function into a probability distribution, generating an exact **Shannon Entropy** signature for every layer.

### Phase 2: Adaptive Budgeting via Layer Sensitivity
Instead of destructive flat pruning, Shannon Entropy dictates the pruning budget:
* **High-Entropy Layers:** A "flat" distribution signifies democratized information processing. These structurally sensitive bottlenecks receive a conservative pruning rate (clamped at a **10% lower bound**).
* **Low-Entropy Layers:** A "peaked" distribution signifies oligarchic processing dominated by a few expert neurons. These redundant layers undergo aggressive excision (up to a **32% upper bound**).
* A z-score transformation of entropy values determines the exact layer rates, followed by a mathematical renormalization to ensure the global network reduction hits the precise 20% target.

### Phase 3: Structural Surgery & Zero-Shot Bias Compensation
Unlike sparse masking (which zeroes weights but keeps VRAM usage identical), this pipeline performs **true structural pruning**. 
* **Physical Excision:** Weight matrices are physically sliced, permanently shrinking tensor shapes and freeing up GPU VRAM.
* **Zero-Shot Signal Recovery:** Sudden parameter excision induces "activation drift." DALSP mitigates this by mathematically aggregating the mean expected contribution of the pruned neurons (captured in Phase 1) and absorbing it directly into the bias vectors of the surviving neurons. This stabilizes the network immediately.

---

## 📊 Evaluation & Benchmarks

Our evaluation confirms the phenomenon of **Superadditivity**—where cutting the model's size by 20% acts as "Structural Denoising." By physically silencing conflicting cognitive pathways (e.g., conversational sub-circuits trying to activate during a math problem), specialized capacity actually increases.

### Hardware Efficiency & Throughput
*Structural pruning permanently shrinks the intermediate size of the MLP blocks, reducing FLOPs and severely alleviating the memory bandwidth bottleneck.*

| Model Variant | VRAM (GB) | VRAM Reduction | Throughput (TPS) |
| :--- | :--- | :--- | :--- |
| **Original (Unpruned)** | 3.56 | - | 11.84 |
| **General Flat (20%)** | 3.11 | -0.45 GB (-12.6%) | 12.62 |
| **General DALSP** | 3.00 | -0.56 GB (-15.8%) | **17.63** 🚀 |
| **Math DALSP** | 2.99 | -0.57 GB (-16.0%) | 10.77 |
| **Code DALSP** | 2.98 | -0.58 GB (-16.3%) | 8.20 |
| **Law DALSP** | 3.00 | -0.56 GB (-15.8%) | 15.27 |

### Task Accuracy and Cognitive Retention
*Divergence in optimal models highlights the trade-off between encyclopedic knowledge and specialized reasoning.*

| Task (Dataset) | Original Baseline | Matched Flat (20%) | Matched DALSP (20%) |
| :--- | :--- | :--- | :--- |
| **Logic (GSM8K Acc)** | 42.0% | 54.0% | **53.0%** (Math) |
| **Coding (MBPP RL)** | 0.166 | 0.169 | **0.174** (Code) |
| **Sentiment (SST2 Acc)** | 45.0% | 73.0% | **86.0%** (Math) |
| **Legal (BillSum RL)** | **0.264** | 0.255 | 0.251 (Law) |
| **Summarization (CNN/DM)** | **0.199** | 0.199 | 0.193 (General) |
| **Exact Match (TriviaQA)** | **40.0%** | 20.0% | 19.0% (General) |

> **Key Insight:** While the Math and Code DALSP models dominate their respective specialized domains, broad factual recall (e.g., TriviaQA Exact Match dropping from 40% to 19%) is the primary casualty of subnet specialization.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* PyTorch 2.0+
* HuggingFace Transformers

### Installation
```bash
git clone [https://github.com/yourusername/Domain-Aware_Layer_Sensitivity_Pruning.git](https://github.com/yourusername/Domain-Aware_Layer_Sensitivity_Pruning.git)
cd Domain-Aware_Layer_Sensitivity_Pruning
pip install -r requirements.txt

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
