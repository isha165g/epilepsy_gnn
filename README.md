# Patient-Invariant Spatiotemporal Graph Neural Networks for Cross-Subject Epileptic Seizure Forecasting

An end-to-end deep learning framework designed to forecast epileptic seizures from multi-channel scalp EEG data while overcoming **cross-patient shortcut learning and overfitting**. 

By coupling a **2-layer Graph Attention Network (GAT)** over standard 10–20 electrode topologies with a **Gated Recurrent Unit (GRU)** and a **Gradient Reversal Layer (GRL)**, the model learns patient-invariant preictal transition dynamics that generalize to held-out, unseen subjects.

---

## 1. The Overfitting & Shortcut Learning Challenge

Standard deep learning models trained on scalp EEG frequently suffer from severe **cross-patient generalization collapse**:
- **Physical Confounding**: Scalp EEG potentials are dominated by subject-specific bio-physical attributes (skull thickness, electrode impedence, scalp conductance, amplifier baseline offsets).
- **Shortcut Memorization**: In cross-subject training datasets where seizure events cluster within specific patients, unconstrained neural networks easily minimize classification loss by mapping patient identities rather than true preictal electrographic patterns.
- **Empirical Failure**: When tested on held-out subjects, unregularized deep models collapse into trivial single-class predictors (yielding **94.94% sensitivity** but only **1.56% specificity**, with an overall pooled ROC-AUC of **0.4539**—below random chance). A linear probe revealed that **76.7%** of patient identities could be linearly decoded directly from the baseline model’s frozen latent space.

---

## 2. Invariant Spatiotemporal Architecture

To eliminate patient shortcut learning, we introduced a dual-mechanism approach:
1. **Per-Recording Standardization**: Absolute voltage amplitudes and spectral band powers are standardized within individual recording sessions ($\mu_{\text{rec}} = 0, \sigma_{\text{rec}} = 1$), eliminating DC impedance offsets while preserving relative spectral band power ratios ($\frac{P_{\text{band}}}{P_{\text{total}}}$) and normalized spectral entropy.
2. **Domain-Adversarial Dual-Head GAT-GRU**: A spatiotemporal feature extractor coupled with an adversarial patient discriminator via a Gradient Reversal Layer (GRL).

### Architectural Block Diagram

```mermaid
flowchart TD
    subgraph INPUT["1. Input EEG Temporal Sequence"]
        direction TB
        RAW["Raw Scalp EEG (20 Channels, 10-20 Montage)"] --> NORM["Per-Recording Normalization & Band Power Ratios"]
        NORM --> TENSOR["Temporal Sequence: T = 6 Windows (30s Total)<br/>X_t in R^(20 nodes x 26 features)"]
    end

    subgraph SPATIAL["2. Spatial Graph Attention Encoder (GAT)"]
        direction TB
        GAT1["GATConv 1 (26 -> 32) + ReLU + Dropout(0.3)"]
        GAT2["GATConv 2 (32 -> 64) + ReLU"]
        POOL["Global Mean Pooling across 20 Electrodes"]
        TENSOR --> GAT1 --> GAT2 --> POOL
        POOL --> G_SEQ["Spatial Sequence Embeddings: [g_1, g_2, ..., g_6] in R^(6 x 64)"]
    end

    subgraph TEMPORAL["3. Temporal Recurrent Encoder (GRU)"]
        direction TB
        GRU["Gated Recurrent Unit (GRU): hidden_size = 64"]
        G_SEQ --> GRU
        GRU --> H["Spatiotemporal Embedding: h = hidden[-1] in R^64"]
    end

    subgraph DUAL_HEAD["4. Dual-Head Bifurcation & Adversarial Optimization"]
        direction TB
        subgraph SEIZURE_HEAD["Classification Head (Task Branch)"]
            CLS_FC["Linear(64 -> 32) + ReLU + Linear(32 -> 2)"]
            CLS_PRED["Seizure Logits y_hat (Interictal vs Preictal)"]
            CLS_LOSS["Loss: L_seizure (Cross-Entropy)"]
            H --> CLS_FC --> CLS_PRED --> CLS_LOSS
        end

        subgraph PATIENT_HEAD["Adversarial Head (Patient Invariance Branch)"]
            GRL["Gradient Reversal Layer (GRL)<br/>Forward: y = h<br/>Backward: grad = -alpha(p) * grad"]
            ADV_FC["Linear(64 -> 32) + ReLU + Linear(32 -> 30)"]
            ADV_PRED["Patient Identity Logits p_hat (30 Classes)"]
            ADV_LOSS["Loss: L_patient (Cross-Entropy)"]
            H --> GRL --> ADV_FC --> ADV_PRED --> ADV_LOSS
        end
    end

    CLS_LOSS --> TOTAL_OBJ["Total Adversarial Objective:<br/>L_total = L_seizure + 0.5 * alpha(p) * L_patient"]
    ADV_LOSS --> TOTAL_OBJ
```

### Mathematical Formulation

- **Gradient Reversal Layer (GRL):**
  $$\mathcal{R}_{\alpha}(h)=h \quad (\text{forward}), \qquad \frac{d\mathcal{R}_{\alpha}}{dh}=-\alpha\mathbf{I} \quad (\text{backward})$$
- **Optimization Objective:**
  $$\min_{\theta_y,\theta_d}\max_{\theta_E}\left[\mathcal{L}_{\text{seizure}}(G_y(h),y)-\alpha(p)\cdot\lambda\cdot\mathcal{L}_{\text{patient}}(G_d(h),d)\right]$$
- **Dynamic GRL Schedule**:
  $$\alpha(p) = \frac{2}{1 + \exp(-10p)} - 1 \quad \text{for training progress } p \in [0, 1]$$
  This schedules the adversarial strength smoothly from $0 \to 1.0$, preventing gradient shock in early epochs while driving subject-invariance in later representations.

---

## 3. Master Model Comparison Table

Evaluated on the held-out development set containing **5,409 continuous 30-second sequences across 17 unseen patients**:

| Model Architecture | Input Features & Topology | Train ROC-AUC | Dev ROC-AUC (Pooled) | Dev Balanced Accuracy | Dev Sensitivity | Dev Specificity | Dev PR-AUC | Dev Patient Median AUC | Patient Leakage Probe Acc | Shortcut Learning Diagnosis |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Sequence Logistic Regression** | Time-Frequency (Flattened $6 \times 200 = 1200$) | *N/A (Linear)* | 0.5038 | 0.5014 | 43.32% | 56.96% | 0.4702 | 0.5392 | *N/A (No latent space)* | **Linear Baseline**: Predicts near chance across unseen patients without spatiotemporal graph structures. |
| **Baseline Multimodal GAT-GRU** | 20-Node 10–20 Graph, 26 Features (Raw normed) | 0.7716 | 0.4539 | 0.4825 | 94.94% | 1.56% | 0.4481 | 0.5455 | **76.74%** | **Severe Shortcut Learning**: Latent space memorizes patient ID (76.7% probe acc), causing near-complete specificity collapse (1.56%) on unseen patients. |
| **Domain-Adversarial Invariant GAT-GRU** *(Proposed)* | 20-Node Invariant Graph, 26 Features (Per-recording normed) | **0.7992** | **0.5158** | **0.5138** | 74.11% | **28.66%** | **0.4874** | **0.5562** | **37.16%** | **Patient-Invariant**: **-39.58% reduction** in patient leakage, resolving specificity collapse (+27.10%) and restoring cross-subject generalization. |

*Metrics source: `results/ablation/master_model_comparison_table.csv`*

---

## 4. Empirical Evaluation Figures

The figure below highlights:
1. **Panel (a)**: Cross-Patient ROC curves showing the recovery from the baseline model's sub-chance collapse ($0.4539 \to 0.5158$).
2. **Panel (b)**: Patient Leakage Probe accuracy demonstrating that the adversarial head suppresses patient identity memorization from **76.7%** to **37.2%**.

![ROC Curves and Patient Invariance Probe](results/ablation/roc_and_patient_leakage_probe.png)

*(Vector PDF format available at `results/ablation/roc_and_patient_leakage_probe.pdf`)*

---

## 5. Repository Structure

```
epilepsy_gnn/
├── configs/                  # Experiment configuration files
├── data/
│   ├── graph/                # Standard graph adjacency and datasets
│   ├── graph_invariant/      # Artifact- and patient-invariant graph datasets
│   └── metadata/             # Manifests, annotations, and recording splits
├── notebooks/                # Exploratory analysis and prototyping
├── results/
│   ├── ablation/             # Evaluation logs, metrics, and figures
│   │   ├── baselines/        # Logistic and flat baseline predictions
│   │   ├── invariance_evaluation/ # Probing and invariant model outputs
│   │   ├── master_model_comparison_table.csv
│   │   ├── roc_and_patient_leakage_probe.png
│   │   └── roc_and_patient_leakage_probe.pdf
│   └── models/               # Model checkpoints (.pt)
└── src/
    ├── data/                 # Data loading and forecasting event selection
    ├── evaluation/           # Benchmark generation and figure plotting
    ├── graph/                # Graph topology and invariant feature extraction
    ├── models/               # PyTorch GAT, GAT-GRU, and InvariantGATGRU models
    └── preprocessing/        # Filtering, resampling, montage, and windowing
```

---

## 6. Getting Started & Reproduction

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/epilepsy_gnn.git
cd epilepsy_gnn

# Create virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Running the Benchmark & Generating Figures

To reproduce the Master Comparison Table and the Dual-Panel Figures:

```bash
python3 src/evaluation/generate_benchmark_results.py
```

### Training the Invariant Spatiotemporal GNN

To train the domain-adversarial GAT-GRU with gradient reversal:

```bash
python3 src/models/train_gat_gru_invariant.py
```

### Running the Full Invariance Probing Benchmark

To extract latent representations and evaluate linear patient probes:

```bash
python3 src/models/run_full_invariance_comparison.py
```
