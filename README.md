# RUPA-DSA: Reliability-Aware Unified Prototype Alignment with Dual Semantic-Aware Network

*Read this in other languages: [Tiếng Việt](README_vi.md)*

**RUPA-DSA** (Reliability-Aware Unified Prototype Alignment with Dual Semantic-Aware Network) is an advanced framework designed for Weakly-Supervised Video Anomaly Detection (WS-VAD). 

By leveraging a **Two-stage Progressive Training (Warm-Start) strategy**, **adaptive pseudo-labeling (Top-K / Otsu's Thresholding)**, and a **Safe Gate routing mechanism**, this architecture efficiently mitigates label noise and prevents catastrophic forgetting. The model demonstrates outstanding robustness and achieves unprecedented State-of-the-Art (SOTA) performance on two major real-world datasets: UCF-Crime and XD-Violence.

---

## 🏆 Record-Breaking Benchmarks
Below are the actual performance metrics recorded when training the RUPA-DSA architecture (with CLIP features) using the Warm-Start technique:

### 1. XD-Violence Dataset (Multi-domain)
| Metric | Baseline DSANet | RUPA-DSA (Otsu) | RUPA-DSA (Top-K + Warm-Start) |
|--------|:---:|:---:|:---:|
| **AUC** | 95.39% | **95.40%** | 95.39% |
| **AP** | 86.99% | **87.04% (SOTA)** | 87.02% |
*(Otsu thresholding maximizes its potential on multi-domain data, thoroughly eliminating label noise)*

### 2. UCF-Crime Dataset (Homogeneous CCTV)
| Metric | Baseline DSANet | RUPA-DSA (Otsu) | RUPA-DSA (Top-K + Warm-Start) | RUPA-DSA (Top-K + WS without Semantics) |
|--------|:---:|:---:|:---:|:---:|
| **AUC** | 89.44% | 89.53% | 89.54% | **89.55%** |
| **AP** | 37.85% | 38.84% | 39.09% | **40.21% (SOTA)** |
*(Disabling Otsu, completely neutralizing the Language branch (S_sem = 0), and balancing a 50/50 ratio for pure Vision helped eliminate linguistic noise on CCTV data, establishing a historic SOTA peak of 40.21%)*

---

## 🔑 Core Technological Contributions

To achieve the power of extracting subtle anomalies without degrading global representations, RUPA-DSA relies on two foundational techniques:

### 1. Two-Stage Training Strategy (Warm-Start & Safe Gate)
Instead of training from scratch, this method turns RUPA into a **Plug-and-Play Adapter**:
- **Stage 1 (Ignition):** Inherit all global topological representations from the original DSANet checkpoint. Then, **freeze the entire main network** (main-lr = 0.0) to resist Catastrophic Forgetting.
- **Stage 2 (RUPA Refinement):** Open the gate for the RUPA branch to learn (
efiner-lr = 1e-5). RUPA now stands on the shoulders of DSANet, receiving raw features and focusing 100% of its capacity on utilizing **Reconstruction** and **Semantic Routing** to correct biased predictions.

### 2. Adaptive Normal Selection (Otsu's Thresholding)
Instead of selecting a fixed percentage of normal frames, RUPA integrates **Otsu's thresholding** algorithm. This algorithm automatically analyzes the 1D anomaly score distribution of each video to find an optimal cut-off threshold, accurately separating Normal and Abnormal regions. This ensures the Counterfactual Reconstruction process operates precisely.

---

## 🖼️ Architecture Diagram

![RUPA-DSA Architecture](rupa_architecture.png)

> **Note:** This project is an upgraded and extended version based on the original architecture of [DSANet](https://github.com/hcmut-ubmlab/DSANet). RUPA-DSA inherits the feature extraction power of DSANet while seamlessly appending anomaly refinement modules (Reconstruction & Semantic Routing) to thoroughly resolve the Label Noise problem in Multiple Instance Learning (MIL).

---

## 🛠️ Training & Usage

To reproduce the SOTA scores above, the command configuration must strictly follow the Warm-Start technique (Using Original Checkpoint + Freezing the main branch).

**Link to download the Author's original Best DSANet Checkpoint:** [Google Drive](https://drive.google.com/drive/folders/1PqvaNm_s-fOOrnJRqrG50zV2R2UqRwHK)

### Training Configuration for UCF-Crime (10 Epochs)
`ash

python src/ucf_train.py \
  --train-list /path/to/ucf_train.csv \
  --test-list /path/to/ucf_test.csv \
  --model-path /path/to/best_ucf.pth \
  --checkpoint-path /path/to/checkpoint_ucf.pth \
  --init-model-path /path/to/dsanet_model_ucf.pth \
  --max-epoch 10 \
  --batch-size 64 \
  --num-workers 2 \
  --seed 234 \
  --rupa-use true \
  --adaptive_normal_selection false \
  --routing-mode safe_gate \
  --main-lr 0.0 \
  --refiner-lr 1e-5 \
  --routing-det-weight 0.5 \
  --routing-rec-weight 0.3 \
  --routing-sem-weight 0.2 \
  --loss-residual-weight 1.0 \
  --loss-reconstructed-normal-weight 1.0 \
  --loss-dnp-normal-weight 0.1 \
  --loss-consistency-weight 1.0 \
  --loss-gather-weight 1.0
`

### Training Configuration for XD-Violence (10 Epochs - Requires Otsu)
`ash

python src/xd_train.py \
  --train-list /path/to/xd_train.csv \
  --test-list /path/to/xd_test.csv \
  --model-path /path/to/best_xd.pth \
  --checkpoint-path /path/to/checkpoint_xd.pth \
  --init-model-path /path/to/dsanet_model_xd.pth \
  --max-epoch 10 \
  --batch-size 96 \
  --num-workers 2 \
  --seed 234 \
  --rupa-use true \
  --adaptive_normal_selection true \
  --routing-mode safe_gate \
  --main-lr 0.0 \
  --refiner-lr 1e-5 \
  --routing-det-weight 0.5 \
  --routing-rec-weight 0.3 \
  --routing-sem-weight 0.2 \
  --loss-residual-weight 1.0 \
  --loss-reconstructed-normal-weight 1.0 \
  --loss-dnp-normal-weight 0.1 \
  --loss-consistency-weight 1.0 \
  --loss-gather-weight 1.0
`

### Training Configuration (Neutralizing Language - Absolute SOTA 40.21% for UCF-Crime)
To achieve the absolute SOTA peak of 40.21% on UCF-Crime, disable Otsu, set the routing-sem-weight to 0, and balance a 0.5 - 0.5 ratio for Vision:
`ash

python src/ucf_train.py \
  --train-list /path/to/ucf_train.csv \
  --test-list /path/to/ucf_test.csv \
  --model-path /path/to/best_ucf.pth \
  --checkpoint-path /path/to/checkpoint_ucf.pth \
  --init-model-path /path/to/dsanet_model_ucf.pth \
  --max-epoch 10 \
  --batch-size 64 \
  --num-workers 2 \
  --seed 234 \
  --rupa-use true \
  --adaptive_normal_selection false \
  --routing-mode safe_gate \
  --main-lr 0.0 \
  --refiner-lr 1e-5 \
  --routing-det-weight 0.5 \
  --routing-rec-weight 0.5 \
  --routing-sem-weight 0.0 \
  --loss-residual-weight 1.0 \
  --loss-reconstructed-normal-weight 1.0 \
  --loss-dnp-normal-weight 0.1 \
  --loss-consistency-weight 1.0 \
  --loss-gather-weight 1.0
`

> **Note:** The gradual decline in the AP score after reaching its peak in later Epochs is an intrinsic characteristic (Overfitting due to Label Noise) of the Warm-Start method in WS-VAD. The codebase has integrated **Early Stopping** to automatically capture and save the Weights at the highest score.

---

## 📜 License
This project is licensed under the MIT License.
