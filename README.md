# FM-OSD+: Foundation Model-Enabled One-Shot Detection of Anatomical Landmarks

> Extended from FM-OSD (MICCAI 2024) with **MLMF** and **TCGR** innovations targeting TMI.

---

## Introduction

We extend FM-OSD — a one-shot anatomical landmark detection framework powered by frozen DINO ViT features — with two modular innovations:

1. **MLMF (Multi-Layer Multi-Facet Adaptive Fusion):** Adaptively fuses descriptors from multiple ViT layers (5, 8, 11) and attention facets (key, value) via a lightweight channel-projection + cross-source attention mechanism, replacing the single-layer single-facet feature extraction in the original FM-OSD.

2. **TCGR (Topology-Constrained Graph Refinement):** A Graph Attention Network (GAT) that models pairwise anatomical topology constraints between landmarks and refines the initial MLMF+FM-OSD predictions in a post-processing step.

---

## Results on Cephalometric Dataset (400 images, 5-fold Cross-Validation)

| Method | MRE (mm) ↓ | SDR 2mm ↑ | SDR 2.5mm ↑ | SDR 3mm ↑ | SDR 4mm ↑ | SDR 6mm ↑ | SDR 8mm ↑ |
|--------|-----------|----------|------------|----------|----------|----------|----------|
| FM-OSD (baseline, test set) | 1.819 | 67.4% | 78.0% | 84.7% | 91.9% | 97.9% | 99.3% |
| FM-OSD + TCGR (5-fold CV) | 1.776 | 69.3% | 79.6% | 86.0% | 93.2% | 97.8% | 99.4% |
| **MLMF + FM-OSD + TCGR (5-fold CV)** | **1.753** | **69.6%** | **79.3%** | **85.9%** | **93.3%** | **98.2%** | **99.4%** |

Per-fold MRE for the full MLMF+FM-OSD+TCGR pipeline:

| Fold | MRE (mm) |
|------|----------|
| Fold 1 | 1.8360 |
| Fold 2 | 1.7171 |
| Fold 3 | 1.7624 |
| Fold 4 | 1.6878 |
| Fold 5 | 1.7615 |
| **Overall** | **1.7530** |

---

## Architecture Overview

```
Query X-ray
    │
    ▼
DINO ViT (frozen)  ─── layers {5, 8, 11} × facets {key, value}
    │
    ▼
AdaptiveFusionModule (MLMF)
  per-source 1×1 projection (6528→256) → cross-source attention → fused descriptor
    │
    ├─── Global matching (bidirectional cosine + top-k verification)
    │
    └─── Local crop + local MLMF descriptor → refined coordinates
    │
    ▼
TCGR (Graph Attention Network, 3 layers)
  node features: [y, x] coords  → topology-constrained refinement
    │
    ▼
Final landmark coordinates
```

---

## Requirements

```
python >= 3.8
torch >= 1.12
torchvision
torch_geometric        # for TCGR GAT layers
tensorboardX
batchgenerators
termcolor
tqdm
```

See `requirements.txt` for the full list.

---

## Usage

### 0. Data preparation

Download the Cephalometric dataset from https://github.com/MIRACLE-Center/Oneshot_landmark_detection and unzip into `dataset/Cephalometric/`.

Run offline augmentation on the template image:
```bash
python data_generate.py
```

### 1. Compile CUDA extension

```bash
cd <repo_root>
python setup.py install
```

### 2. Download DINO weights (optional)

The code uses `torch.hub` to load `dino_vits8`. On first run it downloads automatically. For offline use, pre-download to `~/.cache/torch/hub/facebookresearch_dino_main/` and the code will load from cache.

---

### 3. Original FM-OSD

**Train global branch:**
```bash
python train1.py
```

**Train local branch:**
```bash
python train2.py
```

**Test:**
```bash
python test.py
```

---

### 4. MLMF Extension

**Train MLMF global branch** (`Upnet_v3_MLMF`):
```bash
python train1_mlmf.py \
  --mlmf_layers 5,8,11 \
  --mlmf_facets key,value \
  --exp global_mlmf
```

**Fine-tune MLMF local branch** (`Upnet_v3_MLMF_CoarseToFine`):
```bash
python train2_mlmf.py \
  --global_ckpt models/global_mlmf/model_post_best.pth \
  --mlmf_layers 5,8,11 \
  --mlmf_facets key,value \
  --exp local_mlmf
```

---

### 5. TCGR Extension

**Step 1 — Cache MLMF+FM-OSD predictions for all 400 images:**
```bash
python precompute_mlmf_cache.py \
  --ckpt models/local_mlmf/model_post_fine_iter_20_1.8327.pth \
  --cache_dir data/tcgr_cache_mlmf
```

**Step 2 — Train TCGR with 5-fold cross-validation:**
```bash
python train_tcgr_cv.py \
  --cache_dir data/tcgr_cache_mlmf \
  --save_dir output \
  --exp tcgr_cv_mlmf \
  --max_iterations 3000 \
  --lr 1e-4
```

---

## Module Details

### MLMF — Multi-Layer Multi-Facet Adaptive Fusion (`post_net.py`)

- **`AdaptiveFusionModule`**: Takes `num_sources` feature maps (each `[B, 6528, Hp, Wp]`), projects each to 256 channels via a 1×1 conv at patch resolution, computes global-pooled cross-source attention weights, and returns a weighted sum `[B, 256, Hp, Wp]`.
- **`Upnet_v3_MLMF`**: Replaces single-facet descriptor with `AdaptiveFusionModule` output; keeps the same upsampling + heatmap head.
- **`Upnet_v3_MLMF_CoarseToFine`**: Adds a separate local-branch `AdaptiveFusionModule` (`fusion_local`) and local head (`conv_out2`) alongside the global branch.
- **Feature extractor** (`extractor_gpu.py`): `extract_multi_layer_multi_facet_descriptors` concatenates descriptors from all (layer, facet) pairs into a single tensor before passing to `post_net`.

### TCGR — Topology-Constrained Graph Refinement (`landmark_graph.py`)

- **`TCGRModule`**: A 3-layer Graph Attention Network (GAT). Each landmark is a node; edges are fully connected (anatomical topology is implicitly learned). Input: normalized `[y, x]` coordinates + dummy confidence scores and feature embeddings. Output: refined coordinates.
- **`TCGRLoss`**: Combines coordinate L1 loss with a topology-consistency regularizer (pairwise distance ordering loss).
- **Training**: 5-fold cross-validation on all 400 labeled images using cached MLMF+FM-OSD predictions as input. TCGR is lightweight (<<1M params) and trains in ~5 minutes per fold.

---

## File Structure

```
FM-OSD/
├── extractor_gpu.py          # ViTExtractor + MLMF multi-layer extraction
├── post_net.py               # Upnet_v3, Upnet_v3_MLMF, Upnet_v3_MLMF_CoarseToFine
├── landmark_graph.py         # TCGRModule, TCGRLoss
├── train1.py / train2.py     # Original FM-OSD training
├── train1_mlmf.py            # MLMF global branch training
├── train2_mlmf.py            # MLMF local branch fine-tuning
├── train_tcgr_cv.py          # TCGR 5-fold CV training
├── precompute_mlmf_cache.py  # Cache MLMF predictions for all 400 images
├── test.py                   # Original FM-OSD test
├── datasets/                 # Dataset loaders
├── evaluation/               # MRE / SDR evaluation
└── models/
    ├── global/               # Original FM-OSD global checkpoint
    ├── local/                # Original FM-OSD local checkpoint
    ├── global_mlmf/          # MLMF global checkpoint
    ├── local_mlmf/           # MLMF local checkpoint (best: iter 20, MRE 1.8327)
    └── tcgr_cv_mlmf/         # TCGR fold checkpoints (fold1–5)
```

---

## Acknowledgement

This code is based on [dino-vit-features](https://github.com/ShirAmir/dino-vit-features) and [Oneshot_landmark_detection](https://github.com/MIRACLE-Center/Oneshot_landmark_detection). TCGR uses [PyTorch Geometric](https://pyg.org/).

## Citation

If you find this code useful, please consider citing the original FM-OSD paper:

```bibtex
@inproceedings{miao2024fm,
  title={FM-OSD: Foundation Model-Enabled One-Shot Detection of Anatomical Landmarks},
  author={Miao, Juzheng and Chen, Cheng and Zhang, Keli and Chuai, Jie and Li, Quanzheng and Heng, Pheng-Ann},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={297--307},
  year={2024},
  organization={Springer}
}
```
