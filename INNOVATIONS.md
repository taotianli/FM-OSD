# FM-OSD Innovations for TMI Submission

This document describes the three innovative modules added to FM-OSD for enhanced anatomical landmark detection.

## Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        FM-OSD Enhanced Pipeline                             │
│                                                                              │
│  Template Image ─┬─► DINO ViT (frozen) ─┬─► [MLMF] Multi-Layer Fusion      │
│  Query Image ────┘                      │                                   │
│                                         ▼                                   │
│                           ┌─────────────────────────────┐                   │
│                           │   AdaptiveFusionModule      │                   │
│                           │   (Learned source weights)   │                   │
│                           └─────────────────────────────┘                   │
│                                         │                                   │
│                                         ▼                                   │
│                           ┌─────────────────────────────┐                   │
│                           │   Initial Predictions       │                   │
│                           │   (Cosine Similarity Match) │                   │
│                           └─────────────────────────────┘                   │
│                                         │                                   │
│            ┌────────────────────────────┼────────────────────────────┐     │
│            ▼                            ▼                            ▼     │
│    ┌──────────────┐          ┌──────────────────┐          ┌────────────┐ │
│    │ [TCGR]       │          │ [TEU]            │          │ Standard   │ │
│    │ Graph Refine │          │ Ensemble + Unc   │          │ Output     │ │
│    └──────────────┘          └──────────────────┘          └────────────┘ │
│            │                            │                                   │
│            └────────────────────────────┼───────────────────────────────┘   │
│                                         ▼                                   │
│                              Final Landmark Predictions                     │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Module A: MLMF (Multi-Layer Multi-Facet Adaptive Fusion)

### Motivation
The original FM-OSD uses features from a single ViT layer (layer 8 or 9) and a single facet (key). However, different layers capture different levels of semantic information, and different facets have different geometric meanings. MLMF enables adaptive combination of these diverse features.

### Key Components

1. **Multi-Layer Multi-Facet Extraction** (`extractor_gpu.py`)
   ```python
   descriptors = extractor.extract_multi_layer_multi_facet_descriptors(
       batch,
       layers=[5, 8, 11],    # Shallow, middle, deep layers
       facets=['key', 'value'],  # Different attention facets
       bin=True
   )
   ```

2. **Adaptive Fusion Module** (`post_net.py`)
   ```python
   from post_net import AdaptiveFusionModule, Upnet_v3_MLMF
   
   # Create MLMF-enabled network
   model = Upnet_v3_MLMF(
       size=(224, 224),
       in_channels=6528,
       out_channels=256,
       num_sources=6,  # 3 layers × 2 facets
       fusion_reduction=4
   )
   ```

### Training
```bash
python train1_mlmf.py \
    --dataset_pth /path/to/Cephalometric \
    --mlmf_layers 5,8,11 \
    --mlmf_facets key,value \
    --exp global_mlmf
```

### Paper Contribution
*"We propose MLMF to leverage the complementary nature of multi-layer multi-facet foundation model representations, learning anatomy-specific feature selection."*

---

## Module B: TCGR (Topology-Constrained Graph Refinement)

### Motivation
The original FM-OSD matches each landmark independently, ignoring anatomical structure constraints. TCGR uses graph neural networks to model inter-landmark relationships based on cephalometric anatomy.

### Key Components

1. **Anatomical Graph** (`landmark_graph.py`)
   - 19 cephalometric landmarks as nodes
   - Anatomical connections as edges (cranial base, maxilla, mandible, soft tissue)
   - Hardcoded based on medical knowledge

2. **Graph Refinement Network**
   ```python
   from landmark_graph import TCGRModule
   
   tcgr = TCGRModule(
       num_landmarks=19,
       coord_dim=2,
       score_dim=1,
       feature_dim=64,
       hidden_dim=128,
       num_layers=2,
       use_attention=True  # GAT vs GCN
   )
   
   # Forward pass
   refined_coords, offsets = tcgr(
       initial_coords,  # [B, 19, 2]
       similarity_scores,  # [B, 19]
       local_features  # [B, 19, 64]
   )
   ```

3. **Topology Loss**
   - Coordinate MSE loss
   - Topology consistency loss (preserves inter-landmark distances)

### Training
```bash
python train_tcgr.py \
    --dataset_pth /path/to/Cephalometric \
    --model_post_path models/global/model_post_best.pth \
    --tcgr_hidden_dim 128 \
    --tcgr_num_layers 2 \
    --exp tcgr
```

### Paper Contribution
*"We propose TCGR to enforce anatomical topology constraints via graph-based joint landmark refinement, going beyond independent correspondence matching."*

---

## Module C: TEU (Template Ensemble with Uncertainty)

### Motivation
With only a single template image, predictions are susceptible to template-specific biases. TEU addresses this by:
1. Generating multiple augmented template views
2. Aggregating predictions with uncertainty weighting
3. Providing calibrated confidence estimates

### Key Components

1. **Template Augmentation** (`template_uncertainty.py`)
   ```python
   from template_uncertainty import TemplateAugmentor
   
   augmentor = TemplateAugmentor(
       brightness_range=(0.85, 1.15),
       contrast_range=(0.85, 1.15),
       scale_range=(0.97, 1.03),
       rotation_range=(-3, 3),
       num_augmentations=5
   )
   ```

2. **Uncertainty Estimation**
   ```python
   from template_uncertainty import UncertaintyHead, EnsembleAggregator
   
   uncertainty_head = UncertaintyHead(feature_dim=256, num_landmarks=19)
   aggregator = EnsembleAggregator(
       num_landmarks=19,
       aggregation='uncertainty_weighted'
   )
   ```

3. **Full TEU Module**
   ```python
   from template_uncertainty import TEUModule
   
   teu = TEUModule(
       num_landmarks=19,
       feature_dim=256,
       num_augmentations=5,
       aggregation='uncertainty_weighted'
   )
   ```

### Paper Contribution
*"We propose TEU for robust one-shot learning through template ensemble and uncertainty-aware prediction aggregation."*

---

## Combined Inference

Use `test_mlmf_tcgr.py` for full inference with all modules:

```bash
python test_mlmf_tcgr.py \
    --dataset_pth /path/to/Cephalometric \
    --model_post_path models/global_mlmf/model_post_mlmf_best.pth \
    --tcgr_path models/tcgr/tcgr_best.pth \
    --use_tcgr True \
    --mlmf_layers 5,8,11 \
    --mlmf_facets key,value \
    --json_output results/predictions.json
```

---

## File Structure

```
FM-OSD/
├── extractor_gpu.py          # Added: extract_multi_layer_multi_facet_descriptors()
├── post_net.py               # Added: AdaptiveFusionModule, Upnet_v3_MLMF, Upnet_v3_MLMF_CoarseToFine
├── landmark_graph.py         # NEW: TCGR module with GCN/GAT
├── template_uncertainty.py   # NEW: TEU module with augmentation & uncertainty
├── train1_mlmf.py            # NEW: Training script for MLMF
├── train_tcgr.py             # NEW: Training script for TCGR
├── test_mlmf_tcgr.py         # NEW: Inference with MLMF + TCGR
├── train1.py                 # Original (unchanged)
├── train2.py                 # Original (unchanged)
├── test.py                   # Original (unchanged)
└── INNOVATIONS.md            # This file
```

---

## Ablation Study Design

For TMI submission, consider the following ablation experiments:

| Experiment | MLMF | TCGR | TEU | Description |
|------------|------|------|-----|-------------|
| Baseline | - | - | - | Original FM-OSD |
| +MLMF | ✓ | - | - | Multi-layer multi-facet only |
| +TCGR | - | ✓ | - | Graph refinement only |
| +TEU | - | - | ✓ | Template ensemble only |
| MLMF+TCGR | ✓ | ✓ | - | Feature + structure |
| Full | ✓ | ✓ | ✓ | All innovations |

---

## Citation

If you use these innovations, please cite:

```bibtex
@article{yourname2026fmosd,
  title={Enhanced Foundation Model-Enabled One-Shot Anatomical Landmark Detection with Multi-Layer Fusion and Topology Constraints},
  author={Your Name},
  journal={IEEE Transactions on Medical Imaging},
  year={2026}
}
```
