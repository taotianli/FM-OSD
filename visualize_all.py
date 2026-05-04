"""
Generate all paper figures:
  V1 - Qualitative comparison (GT vs FM-OSD vs MLMF+TCGR)
  V2 - Per-landmark MRE bar chart
  V3 - MLMF source attention weight heatmap
  V4 - SDR curves for all methods

Usage:
  python visualize_all.py --output_dir figures/
"""
import argparse, os, json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image

LANDMARK_NAMES = [
    'Sella', 'Nasion', 'Orbitale', 'Porion', 'Subspinale',
    'Supramentale', 'Pogonion', 'Menton', 'Gnathion', 'Gonion',
    'Lower Incisor Tip', 'Upper Incisor Tip', 'Upper Lip', 'Lower Lip',
    'Subnasale', 'Soft Tissue Pogonion', 'Post. Nasal Spine',
    'Ant. Nasal Spine', 'Articulare'
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_cache(cache_dir, split='test'):
    """Load all records from a cache directory split.
    Returns dict: img_id → {'pred': [...], 'gt': [...]}
    Handles both new format (dict with pred/gt) and old format (list = pred only).
    """
    d = os.path.join(cache_dir, split)
    records = {}
    for f in sorted(glob.glob(os.path.join(d, '*.json'))):
        img_id = os.path.basename(f).replace('.json', '')
        with open(f) as fp:
            data = json.load(fp)
        if isinstance(data, dict):
            records[img_id] = data           # new format: {'pred': ..., 'gt': ...}
        else:
            records[img_id] = {'pred': data, 'gt': None}   # old format: list of preds
    return records


def parse_metric_csv(csv_path):
    """Return (mre_mm, sdr_radii, sdr_values, per_landmark_mre)."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    mre     = float(df['Mean_RE_alllandmark'].dropna().iloc[0])
    sdr_r   = df['SDR'].dropna().values.tolist()
    sdr_v   = df['SDR-value'].dropna().values.tolist()
    per_lm  = df['Mean_RE_perlandmark'].dropna().values.tolist()
    return mre, sdr_r, sdr_v, per_lm


def read_gt_labels(dataset_pth, img_id):
    """Read averaged junior/senior GT for one image."""
    label_junior = os.path.join(dataset_pth, '400_junior')
    label_senior = os.path.join(dataset_pth, '400_senior')
    gt = []
    with open(os.path.join(label_junior, img_id + '.txt')) as f1, \
         open(os.path.join(label_senior, img_id + '.txt')) as f2:
        for _ in range(19):
            l1 = f1.readline().split()[0].split(',')
            l2 = f2.readline().split()[0].split(',')
            lm = [int(0.5*(int(l1[k])+int(l2[k]))) for k in range(len(l1))]
            gt.append([lm[1], lm[0]])  # [y, x]
    return gt


def compute_mre_per_landmark(cache_train, cache_test, spacing=0.1, dataset_pth=None):
    """Compute per-landmark MRE in mm from cache dicts."""
    errors = None
    all_records = list(cache_train.items()) + list(cache_test.items())
    for img_id, rec in all_records:
        pred = np.array(rec['pred'])
        gt   = rec.get('gt')
        if gt is None:
            if dataset_pth is None:
                continue
            gt = read_gt_labels(dataset_pth, img_id)
        gt = np.array(gt)
        err = np.sqrt(((pred - gt)**2).sum(axis=-1)) * spacing
        errors = err[None] if errors is None else np.vstack([errors, err[None]])
    return errors.mean(axis=0)


# ---------------------------------------------------------------------------
# V1: Qualitative comparison
# ---------------------------------------------------------------------------
def plot_qualitative(dataset_pth, fmosd_cache_dir, mlmf_cache_dir,
                     sample_ids, out_dir):
    """Overlay GT, FM-OSD, and MLMF+TCGR predictions on X-ray images."""
    img_dir_test = os.path.join(dataset_pth, 'RawImage', 'Test1Data')
    fmosd_test   = load_cache(fmosd_cache_dir, 'test')
    mlmf_test    = load_cache(mlmf_cache_dir,  'test')

    n_cols = 3  # GT | FM-OSD | MLMF+TCGR
    n_rows = len(sample_ids)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4, n_rows * 5))
    if n_rows == 1:
        axes = axes[None]

    col_labels = ['Ground Truth', 'FM-OSD', 'MLMF+FM-OSD+TCGR']
    colors     = ['#2ecc71', '#e74c3c', '#3498db']
    marker_sz  = 12

    for ri, img_id in enumerate(sample_ids):
        img_path = os.path.join(img_dir_test, img_id + '.bmp')
        if not os.path.exists(img_path):
            print(f"  WARNING: {img_path} not found, skipping V1 sample {img_id}")
            continue
        img = np.array(Image.open(img_path).convert('L'))
        H, W = img.shape

        rec_f = fmosd_test.get(img_id)
        rec_m = mlmf_test.get(img_id)
        if rec_f is None or rec_m is None:
            continue

        gt_raw = rec_f.get('gt')
        if gt_raw is None:
            gt_raw = read_gt_labels(args.dataset_pth, img_id)
        gt     = np.array(gt_raw)
        pred_f = np.array(rec_f['pred'])
        pred_m = np.array(rec_m['pred'])

        mre_f = np.sqrt(((pred_f - gt)**2).sum(axis=-1)).mean() * 0.1
        mre_m = np.sqrt(((pred_m - gt)**2).sum(axis=-1)).mean() * 0.1

        for ci, (pts, lbl, col) in enumerate(zip(
                [gt, pred_f, pred_m], col_labels, colors)):
            ax = axes[ri, ci]
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)
            ax.scatter(pts[:, 1], pts[:, 0], c=col, s=marker_sz,
                       marker='+', linewidths=1.5, zorder=3)
            title = lbl
            if ci == 1:
                title += f'\nMRE={mre_f:.2f}mm'
            elif ci == 2:
                title += f'\nMRE={mre_m:.2f}mm'
            ax.set_title(title, fontsize=9)
            ax.axis('off')

    plt.tight_layout()
    out = os.path.join(out_dir, 'V1_qualitative_comparison.pdf')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.savefig(out.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved V1 → {out}")


# ---------------------------------------------------------------------------
# V2: Per-landmark MRE bar chart
# ---------------------------------------------------------------------------
def plot_per_landmark_mre(fmosd_cache_dir, mlmf_cache_dir, out_dir):
    """Horizontal bar chart: per-landmark MRE for FM-OSD vs MLMF+TCGR."""
    fmosd_train = load_cache(fmosd_cache_dir, 'train')
    fmosd_test  = load_cache(fmosd_cache_dir, 'test')
    mlmf_train  = load_cache(mlmf_cache_dir,  'train')
    mlmf_test   = load_cache(mlmf_cache_dir,  'test')

    dset = '/home/taotl/Desktop/FM-OSD/dataset/Cephalometric/'
    mre_f = compute_mre_per_landmark(fmosd_train, fmosd_test, dataset_pth=dset)
    mre_m = compute_mre_per_landmark(mlmf_train,  mlmf_test,  dataset_pth=dset)

    idx   = np.arange(19)
    h     = 0.35

    fig, ax = plt.subplots(figsize=(8, 7))
    bars_f = ax.barh(idx + h/2, mre_f, h, label='FM-OSD',
                     color='#e74c3c', alpha=0.85)
    bars_m = ax.barh(idx - h/2, mre_m, h, label='MLMF+FM-OSD+TCGR',
                     color='#3498db', alpha=0.85)

    ax.set_yticks(idx)
    ax.set_yticklabels(LANDMARK_NAMES, fontsize=9)
    ax.set_xlabel('Mean Radial Error (mm)', fontsize=11)
    ax.set_title('Per-Landmark MRE Comparison', fontsize=12)
    ax.legend(fontsize=10)
    ax.axvline(mre_f.mean(), color='#e74c3c', linestyle='--', linewidth=1,
               alpha=0.6, label=f'FM-OSD avg={mre_f.mean():.2f}')
    ax.axvline(mre_m.mean(), color='#3498db', linestyle='--', linewidth=1,
               alpha=0.6, label=f'MLMF avg={mre_m.mean():.2f}')
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    out = os.path.join(out_dir, 'V2_per_landmark_mre.pdf')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.savefig(out.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved V2 → {out}")


# ---------------------------------------------------------------------------
# V3: MLMF attention weight visualization
# ---------------------------------------------------------------------------
def plot_attention_weights(out_dir, num_sources=6,
                           layers=(5, 8, 11), facets=('key', 'value')):
    """
    Visualise the learned attention-weight distribution.
    We probe the trained MLMF model by reading source_scale and source_attention
    bias, or approximate via random forward passes on the template.
    Here we show a static heatmap of the source_scale parameters which directly
    reflect each source's learnt importance.
    """
    import torch
    from post_net import Upnet_v3_MLMF_CoarseToFine

    ckpt = 'models/local_mlmf/model_post_fine_iter_20_1.8327.pth'
    if not os.path.exists(ckpt):
        print(f"V3: checkpoint not found ({ckpt}), skipping.")
        return

    model = Upnet_v3_MLMF_CoarseToFine(
        size=(224, 224), in_channels=6528, out_channels=256,
        num_sources=6, fusion_reduction=4)
    model.load_state_dict(torch.load(ckpt, map_location='cpu'))
    model.eval()

    scale_g = model.fusion_global.source_scale.detach().numpy()
    scale_l = model.fusion_local.source_scale.detach().numpy()
    scale_g = np.exp(scale_g) / np.exp(scale_g).sum()
    scale_l = np.exp(scale_l) / np.exp(scale_l).sum()

    source_labels = [f'L{l}-{f[0].upper()}'
                     for l in layers for f in facets]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, weights, title in zip(
            axes, [scale_g, scale_l],
            ['Global Branch Source Weights', 'Local Branch Source Weights']):
        weights_2d = weights.reshape(len(layers), len(facets))
        im = ax.imshow(weights_2d, cmap='YlOrRd', vmin=0, vmax=weights_2d.max()*1.2)
        ax.set_xticks(range(len(facets)))
        ax.set_xticklabels([f.capitalize() for f in facets], fontsize=11)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([f'Layer {l}' for l in layers], fontsize=11)
        ax.set_title(title, fontsize=11)
        for i in range(len(layers)):
            for j in range(len(facets)):
                ax.text(j, i, f'{weights_2d[i,j]:.3f}',
                        ha='center', va='center', fontsize=10,
                        color='white' if weights_2d[i,j] > 0.25 else 'black')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle('MLMF Adaptive Fusion — Learned Source Importance', fontsize=12)
    plt.tight_layout()
    out = os.path.join(out_dir, 'V3_mlmf_attention_weights.pdf')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.savefig(out.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved V3 → {out}")


# ---------------------------------------------------------------------------
# V4: SDR curves
# ---------------------------------------------------------------------------
def plot_sdr_curves(methods_data, out_dir):
    """
    methods_data: list of (label, color, list_of_(radius, sdr)) tuples.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, color, rd_pairs in methods_data:
        radii = [r for r, _ in rd_pairs]
        sdrs  = [s for _, s in rd_pairs]
        ax.plot(radii, sdrs, marker='o', linewidth=2, label=label,
                color=color, markersize=6)

    ax.set_xlabel('Detection Radius (mm)', fontsize=12)
    ax.set_ylabel('Success Detection Rate (%)', fontsize=12)
    ax.set_title('SDR Curves — Cephalometric Dataset', fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xticks([2, 2.5, 3, 4, 6, 8])
    ax.set_ylim(50, 101)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(out_dir, 'V4_sdr_curves.pdf')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.savefig(out.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved V4 → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_pth',
        default='/home/taotl/Desktop/FM-OSD/dataset/Cephalometric/')
    parser.add_argument('--fmosd_cache',
        default='/home/taotl/Desktop/FM-OSD/data/tcgr_cache')
    parser.add_argument('--mlmf_cache',
        default='/home/taotl/Desktop/FM-OSD/data/tcgr_cache_mlmf')
    parser.add_argument('--output_dir', default='figures')
    parser.add_argument('--qualitative_ids',
        default='160,170,183,200,230',
        help='Comma-separated test image IDs for V1 qualitative figure')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    sample_ids = args.qualitative_ids.split(',')

    # ------ V1: Qualitative ------
    print("\n--- V1: Qualitative comparison ---")
    plot_qualitative(args.dataset_pth, args.fmosd_cache, args.mlmf_cache,
                     sample_ids, args.output_dir)

    # ------ V2: Per-landmark MRE ------
    print("\n--- V2: Per-landmark MRE bar chart ---")
    plot_per_landmark_mre(args.fmosd_cache, args.mlmf_cache, args.output_dir)

    # ------ V3: MLMF attention ------
    print("\n--- V3: Attention weight heatmap ---")
    plot_attention_weights(args.output_dir)

    # ------ V4: SDR curves ------
    print("\n--- V4: SDR curves ---")
    methods_sdr = [
        ('FM-OSD (baseline)', '#e74c3c',
         [(2, 67.4), (2.5, 78.0), (3, 84.7), (4, 91.9), (6, 97.9), (8, 99.3)]),
        ('FM-OSD + TCGR', '#f39c12',
         [(2, 69.3), (2.5, 79.6), (3, 86.0), (4, 93.2), (6, 97.8), (8, 99.4)]),
        ('MLMF+FM-OSD+TCGR (Ours)', '#3498db',
         [(2, 69.6), (2.5, 79.3), (3, 85.9), (4, 93.3), (6, 98.2), (8, 99.4)]),
    ]
    plot_sdr_curves(methods_sdr, args.output_dir)

    print(f"\nAll figures saved to ./{args.output_dir}/")
