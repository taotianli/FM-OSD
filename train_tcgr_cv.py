# TCGR training with 5-fold cross-validation on ALL 400 labeled images
# Uses FM-OSD cached predictions from data/tcgr_cache/{train,test}/
# This script can also be rerun with MLMF predictions by passing --cache_dir

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from tensorboardX import SummaryWriter

from evaluation.eval import Evaluater
from landmark_graph import TCGRModule, TCGRLoss, normalize_coordinates, denormalize_coordinates


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


# ---------------------------------------------------------------------------
# Load all cached predictions + GT for all 400 images
# ---------------------------------------------------------------------------
def load_all_cache(train_cache_dir, test_cache_dir, dataset_pth):
    """Returns list of (img_id, pred [N,2], gt [N,2]) for all 400 images."""
    label_junior = os.path.join(dataset_pth, '400_junior')
    label_senior = os.path.join(dataset_pth, '400_senior')

    def read_gt(img_id):
        gt = []
        with open(os.path.join(label_junior, img_id + '.txt')) as f1, \
             open(os.path.join(label_senior, img_id + '.txt')) as f2:
            for _ in range(19):
                l1 = f1.readline().split()[0].split(',')
                l2 = f2.readline().split()[0].split(',')
                lm = [int(0.5 * (int(l1[k]) + int(l2[k]))) for k in range(len(l1))]
                gt.append([lm[1], lm[0]])  # [y, x]
        return gt

    all_items = []

    # Training images 001-150
    for i in range(1, 151):
        img_id = f"{i:03d}"
        cache_f = os.path.join(train_cache_dir, f"{img_id}.json")
        if not os.path.exists(cache_f):
            continue
        with open(cache_f) as f:
            data = json.load(f)
        pred = data['pred'] if isinstance(data, dict) else data
        gt   = data['gt']  if isinstance(data, dict) else read_gt(img_id)
        all_items.append((img_id, pred, gt))

    # Test images 151-400
    for i in range(151, 401):
        img_id = f"{i:03d}"
        cache_f = os.path.join(test_cache_dir, f"{img_id}.json")
        if not os.path.exists(cache_f):
            continue
        with open(cache_f) as f:
            data = json.load(f)
        pred = data['pred'] if isinstance(data, dict) else data
        gt   = data['gt']  if isinstance(data, dict) else read_gt(img_id)
        all_items.append((img_id, pred, gt))

    print(f"Total cached samples: {len(all_items)}")
    return all_items


class CachedListDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        _, pred, gt = self.items[idx]
        return (torch.tensor(pred, dtype=torch.float32),
                torch.tensor(gt,   dtype=torch.float32))


# ---------------------------------------------------------------------------
# Train one TCGR fold and return val predictions + GT
# ---------------------------------------------------------------------------
def train_fold(fold_idx, train_items, val_items, args, device, fold_writer):
    tcgr = TCGRModule(
        num_landmarks=19, coord_dim=2, score_dim=1,
        feature_dim=args.tcgr_feature_dim,
        hidden_dim=args.tcgr_hidden_dim,
        num_layers=args.tcgr_num_layers,
        use_attention=args.tcgr_use_attention
    ).to(device)

    loss_fn   = TCGRLoss(coord_weight=args.coord_loss_weight,
                         topo_weight=args.topo_loss_weight)
    optimizer = optim.Adam(tcgr.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_iterations, eta_min=1e-6)

    train_ds  = CachedListDataset(train_items)
    train_ldr = torch.utils.data.DataLoader(
        train_ds, batch_size=args.bs, shuffle=True, num_workers=2)

    best_val_mre = float('inf')
    best_state   = None
    iter_num = 0

    for epoch in range(1000):
        tcgr.train()
        for init_c, gt_c in train_ldr:
            iter_num += 1
            init_c = init_c.to(device)
            gt_c   = gt_c.to(device)
            B, N, _ = init_c.shape

            init_n = normalize_coordinates(init_c, args.input_size)
            gt_n   = normalize_coordinates(gt_c,   args.input_size)
            dummy_s = torch.ones(B, N, device=device)
            dummy_f = torch.zeros(B, N, args.tcgr_feature_dim, device=device)

            refined_n, _ = tcgr(init_n, dummy_s, dummy_f)
            loss, loss_dict = loss_fn(refined_n, gt_n, tcgr)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tcgr.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            fold_writer.add_scalar(f'fold{fold_idx}/loss', loss_dict['total_loss'], iter_num)

            if iter_num % 50 == 0:
                print(f"  [Fold {fold_idx} | Iter {iter_num}] "
                      f"loss={loss_dict['total_loss']:.4f}")

            if iter_num >= args.max_iterations:
                break
        if iter_num >= args.max_iterations:
            break

    # ---- evaluate on val set ----
    tcgr.eval()
    val_ds  = CachedListDataset(val_items)
    val_ldr = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)

    all_preds, all_gts = [], []
    with torch.no_grad():
        for init_c, gt_c in val_ldr:
            init_c = init_c.to(device)
            B, N, _ = init_c.shape
            init_n  = normalize_coordinates(init_c, args.input_size)
            dummy_s = torch.ones(B, N, device=device)
            dummy_f = torch.zeros(B, N, args.tcgr_feature_dim, device=device)

            refined_n, _ = tcgr(init_n, dummy_s, dummy_f)
            refined = denormalize_coordinates(refined_n, args.input_size)

            all_preds.extend(refined.cpu().numpy().tolist())
            all_gts.extend(gt_c.numpy().tolist())

    return all_preds, all_gts, tcgr.state_dict()


# ---------------------------------------------------------------------------
# Main: 5-fold CV
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TCGR 5-fold CV on all 400 images.')
    parser.add_argument('--dataset_pth', type=str,
                        default='/home/taotl/Desktop/FM-OSD/dataset/Cephalometric/')
    parser.add_argument('--input_size',  default=[2400, 1935])
    parser.add_argument('--eval_radius', default=[2, 2.5, 3, 4, 6, 8])

    parser.add_argument('--cache_dir', type=str,
                        default='/home/taotl/Desktop/FM-OSD/data/tcgr_cache')
    parser.add_argument('--save_dir',  type=str,
                        default='/home/taotl/Desktop/FM-OSD/output')

    parser.add_argument('--bs',              default=32,   type=int)
    parser.add_argument('--max_iterations',  default=3000, type=int)
    parser.add_argument('--lr',              default=1e-4, type=float)
    parser.add_argument('--n_folds',         default=5,    type=int)
    parser.add_argument('--exp',             default='tcgr_cv_alldata')

    parser.add_argument('--tcgr_hidden_dim',    default=128, type=int)
    parser.add_argument('--tcgr_num_layers',    default=3,   type=int)
    parser.add_argument('--tcgr_use_attention', default='True', type=str2bool)
    parser.add_argument('--tcgr_feature_dim',   default=64,  type=int)
    parser.add_argument('--coord_loss_weight',  default=1.0, type=float)
    parser.add_argument('--topo_loss_weight',   default=0.2, type=float)

    args = parser.parse_args()

    random_seed = 2022
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    train_cache = os.path.join(args.cache_dir, 'train')
    test_cache  = os.path.join(args.cache_dir, 'test')
    all_items   = load_all_cache(train_cache, test_cache, args.dataset_pth)

    # Shuffle with fixed seed for reproducibility
    rng = np.random.RandomState(2022)
    indices = rng.permutation(len(all_items))
    all_items = [all_items[i] for i in indices]

    snapshot_path = f'models/{args.exp}'
    os.makedirs(snapshot_path, exist_ok=True)
    log_writer = SummaryWriter(snapshot_path + '/log')

    eval_dir = Path(args.save_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # ----- 5-fold CV -----
    fold_size = len(all_items) // args.n_folds
    all_fold_preds, all_fold_gts = [], []

    for fold in range(args.n_folds):
        val_start = fold * fold_size
        val_end   = val_start + fold_size if fold < args.n_folds - 1 else len(all_items)
        val_items  = all_items[val_start:val_end]
        train_items = all_items[:val_start] + all_items[val_end:]

        print(f"\n{'='*60}")
        print(f"Fold {fold+1}/{args.n_folds}: "
              f"train={len(train_items)}, val={len(val_items)}")
        print(f"{'='*60}")

        fold_writer = SummaryWriter(snapshot_path + f'/log_fold{fold+1}')
        preds, gts, state = train_fold(fold+1, train_items, val_items, args, device, fold_writer)
        fold_writer.close()

        all_fold_preds.extend(preds)
        all_fold_gts.extend(gts)

        # Per-fold metrics
        evaluater = Evaluater(
            preds, gts, args.eval_radius, eval_dir,
            name=f'tcgr_cv_fold{fold+1}', spacing=[0.1, 0.1])
        evaluater.calculate()
        evaluater.cal_metrics()
        print(f"Fold {fold+1} MRE: {evaluater.mre:.4f} mm")
        log_writer.add_scalar('cv/fold_mre', evaluater.mre, fold)

        # Save this fold's model
        torch.save(state, os.path.join(snapshot_path, f'tcgr_fold{fold+1}.pth'))

    # ----- Overall cross-validation result -----
    print(f"\n{'='*60}")
    print("FINAL 5-fold CV results on ALL 400 images:")
    overall_eval = Evaluater(
        all_fold_preds, all_fold_gts, args.eval_radius, eval_dir,
        name='tcgr_cv_all400', spacing=[0.1, 0.1])
    overall_eval.calculate()
    overall_eval.cal_metrics()
    print(f"Overall MRE: {overall_eval.mre:.4f} mm")
    log_writer.close()
