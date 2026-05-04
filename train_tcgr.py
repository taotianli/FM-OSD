# Train the TCGR (Topology-Constrained Graph Refinement) module -- Plan B
#
# Plan B: Use the FULL FM-OSD pipeline (global + local coarse-to-fine) on real
# training images (001-150) to generate initial predictions, then train TCGR to
# refine them. Predictions are cached on disk so they are computed only once.
#
# At test time (test_mlmf_tcgr.py), TCGR is applied after the full FM-OSD
# coarse-to-fine pipeline, targeting improvement over the 1.819 mm baseline.

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

from extractor_gpu import ViTExtractor
from datasets.head import Head_SSL_Infer
from evaluation.eval import Evaluater
from post_net import Upnet_v3_coarsetofine2_tran_new
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
# Replicate the full FM-OSD coarse-to-fine inference for a single query image
# Returns: list of [y, x] coordinates in original-image space
# ---------------------------------------------------------------------------
def fmosd_predict_one(extractor, device, model_post,
                       image_path1, lab,
                       lab_feature_all, lab_feature_all_local,
                       descriptors1_post, descriptors1_post_local_all, gt_local_all,
                       query_path, original_size, load_size, layer, facet, bin, topk):
    """Run full FM-OSD (global + local) for ONE query image.

    Template features are precomputed and passed in to avoid repeated work.
    """
    with torch.no_grad():
        image2_batch, _ = extractor.preprocess(query_path, load_size)
        descriptors2 = extractor.extract_descriptors(image2_batch.to(device), layer, facet, bin)
        num_patches2, load_size2 = extractor.num_patches, extractor.load_size

        descriptors2_post = model_post(descriptors2, num_patches2, load_size2, islocal=False)
        descriptors2_post_large = torch.nn.functional.interpolate(
            descriptors2_post, original_size, mode='bilinear')

        size_y1, size_x1 = descriptors1_post.shape[-2:]

        predictions = []

        for i in range(len(lab)):
            # ---- global matching with reverse verification ----
            similarities = torch.nn.CosineSimilarity(dim=0)(
                lab_feature_all[i], descriptors2_post[0])
            h2, w2 = similarities.shape
            similarities_flat = similarities.reshape(-1)
            sim_k, nn_k = torch.topk(similarities_flat, k=topk, dim=-1, largest=True)

            distance_best = 1e18
            index_best = 0
            for k in range(topk):
                iy = nn_k[k] // w2
                ix = nn_k[k] % w2
                sim_rev = torch.nn.CosineSimilarity(dim=0)(
                    descriptors2_post[0, :, iy, ix].unsqueeze(1).unsqueeze(2),
                    descriptors1_post[0])
                _, nn_rev = torch.max(sim_rev.reshape(-1), dim=-1)
                ry = nn_rev // size_x1
                rx = nn_rev % size_x1
                x1s = rx / size_x1 * original_size[1]
                y1s = ry / size_y1 * original_size[0]
                d = (y1s - int(lab[i][0])) ** 2 + (x1s - int(lab[i][1])) ** 2
                if d < distance_best:
                    distance_best = d
                    index_best = k

            size_y2, size_x2 = descriptors2_post.shape[-2:]
            best_flat = nn_k[index_best].item()
            y2_global = best_flat // size_x2
            x2_global = best_flat % size_x2
            y2_show = round(y2_global / size_y2 * original_size[0])
            x2_show = round(x2_global / size_x2 * original_size[1])

            # ---- local refinement ----
            image2_batch_local, _, gt_local2, offset2, crop_feature2 = \
                extractor.preprocess_local_withfeature(
                    query_path, load_size,
                    [int(y2_show), int(x2_show)],
                    descriptors2_post_large)

            descriptors2_local = extractor.extract_descriptors(
                image2_batch_local.to(device), layer, facet, bin)
            num_patches2_local, load_size2_local = extractor.num_patches, extractor.load_size

            descriptors2_post_local = model_post(
                descriptors2_local, num_patches2_local, load_size2_local, islocal=True)
            descriptors2_post_local = (
                nn.functional.normalize(descriptors2_post_local, dim=1)
                + nn.functional.normalize(crop_feature2, dim=1))

            similarities_local = torch.nn.CosineSimilarity(dim=0)(
                lab_feature_all_local[i], descriptors2_post_local[0])
            h2l, w2l = similarities_local.shape
            sim_local_flat = similarities_local.reshape(-1)
            sim_kl, nn_kl = torch.topk(sim_local_flat, k=topk, dim=-1, largest=True)

            distance_best_local = 1e18
            index_best_local = 0
            size_y1l, size_x1l = descriptors1_post_local_all[i].shape[-2:]
            for k in range(topk):
                iyl = nn_kl[k] // w2l
                ixl = nn_kl[k] % w2l
                sim_rev_l = torch.nn.CosineSimilarity(dim=0)(
                    descriptors2_post_local[0, :, iyl, ixl].unsqueeze(1).unsqueeze(2),
                    descriptors1_post_local_all[i][0])
                _, nn_rev_l = torch.max(sim_rev_l.reshape(-1), dim=-1)
                ryl = nn_rev_l // size_x1l
                rxl = nn_rev_l % size_x1l
                d = (ryl - gt_local_all[i][0]) ** 2 + (rxl - gt_local_all[i][1]) ** 2
                if d < distance_best_local:
                    distance_best_local = d
                    index_best_local = k

            size_y2l, size_x2l = descriptors2_post_local.shape[-2:]
            best_flat_local = nn_kl[index_best_local].item()
            y2_local = best_flat_local // size_x2l
            x2_local = best_flat_local % size_x2l
            y_final = int(offset2[0]) + y2_local
            x_final = int(offset2[1]) + x2_local

            predictions.append([float(y_final), float(x_final)])

    return predictions


# ---------------------------------------------------------------------------
# Pre-compute and cache FM-OSD predictions on training images
# ---------------------------------------------------------------------------
def precompute_train_predictions(extractor, device, model_post,
                                  template_path, template_landmarks,
                                  dataset_pth, original_size,
                                  load_size, layer, facet, bin, topk,
                                  cache_dir):
    """Compute FM-OSD predictions for all training images and save to cache_dir.

    Skips images whose cache file already exists.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # ---- precompute template features (done once) ----
    print("Precomputing template features...")
    with torch.no_grad():
        image1_batch, _ = extractor.preprocess(template_path, load_size)
        descriptors1 = extractor.extract_descriptors(image1_batch.to(device), layer, facet, bin)
        num_patches1, load_size1 = extractor.num_patches, extractor.load_size
        descriptors1_post = model_post(descriptors1, num_patches1, load_size1, islocal=False)
        descriptors1_post_large = torch.nn.functional.interpolate(
            descriptors1_post, original_size, mode='bilinear')

        lab = template_landmarks
        lab_feature_all = []
        lab_feature_all_local = []
        descriptors1_post_local_all = []
        gt_local_all = []

        for i in range(len(lab)):
            lab_y = int(lab[i][0])
            lab_x = int(lab[i][1])
            size_y, size_x = descriptors1_post.shape[-2:]
            lab_y_f = int(lab_y / original_size[0] * size_y)
            lab_x_f = int(lab_x / original_size[1] * size_x)
            lf = descriptors1_post[0, :, lab_y_f, lab_x_f].unsqueeze(1).unsqueeze(2)
            lab_feature_all.append(lf)

            image1_batch_local, _, gt_local, offset, crop_feature = \
                extractor.preprocess_local_withfeature(
                    template_path, load_size, [int(lab[i][0]), int(lab[i][1])],
                    descriptors1_post_large)
            gt_local_all.append(gt_local)

            descriptors1_local = extractor.extract_descriptors(
                image1_batch_local.to(device), layer, facet, bin)
            num_patches1_local, load_size1_local = extractor.num_patches, extractor.load_size
            descriptors1_post_local = model_post(
                descriptors1_local, num_patches1_local, load_size1_local, islocal=True)
            descriptors1_post_local = (
                nn.functional.normalize(descriptors1_post_local, dim=1)
                + nn.functional.normalize(crop_feature, dim=1))
            descriptors1_post_local_all.append(descriptors1_post_local)

            lf_local = descriptors1_post_local[0, :, gt_local[0], gt_local[1]].unsqueeze(1).unsqueeze(2)
            lab_feature_all_local.append(lf_local)

    # ---- iterate training images (001-150) ----
    train_img_dir = os.path.join(dataset_pth, 'RawImage', 'TrainingData')
    ids = [f"{i:03d}" for i in range(1, 151)]

    print(f"Caching FM-OSD predictions for {len(ids)} training images...")
    for img_id in tqdm(ids):
        cache_file = os.path.join(cache_dir, f"{img_id}.json")
        if os.path.exists(cache_file):
            continue
        query_path = os.path.join(train_img_dir, img_id + '.bmp')
        if not os.path.exists(query_path):
            continue
        preds = fmosd_predict_one(
            extractor, device, model_post,
            template_path, lab,
            lab_feature_all, lab_feature_all_local,
            descriptors1_post, descriptors1_post_local_all, gt_local_all,
            query_path, original_size, load_size, layer, facet, bin, topk)
        with open(cache_file, 'w') as f:
            json.dump(preds, f)

    print("Cache complete.")
    return (lab_feature_all, lab_feature_all_local,
            descriptors1_post, descriptors1_post_local_all, gt_local_all)


# ---------------------------------------------------------------------------
# Pre-compute and cache FM-OSD predictions on TEST images (one-time cost)
# ---------------------------------------------------------------------------
def precompute_test_predictions(extractor, device, model_post,
                                 template_path, template_landmarks,
                                 dataset_pth, original_size,
                                 load_size, layer, facet, bin, topk,
                                 cache_dir,
                                 lab_feature_all, lab_feature_all_local,
                                 descriptors1_post, descriptors1_post_local_all, gt_local_all):
    """Cache FM-OSD predictions for test images (151-400). Skips existing files."""
    os.makedirs(cache_dir, exist_ok=True)
    test_img_dir = os.path.join(dataset_pth, 'RawImage', 'Test1Data')
    label_junior = os.path.join(dataset_pth, '400_junior')
    label_senior = os.path.join(dataset_pth, '400_senior')

    ids = [f"{i:03d}" for i in range(151, 401)]
    missing = [img_id for img_id in ids
               if not os.path.exists(os.path.join(cache_dir, f"{img_id}.json"))]
    if not missing:
        print(f"Test cache already complete ({len(ids)} files).")
        return

    print(f"Caching FM-OSD predictions for {len(missing)} test images...")
    lab = template_landmarks
    for img_id in tqdm(missing):
        query_path = os.path.join(test_img_dir, img_id + '.bmp')
        if not os.path.exists(query_path):
            continue
        preds = fmosd_predict_one(
            extractor, device, model_post,
            template_path, lab,
            lab_feature_all, lab_feature_all_local,
            descriptors1_post, descriptors1_post_local_all, gt_local_all,
            query_path, original_size, load_size, layer, facet, bin, topk)

        # also read ground truth and save together
        gt = []
        with open(os.path.join(label_junior, img_id + '.txt')) as f1, \
             open(os.path.join(label_senior, img_id + '.txt')) as f2:
            for _ in range(19):
                l1 = f1.readline().split()[0].split(',')
                l2 = f2.readline().split()[0].split(',')
                lm = [int(0.5 * (int(l1[k]) + int(l2[k]))) for k in range(len(l1))]
                gt.append([lm[1], lm[0]])  # [y, x]
        with open(os.path.join(cache_dir, f"{img_id}.json"), 'w') as f:
            json.dump({'pred': preds, 'gt': gt}, f)
    print("Test cache complete.")


# ---------------------------------------------------------------------------
# Dataset that reads cached FM-OSD predictions + ground truth
# ---------------------------------------------------------------------------
class CachedPredDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir, dataset_pth, original_size, id_range=(1, 150)):
        self.cache_dir = cache_dir
        self.original_size = original_size
        self.pth_label_junior = os.path.join(dataset_pth, '400_junior')
        self.pth_label_senior = os.path.join(dataset_pth, '400_senior')
        self.num_landmark = 19
        self.ids = []
        for i in range(id_range[0], id_range[1] + 1):
            img_id = f"{i:03d}"
            cache_file = os.path.join(cache_dir, f"{img_id}.json")
            if os.path.exists(cache_file):
                self.ids.append(img_id)
        print(f"CachedPredDataset: {len(self.ids)} samples (ids {id_range[0]}-{id_range[1]}).")

    def _get_gt(self, img_id):
        gt = []
        with open(os.path.join(self.pth_label_junior, img_id + '.txt')) as f1, \
             open(os.path.join(self.pth_label_senior, img_id + '.txt')) as f2:
            for _ in range(self.num_landmark):
                l1 = f1.readline().split()[0].split(',')
                l2 = f2.readline().split()[0].split(',')
                lm = [int(0.5 * (int(l1[k]) + int(l2[k]))) for k in range(len(l1))]
                gt.append([lm[1], lm[0]])  # [y, x] in original space
        return gt

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        with open(os.path.join(self.cache_dir, f"{img_id}.json")) as f:
            data = json.load(f)
        # Training cache stores raw list; test cache stores {'pred': ..., 'gt': ...}
        if isinstance(data, dict):
            preds = data['pred']
            gt    = data['gt']
        else:
            preds = data
            gt    = self._get_gt(img_id)
        return (torch.tensor(preds, dtype=torch.float32),   # [N, 2]
                torch.tensor(gt,    dtype=torch.float32))   # [N, 2]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train TCGR -- Plan B (full FM-OSD inputs).')
    parser.add_argument('--save_dir',    type=str,  default='output')
    parser.add_argument('--load_size',   default=224,        type=int)
    parser.add_argument('--stride',      default=4,          type=int)
    parser.add_argument('--model_type',  default='dino_vits8', type=str)
    parser.add_argument('--facet',       default='key',      type=str)
    parser.add_argument('--layer',       default=8,          type=int)
    parser.add_argument('--bin',         default='True',     type=str2bool)
    parser.add_argument('--topk',        default=3,          type=int)

    parser.add_argument('--dataset_pth', type=str,
                        default='/home/taotl/Desktop/FM-OSD/dataset/Cephalometric/')
    parser.add_argument('--input_size',  default=[2400, 1935])
    parser.add_argument('--id_shot',     default=125,        type=int)
    parser.add_argument('--eval_radius', default=[2, 2.5, 3, 4, 6, 8])

    parser.add_argument('--bs',              default=16,      type=int)
    parser.add_argument('--max_epoch',       default=200,     type=int)
    parser.add_argument('--max_iterations',  default=10000,   type=int)
    parser.add_argument('--lr',              default=1e-4,    type=float)
    parser.add_argument('--exp',             default='tcgr_planb')

    parser.add_argument('--tcgr_hidden_dim',    default=128, type=int)
    parser.add_argument('--tcgr_num_layers',    default=3,   type=int)
    parser.add_argument('--tcgr_use_attention', default='True', type=str2bool)
    parser.add_argument('--tcgr_feature_dim',   default=64,  type=int)
    parser.add_argument('--coord_loss_weight',  default=1.0, type=float)
    parser.add_argument('--topo_loss_weight',   default=0.1, type=float)

    parser.add_argument('--cache_dir', type=str,
                        default='/home/taotl/Desktop/FM-OSD/data/tcgr_cache')
    parser.add_argument('--force_recompute', default='False', type=str2bool,
                        help='Recompute cached predictions even if they exist.')
    args = parser.parse_args()

    random_seed = 2022
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ---- load template info ----
    one_shot_loader = Head_SSL_Infer(
        pathDataset=args.dataset_pth,
        mode='Oneshot', size=args.input_size, id_oneshot=args.id_shot)
    _, landmarks_temp, img_path_temp = one_shot_loader.__getitem__(0)
    print(f"Template: {img_path_temp}, {len(landmarks_temp)} landmarks")

    # ---- load full FM-OSD model (frozen) ----
    image_size = (args.load_size, args.load_size)
    model_post = Upnet_v3_coarsetofine2_tran_new(image_size, 6528, 256).to(device)

    fine_ckpt  = '/home/taotl/Desktop/FM-OSD/models/model_post_fine_iter_20.pth'
    global_ckpt = '/home/taotl/Desktop/FM-OSD/models/model_post_iter_9450.pth'

    # Load fine checkpoint and patch conv_out1 from global checkpoint
    model_post.load_state_dict(torch.load(fine_ckpt, map_location=device))
    model_dict = model_post.state_dict()
    pretrained_global = torch.load(global_ckpt, map_location=device)
    keys_co1 = [k for k in model_dict.keys() if 'conv_out1' in k]
    for k in keys_co1:
        parts = k.split('.')
        key_g = parts[0][:-1]
        for p in parts[1:]:
            key_g += '.' + p
        if key_g in pretrained_global:
            model_dict[k] = pretrained_global[key_g]
    model_post.load_state_dict(model_dict)
    model_post.eval()
    for p in model_post.parameters():
        p.requires_grad_(False)
    print("FM-OSD post_net loaded and frozen.")

    extractor = ViTExtractor(args.model_type, args.stride, device=device)

    # ---- pre-compute (or load from cache) FM-OSD predictions ----
    if args.force_recompute and os.path.exists(args.cache_dir):
        import shutil
        shutil.rmtree(args.cache_dir)

    train_cache_dir = os.path.join(args.cache_dir, 'train')
    test_cache_dir  = os.path.join(args.cache_dir, 'test')

    (lab_feature_all, lab_feature_all_local,
     descriptors1_post, descriptors1_post_local_all, gt_local_all) = \
        precompute_train_predictions(
            extractor, device, model_post,
            img_path_temp, landmarks_temp,
            args.dataset_pth, args.input_size,
            args.load_size, args.layer, args.facet, args.bin, args.topk,
            train_cache_dir)

    # ---- also cache test predictions (avoids running FM-OSD on every eval) ----
    precompute_test_predictions(
        extractor, device, model_post,
        img_path_temp, landmarks_temp,
        args.dataset_pth, args.input_size,
        args.load_size, args.layer, args.facet, args.bin, args.topk,
        test_cache_dir,
        lab_feature_all, lab_feature_all_local,
        descriptors1_post, descriptors1_post_local_all, gt_local_all)

    # ---- training dataset (cached predictions + GT) ----
    train_dataset = CachedPredDataset(train_cache_dir, args.dataset_pth, args.input_size,
                                       id_range=(1, 150))
    train_loader  = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.bs, shuffle=True, num_workers=2)

    # ---- test dataset (cached FM-OSD predictions) ----
    test_dataset = CachedPredDataset(test_cache_dir, args.dataset_pth, args.input_size,
                                      id_range=(151, 400))
    test_loader  = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # ---- TCGR module ----
    from landmark_graph import TCGRModule, TCGRLoss, normalize_coordinates, denormalize_coordinates
    tcgr_module = TCGRModule(
        num_landmarks=19,
        coord_dim=2,
        score_dim=1,
        feature_dim=args.tcgr_feature_dim,
        hidden_dim=args.tcgr_hidden_dim,
        num_layers=args.tcgr_num_layers,
        use_attention=args.tcgr_use_attention
    ).to(device)

    tcgr_loss_fn = TCGRLoss(
        coord_weight=args.coord_loss_weight,
        topo_weight=args.topo_loss_weight)

    optimizer = optim.Adam(tcgr_module.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_iterations, eta_min=1e-6)

    snapshot_path = 'models/' + args.exp
    os.makedirs(snapshot_path, exist_ok=True)
    writer = SummaryWriter(snapshot_path + '/log')

    best_mre = float('inf')
    iter_num  = 0

    print("Starting TCGR Plan-B training...")
    print(f"  Training on {len(train_dataset)} cached FM-OSD predictions (train images 001-150)")
    print(f"  Evaluating on {len(test_dataset)} test images (151-400)")

    for epoch in range(args.max_epoch):
        tcgr_module.train()

        for init_coords, gt_coords in train_loader:
            iter_num += 1
            init_coords = init_coords.to(device)   # [B, N, 2]  y,x in orig space
            gt_coords   = gt_coords.to(device)     # [B, N, 2]

            init_coords_norm = normalize_coordinates(init_coords, args.input_size)
            gt_coords_norm   = normalize_coordinates(gt_coords,   args.input_size)

            # TCGR has no per-landmark feature input from cache -- use zeros
            B, N, _ = init_coords.shape
            dummy_scores   = torch.ones(B, N, device=device)
            dummy_features = torch.zeros(B, N, args.tcgr_feature_dim, device=device)

            refined_coords_norm, offsets = tcgr_module(
                init_coords_norm, dummy_scores, dummy_features)

            loss, loss_dict = tcgr_loss_fn(refined_coords_norm, gt_coords_norm, tcgr_module)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tcgr_module.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            writer.add_scalar('loss/total', loss_dict['total_loss'], iter_num)
            writer.add_scalar('loss/coord', loss_dict['coord_loss'], iter_num)
            writer.add_scalar('loss/topo',  loss_dict['topo_loss'],  iter_num)

            if iter_num % 20 == 0:
                print(f"[Iter {iter_num}] loss={loss_dict['total_loss']:.4f}  "
                      f"coord={loss_dict['coord_loss']:.4f}  "
                      f"topo={loss_dict['topo_loss']:.4f}")

            # ---- evaluation (fast: TCGR-only on cached FM-OSD predictions) ----
            if iter_num % 200 == 0:
                tcgr_module.eval()
                all_preds = []
                all_gts   = []

                with torch.no_grad():
                    for init_coords_t, gt_coords_t in test_loader:
                        init_coords_t = init_coords_t.to(device)
                        gt_coords_t   = gt_coords_t.to(device)

                        B = init_coords_t.shape[0]
                        init_norm = normalize_coordinates(init_coords_t, args.input_size)
                        dummy_s = torch.ones(B, 19, device=device)
                        dummy_f = torch.zeros(B, 19, args.tcgr_feature_dim, device=device)

                        refined_n, _ = tcgr_module(init_norm, dummy_s, dummy_f)
                        refined = denormalize_coordinates(refined_n, args.input_size)

                        all_preds.extend(refined.cpu().numpy().tolist())
                        all_gts.extend(gt_coords_t.cpu().numpy().tolist())

                eval_dir = Path(args.save_dir)
                eval_dir.mkdir(parents=True, exist_ok=True)
                evaluater = Evaluater(
                    all_preds, all_gts, args.eval_radius, eval_dir,
                    name=f'tcgr_planb_iter_{iter_num}',
                    spacing=[0.1, 0.1])
                evaluater.calculate()
                evaluater.cal_metrics()

                writer.add_scalar('eval/mre', evaluater.mre, iter_num)

                if evaluater.mre < best_mre:
                    best_mre = evaluater.mre
                    ckpt = os.path.join(snapshot_path, f'tcgr_best_{best_mre:.4f}.pth')
                    torch.save(tcgr_module.state_dict(), ckpt)
                    print(f"  ** New best MRE: {best_mre:.4f} mm — saved to {ckpt}")

                tcgr_module.train()

            if iter_num >= args.max_iterations:
                break
        if iter_num >= args.max_iterations:
            break

    final_path = os.path.join(snapshot_path, 'tcgr_final.pth')
    torch.save(tcgr_module.state_dict(), final_path)
    print(f"Training done. Best MRE: {best_mre:.4f} mm")
    print(f"Final model: {final_path}")
