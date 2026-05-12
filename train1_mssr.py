# Train global branch with MLMF + MSSR (Mamba-Style Selective State-Space Refinement)
# Drop-in replacement for train1_mlmf.py — only the model class changes.
import argparse
import torch
from pathlib import Path
from extractor_gpu import ViTExtractor
import numpy as np
from PIL import Image

from datasets.head_train import TrainDataset
from datasets.head import Head_SSL_Infer
from torch.utils.data import DataLoader
from evaluation.eval import Evaluater
from post_net import Upnet_v3_MLMF_MSSR
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import os
import random


# ── Feature cache helpers (same pattern as train1_mlmf_hand.py) ──────────────

def init_feat_cache(feat_cache_dir, model_type):
    if not feat_cache_dir:
        return None
    cache_dir = Path(feat_cache_dir) / model_type
    if not cache_dir.exists():
        print(f'[cache] {cache_dir} not found, falling back to live encode.')
        return None
    return cache_dir


def get_cached_feat(cache_dir, prefix, stem, device):
    cache_f = cache_dir / f'{prefix}_{stem}.pt'
    if not cache_f.exists():
        return None, None
    cached = torch.load(cache_f, map_location=device)
    desc_list = [d.float() for d in cached['desc_list']]
    return desc_list, cached['num_patches']


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def make_heatmap(landmark, size, var=5.0):
    length, width = size
    x, y = torch.meshgrid(torch.arange(0, length), torch.arange(0, width))
    p = torch.stack([x, y], dim=2).float()
    inner_factor = -1 / (2 * (var ** 2))
    mean = torch.as_tensor(landmark).float()
    heatmap = (p - mean).pow(2).sum(dim=-1)
    return torch.exp(heatmap * inner_factor)


def heatmap_mse_loss(features, landmarks, var=5.0, criterion=nn.MSELoss()):
    lab = []
    for i in range(len(landmarks)):
        labels = landmarks[i]
        labtemp = [make_heatmap(labels[l], [features.shape[-2], features.shape[-1]], var=var)
                   for l in range(labels.shape[0])]
        lab.append(torch.stack(labtemp, dim=0))
    label = torch.stack(lab, dim=0).to(features.device)

    pred = []
    for i in range(len(landmarks)):
        feature_temp = features[i]
        pred_temp = []
        for j in range(landmarks[i].shape[0]):
            gt = feature_temp[:, landmarks[i, j, 0], landmarks[i, j, 1]].unsqueeze(1).unsqueeze(2)
            similarity = torch.nn.CosineSimilarity(dim=0)(gt, feature_temp).unsqueeze(0)
            pred_temp.append(similarity)
        pred.append(torch.cat(pred_temp, dim=0).unsqueeze(0))
    pred = torch.cat(pred, dim=0)
    return criterion(pred, label)


def find_landmark_all_mssr(extractor, device, model_post, image_path1, dataloader, lab,
                            mlmf_config, load_size=224, original_size=[2400, 1935], topk=5,
                            bin=True, feat_cache_dir=None, shot_mem_cache=None):
    # ── Template ──────────────────────────────────────────────────────────────
    if shot_mem_cache is not None:
        desc1_list, num_patches1 = shot_mem_cache
    elif feat_cache_dir is not None:
        stem = Path(image_path1).stem
        desc1_list, num_patches1 = get_cached_feat(feat_cache_dir, 'shot', stem, device)
        if desc1_list is None:
            image1_batch, _ = extractor.preprocess(image_path1, load_size)
            desc1_list = extractor.extract_multi_layer_multi_facet_descriptors(
                image1_batch.to(device),
                layers=mlmf_config['layers'], facets=mlmf_config['facets'], bin=bin)
            num_patches1 = extractor.num_patches
    else:
        image1_batch, _ = extractor.preprocess(image_path1, load_size)
        desc1_list = extractor.extract_multi_layer_multi_facet_descriptors(
            image1_batch.to(device),
            layers=mlmf_config['layers'], facets=mlmf_config['facets'], bin=bin)
        num_patches1 = extractor.num_patches

    descriptors1_post = model_post(desc1_list, num_patches1)

    lab_feature_all = []
    for i in range(len(lab)):
        size_y, size_x = descriptors1_post.shape[-2:]
        ly = int(int(lab[i][0]) / original_size[0] * size_y)
        lx = int(int(lab[i][1]) / original_size[1] * size_x)
        lab_feature_all.append(descriptors1_post[0, :, ly, lx].unsqueeze(1).unsqueeze(2))

    pred_all, gt_all, imgs_all, img_names_all = [], [], [], []

    for image, landmark_list, img_path_query in dataloader:
        # ── Query ─────────────────────────────────────────────────────────────
        if feat_cache_dir is not None:
            stem2 = Path(img_path_query[0]).stem
            desc2_list, num_patches2 = get_cached_feat(feat_cache_dir, 'test', stem2, device)
            if desc2_list is None:
                image2_batch, _ = extractor.preprocess(img_path_query[0], load_size)
                desc2_list = extractor.extract_multi_layer_multi_facet_descriptors(
                    image2_batch.to(device),
                    layers=mlmf_config['layers'], facets=mlmf_config['facets'], bin=bin)
                num_patches2 = extractor.num_patches
        else:
            image2_batch, _ = extractor.preprocess(img_path_query[0], load_size)
            desc2_list = extractor.extract_multi_layer_multi_facet_descriptors(
                image2_batch.to(device),
                layers=mlmf_config['layers'], facets=mlmf_config['facets'], bin=bin)
            num_patches2 = extractor.num_patches

        descriptors2_post = model_post(desc2_list, num_patches2)

        points1, points2 = [], []
        imgs_all.append(image)
        img_names_all.append(img_path_query)

        for i in range(len(lab)):
            points1.append([landmark_list[i][0].item(), landmark_list[i][1].item()])

            sim = torch.nn.CosineSimilarity(dim=0)(lab_feature_all[i], descriptors2_post[0])
            h2, w2 = sim.shape
            _, nn_k = torch.topk(sim.reshape(-1), k=topk, largest=True)

            dist_best, idx_best = 1e18, 0
            sz_y, sz_x = descriptors1_post.shape[-2:]
            for k in range(topk):
                iy, ix = nn_k[k] // w2, nn_k[k] % w2
                sim_rev = torch.nn.CosineSimilarity(dim=0)(
                    descriptors2_post[0, :, iy, ix].unsqueeze(1).unsqueeze(2),
                    descriptors1_post[0])
                _, nn_1 = torch.max(sim_rev.reshape(-1), dim=-1)
                y1s = (nn_1 // sz_x) / sz_y * original_size[0]
                x1s = (nn_1 % sz_x) / sz_x * original_size[1]
                d = (y1s - int(lab[i][0])) ** 2 + (x1s - int(lab[i][1])) ** 2
                if d < dist_best:
                    dist_best, idx_best = d, k

            best = nn_k[idx_best].item()
            sz_y2, sz_x2 = descriptors2_post.shape[-2:]
            y2 = np.round(best // sz_x2 / sz_y2 * original_size[0])
            x2 = np.round(best % sz_x2 / sz_x2 * original_size[1])
            points2.append([y2, x2])

        pred_all.append(points2)
        gt_all.append(points1)

    return pred_all, gt_all, imgs_all, img_names_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train global branch with MLMF + MSSR.')
    parser.add_argument('--save_dir', type=str, default='/home/taotl/Desktop/FM-OSD/output')
    parser.add_argument('--load_size', default=224, type=int)
    parser.add_argument('--stride', default=4, type=int)
    parser.add_argument('--model_type', default='dino_vits8', type=str)
    parser.add_argument('--bin', default='True', type=str2bool)
    parser.add_argument('--topk', default=5, type=int)

    parser.add_argument('--dataset_pth', type=str,
                        default='/home/taotl/Desktop/FM-OSD/dataset/Cephalometric/')
    parser.add_argument('--input_size', default=[2400, 1935])
    parser.add_argument('--id_shot', default=125, type=int)
    parser.add_argument('--eval_radius', default=[2, 2.5, 3, 4, 6, 8])

    parser.add_argument('--bs', default=4, type=int)
    parser.add_argument('--max_epoch', default=300, type=int)
    parser.add_argument('--max_iterations', type=int, default=1000)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--exp', default='global_mssr')
    parser.add_argument('--eval_freq', type=int, default=100,
                        help='Evaluate every N iterations')
    parser.add_argument('--early_stop_patience', type=int, default=5,
                        help='Stop if MRE does not improve for this many eval rounds. 0=disabled')

    parser.add_argument('--mlmf_layers', default='5,8,11', type=str)
    parser.add_argument('--mlmf_facets', default='key,value', type=str)
    parser.add_argument('--mssr_d_state', default=8, type=int,
                        help='SSM state dimension (C2 ablation: 8/16/32). '
                             'Default 8 is 2x faster than 16 with minimal quality loss.')
    parser.add_argument('--mssr_expand', default=1, type=int,
                        help='Inner expansion factor (C3 ablation: 1/2/4). '
                             'Default 1 avoids doubling channel count, ~2x faster.')
    parser.add_argument('--mssr_direction', default='forward', type=str,
                        choices=['bidir', 'forward', 'backward'],
                        help='Scan direction (C1 ablation). '
                             'Default forward is 2x faster than bidir.')
    parser.add_argument('--feat_cache_dir', default='', type=str,
                        help='Path to precomputed MLMF feature cache (cache/feat_head). '
                             'Empty = live encode every iter (slow).')

    args = parser.parse_args()

    mlmf_config = {
        'layers': [int(x) for x in args.mlmf_layers.split(',')],
        'facets': args.mlmf_facets.split(','),
        'num_sources': len(args.mlmf_layers.split(',')) * len(args.mlmf_facets.split(','))
    }
    print(f"MLMF+MSSR Config: {mlmf_config}, d_state={args.mssr_d_state}, "
          f"expand={args.mssr_expand}, direction={args.mssr_direction}")

    random.seed(2022)
    np.random.seed(2022)
    torch.manual_seed(2022)
    torch.cuda.manual_seed(2022)

    train_dataset = TrainDataset(istrain=0, original_size=args.input_size, load_size=args.load_size)
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=2)

    one_shot = Head_SSL_Infer(args.dataset_pth, mode='Oneshot',
                               size=args.input_size, id_oneshot=args.id_shot)
    _, landmarks_temp, img_path_temp = one_shot.__getitem__(0)

    dataset_test = Head_SSL_Infer(args.dataset_pth, mode='Test', size=args.input_size)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=2)

    image, _, _, _ = train_dataset.__getitem__(0)
    image_size = (image.shape[-2], image.shape[-1])

    model_post = Upnet_v3_MLMF_MSSR(
        size=image_size, in_channels=6528, out_channels=256,
        num_sources=mlmf_config['num_sources'], fusion_reduction=4,
        mssr_d_state=args.mssr_d_state, mssr_expand=args.mssr_expand,
        mssr_direction=args.mssr_direction
    ).cuda()
    model_post.train()

    optimizer = optim.Adam(model_post.parameters(), lr=args.lr)

    snapshot_path = f'/home/taotl/Desktop/FM-OSD/models/{args.exp}'
    os.makedirs(snapshot_path, exist_ok=True)
    writer = SummaryWriter(snapshot_path + '/log')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTExtractor(args.model_type, args.stride, device=device)

    # ── Feature cache setup ───────────────────────────────────────────────────
    feat_cache_dir = init_feat_cache(args.feat_cache_dir or None, args.model_type)
    if feat_cache_dir:
        print(f'[cache] Using precomputed features from {feat_cache_dir}')
    else:
        print('[cache] No cache — live encoding every iteration (slow)')

    shot_mem_cache = None
    if feat_cache_dir is not None:
        _dl, _np = get_cached_feat(feat_cache_dir, 'shot', str(args.id_shot), device)
        if _dl is not None:
            shot_mem_cache = (_dl, _np)
            print(f'[cache] Shot feature pre-loaded into memory')

    best_mre = float('inf')
    no_improve_count = 0   # early stop counter
    iter_num = 0
    stop_training = False

    for epoch in np.arange(0, args.max_epoch) + 1:
        if stop_training:
            break
        model_post.train()
        for images, labs, lab_smalls, image_paths in train_loader:
            iter_num += 1
            with torch.no_grad():
                desc_list = extractor.extract_multi_layer_multi_facet_descriptors(
                    images.to(device),
                    layers=mlmf_config['layers'], facets=mlmf_config['facets'],
                    bin=args.bin)
                num_patches = extractor.num_patches

            descriptors_post = model_post(desc_list, num_patches)
            optimizer.zero_grad()
            loss = heatmap_mse_loss(descriptors_post, lab_smalls)
            loss.backward()
            optimizer.step()

            writer.add_scalar('info/loss', loss, iter_num)
            print(f'iter: {iter_num}, loss: {loss.item():.6f}')

            if iter_num % args.eval_freq == 0:
                model_post.eval()
                with torch.no_grad():
                    pred_all, gt_all, imgs_all, img_names_all = find_landmark_all_mssr(
                        extractor, device, model_post, img_path_temp, dataloader_test,
                        landmarks_temp, mlmf_config, args.load_size, args.input_size,
                        args.topk, args.bin,
                        feat_cache_dir=feat_cache_dir, shot_mem_cache=shot_mem_cache)
                save_root = Path(args.save_dir) / 'dino_s_mssr'
                save_root.mkdir(exist_ok=True, parents=True)
                ev = Evaluater(pred_all, gt_all, args.eval_radius, save_root,
                               name=f'mssr_id{args.id_shot}_iter{iter_num}',
                               spacing=[0.1, 0.1], imgs=imgs_all, img_names=img_names_all)
                ev.calculate()
                ev.cal_metrics()
                writer.add_scalar('eval/mre', ev.mre, iter_num)

                if ev.mre < best_mre:
                    best_mre = ev.mre
                    no_improve_count = 0
                    torch.save(model_post.state_dict(),
                               os.path.join(snapshot_path,
                                            f'model_post_mssr_iter_{iter_num}_{best_mre:.4f}.pth'))
                    print(f'  ** best MRE: {best_mre:.4f} mm')
                else:
                    no_improve_count += 1
                    print(f'  no improve {no_improve_count}/{args.early_stop_patience}'
                          f'  (best={best_mre:.4f})')
                    if args.early_stop_patience > 0 and no_improve_count >= args.early_stop_patience:
                        print(f'[early stop] No improvement for {args.early_stop_patience} '
                              f'eval rounds. Stopping at iter {iter_num}.')
                        stop_training = True

                model_post.train()

            if iter_num % 1000 == 0:
                torch.save(model_post.state_dict(),
                           os.path.join(snapshot_path, f'model_post_mssr_iter_{iter_num}.pth'))

            if stop_training or iter_num >= args.max_iterations:
                break

        if stop_training or iter_num >= args.max_iterations:
            break

    torch.save(model_post.state_dict(),
               os.path.join(snapshot_path, 'model_post_mssr_final.pth'))
    print(f"Training completed. Best MRE: {best_mre:.4f} mm  (iter stopped at {iter_num})")
