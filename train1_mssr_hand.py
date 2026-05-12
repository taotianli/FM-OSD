# Train global branch with MLMF + MSSR — Hand dataset (37 landmarks, variable resolution)
# Drop-in replacement for train1_mlmf_hand.py — only the model class changes.
import argparse
import torch
from pathlib import Path
from extractor_gpu import ViTExtractor
import numpy as np
from PIL import Image

from datasets.hand_train import HandTrainDataset
from datasets.hand import Hand_SSL_Infer
from torch.utils.data import DataLoader
from evaluation.eval import Evaluater
from post_net import Upnet_v3_MLMF_MSSR
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import os
import random


def init_feat_cache_mlmf(feat_cache_dir, model_type):
    if not feat_cache_dir:
        return None
    cache_dir = Path(feat_cache_dir) / model_type
    if not cache_dir.exists():
        print(f'[cache] Warning: {cache_dir} not found, falling back to live encode.')
        return None
    return cache_dir


def get_cached_feat_mlmf(cache_dir, prefix, stem, device):
    cache_f = cache_dir / f'{prefix}_{stem}.pt'
    if not cache_f.exists():
        return None, None
    cached = torch.load(cache_f, map_location=device)
    desc_list = [d.float() for d in cached['desc_list']]
    return desc_list, cached['num_patches']


def _norm_lab_entry(t):
    if hasattr(t[0], 'item'):
        return int(t[0].item()), int(t[1].item())
    return int(t[0]), int(t[1])


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
        n_lm = landmarks[i].shape[0]
        for j in range(n_lm):
            gt = feature_temp[:, landmarks[i, j, 0], landmarks[i, j, 1]].unsqueeze(1).unsqueeze(2)
            similarity = nn.CosineSimilarity(dim=0)(gt, feature_temp).unsqueeze(0)
            pred_temp.append(similarity)
        pred.append(torch.cat(pred_temp, dim=0).unsqueeze(0))
    pred = torch.cat(pred, dim=0)
    return criterion(pred, label)


def find_landmark_all_mssr_hand(
        extractor, device, model_post, image_path1, dataloader, lab,
        mlmf_config, load_size=224, bin=True,
        template_orig_size=None, topk=5,
        feat_cache_dir=None, shot_mem_cache=None):
    """Global matching only; per-query original_size from Hand loader."""
    if template_orig_size is None:
        im = Image.open(image_path1).convert('RGB')
        template_orig_size = [im.size[1], im.size[0]]

    if shot_mem_cache is not None:
        descriptors1_list, num_patches1 = shot_mem_cache
    elif feat_cache_dir is not None:
        stem = Path(image_path1).stem
        descriptors1_list, num_patches1 = get_cached_feat_mlmf(
            feat_cache_dir, 'shot', stem, device)
        if descriptors1_list is None:
            image1_batch, _ = extractor.preprocess(image_path1, load_size)
            descriptors1_list = extractor.extract_multi_layer_multi_facet_descriptors(
                image1_batch.to(device),
                layers=mlmf_config['layers'], facets=mlmf_config['facets'], bin=bin)
            num_patches1 = extractor.num_patches
    else:
        image1_batch, _ = extractor.preprocess(image_path1, load_size)
        descriptors1_list = extractor.extract_multi_layer_multi_facet_descriptors(
            image1_batch.to(device),
            layers=mlmf_config['layers'], facets=mlmf_config['facets'], bin=bin)
        num_patches1 = extractor.num_patches

    descriptors1_post = model_post(descriptors1_list, num_patches1)

    lab_feature_all = []
    for i in range(len(lab)):
        ly = int(lab[i][0] / template_orig_size[0] * descriptors1_post.shape[-2])
        lx = int(lab[i][1] / template_orig_size[1] * descriptors1_post.shape[-1])
        lab_feature_all.append(
            descriptors1_post[0, :, ly, lx].unsqueeze(1).unsqueeze(2))

    pred_all, gt_all, imgs_all, img_names_all = [], [], [], []

    for sample in dataloader:
        img_path_query, landmark_list, origin_size, _ = sample
        original_size_q = [int(origin_size[0]), int(origin_size[1])]

        if feat_cache_dir is not None:
            stem2 = Path(img_path_query).stem
            descriptors2_list, num_patches2 = get_cached_feat_mlmf(
                feat_cache_dir, 'test', stem2, device)
            if descriptors2_list is None:
                image2_batch, _ = extractor.preprocess(img_path_query, load_size)
                descriptors2_list = extractor.extract_multi_layer_multi_facet_descriptors(
                    image2_batch.to(device),
                    layers=mlmf_config['layers'], facets=mlmf_config['facets'], bin=bin)
                num_patches2 = extractor.num_patches
        else:
            image2_batch, _ = extractor.preprocess(img_path_query, load_size)
            descriptors2_list = extractor.extract_multi_layer_multi_facet_descriptors(
                image2_batch.to(device),
                layers=mlmf_config['layers'], facets=mlmf_config['facets'], bin=bin)
            num_patches2 = extractor.num_patches
        descriptors2_post = model_post(descriptors2_list, num_patches2)

        points1, points2 = [], []
        imgs_all.append(None)
        img_names_all.append([img_path_query])

        for i in range(len(lab)):
            yi, xi = _norm_lab_entry(landmark_list[i])
            points1.append([yi, xi])

            sim = nn.CosineSimilarity(dim=0)(lab_feature_all[i], descriptors2_post[0])
            h2, w2 = sim.shape
            _, nn_k = torch.topk(sim.reshape(-1), k=topk, largest=True)

            dist_best, idx_best = 1e18, 0
            sz_y, sz_x = descriptors1_post.shape[-2:]
            for k in range(topk):
                iy, ix = nn_k[k] // w2, nn_k[k] % w2
                sim_rev = nn.CosineSimilarity(dim=0)(
                    descriptors2_post[0, :, iy, ix].unsqueeze(1).unsqueeze(2),
                    descriptors1_post[0])
                _, nn_1 = torch.max(sim_rev.reshape(-1), dim=-1)
                y1s = (nn_1 // sz_x) / sz_y * template_orig_size[0]
                x1s = (nn_1 % sz_x) / sz_x * template_orig_size[1]
                d = (y1s - lab[i][0]) ** 2 + (x1s - lab[i][1]) ** 2
                if d < dist_best:
                    dist_best, idx_best = d, k

            best = nn_k[idx_best].cpu().item()
            sz_y2, sz_x2 = descriptors2_post.shape[-2:]
            y2 = np.round(best // sz_x2 / sz_y2 * original_size_q[0])
            x2 = np.round(best % sz_x2 / sz_x2 * original_size_q[1])
            points2.append([float(y2), float(x2)])

        pred_all.append(points2)
        gt_all.append(points1)

    return pred_all, gt_all, imgs_all, img_names_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train global branch MLMF+MSSR on Hand.')
    parser.add_argument('--save_dir', type=str, default='/home/u6da/taotl.u6da/FM-OSD/output')
    parser.add_argument('--load_size', default=224, type=int)
    parser.add_argument('--stride', default=4, type=int)
    parser.add_argument('--model_type', default='dino_vits8', type=str)
    parser.add_argument('--bin', default='True', type=str2bool)
    parser.add_argument('--topk', default=5, type=int)

    parser.add_argument('--dataset_pth', type=str,
                        default='/home/u6da/taotl.u6da/FM-OSD/data/Hand/hand/')
    parser.add_argument('--input_size', default=[2048, 2048])
    parser.add_argument('--id_shot', default=0, type=int)
    parser.add_argument('--eval_radius', default=[2, 2.5, 3, 4, 6, 8])

    parser.add_argument('--bs', default=4, type=int)
    parser.add_argument('--max_epoch', default=300, type=int)
    parser.add_argument('--max_iterations', type=int, default=8000)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--exp', default='global_mssr_hand')

    parser.add_argument('--mlmf_layers', default='5,8,11', type=str)
    parser.add_argument('--mlmf_facets', default='key,value', type=str)
    parser.add_argument('--mssr_d_state', default=8, type=int,
                        help='SSM state dim. Default 8 is 2x faster than 16.')
    parser.add_argument('--mssr_expand', default=1, type=int,
                        help='Channel expansion. Default 1 avoids doubling width.')
    parser.add_argument('--auto_input_size', default='True', type=str2bool)
    parser.add_argument('--feat_cache_dir', type=str, default='')

    args = parser.parse_args()

    mlmf_config = {
        'layers': [int(x) for x in args.mlmf_layers.split(',')],
        'facets': args.mlmf_facets.split(','),
        'num_sources': len(args.mlmf_layers.split(',')) * len(args.mlmf_facets.split(','))
    }
    print(f"MLMF+MSSR Config: {mlmf_config}, d_state={args.mssr_d_state}")

    random.seed(2022)
    np.random.seed(2022)
    torch.manual_seed(2022)
    torch.cuda.manual_seed(2022)

    train_dataset = HandTrainDataset(istrain=0, original_size=args.input_size,
                                     load_size=args.load_size)
    if args.auto_input_size and len(train_dataset.data) > 0:
        im = Image.open(train_dataset.data[0][0])
        args.input_size = [im.size[1], im.size[0]]
        train_dataset.original_size = args.input_size
        print(f"auto_input_size: {args.input_size}")

    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=2)

    one_shot = Hand_SSL_Infer(args.dataset_pth, mode='Oneshot', id_oneshot=args.id_shot)
    img_path_temp, landmarks_temp, _, _ = one_shot.__getitem__(0)
    landmarks_temp = [_norm_lab_entry(t) for t in landmarks_temp]

    dataset_test = Hand_SSL_Infer(args.dataset_pth, mode='Test')

    def _collate_infer(x):
        return x[0]

    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False,
                                 num_workers=2, collate_fn=_collate_infer)

    image, _, _, _ = train_dataset.__getitem__(0)
    image_size = (image.shape[-2], image.shape[-1])

    model_post = Upnet_v3_MLMF_MSSR(
        size=image_size, in_channels=6528, out_channels=256,
        num_sources=mlmf_config['num_sources'], fusion_reduction=4,
        mssr_d_state=args.mssr_d_state, mssr_expand=args.mssr_expand,
        mssr_direction='forward',
    ).cuda()
    model_post.train()

    optimizer = optim.Adam(model_post.parameters(), lr=args.lr)

    snapshot_path = f'/home/u6da/taotl.u6da/FM-OSD/models/{args.exp}'
    os.makedirs(snapshot_path, exist_ok=True)
    writer = SummaryWriter(snapshot_path + '/log')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTExtractor(args.model_type, args.stride, device=device)

    tpl_im = Image.open(img_path_temp).convert('RGB')
    template_orig_size = [tpl_im.size[1], tpl_im.size[0]]

    feat_cache_dir = init_feat_cache_mlmf(args.feat_cache_dir or None, args.model_type)
    shot_mem_cache = None
    if feat_cache_dir is not None:
        shot_stem = Path(img_path_temp).stem
        _dl, _np = get_cached_feat_mlmf(feat_cache_dir, 'shot', shot_stem, device)
        if _dl is not None:
            shot_mem_cache = (_dl, _np)
            print(f'[cache] Shot feature pre-loaded from {feat_cache_dir}')

    best_mre = float('inf')
    iter_num = 0

    for epoch in np.arange(0, args.max_epoch) + 1:
        model_post.train()
        for images, labs, lab_smalls, image_paths in train_loader:
            iter_num += 1
            with torch.no_grad():
                loaded_from_cache = False
                if feat_cache_dir is not None:
                    desc_batch, num_patches = [], None
                    for img_path in image_paths:
                        stem = Path(img_path).stem
                        dl, np_ = get_cached_feat_mlmf(feat_cache_dir, 'train', stem, device)
                        if dl is None:
                            desc_batch = None
                            break
                        desc_batch.append(dl)
                        num_patches = np_
                    if desc_batch is not None:
                        n_src = len(desc_batch[0])
                        descriptors_list = [
                            torch.cat([desc_batch[b][s] for b in range(len(desc_batch))], dim=0)
                            for s in range(n_src)
                        ]
                        loaded_from_cache = True

                if not loaded_from_cache:
                    descriptors_list = extractor.extract_multi_layer_multi_facet_descriptors(
                        images.to(device),
                        layers=mlmf_config['layers'], facets=mlmf_config['facets'], bin=args.bin)
                    num_patches = extractor.num_patches

            descriptors_post = model_post(descriptors_list, num_patches)
            optimizer.zero_grad()
            loss = heatmap_mse_loss(descriptors_post, lab_smalls)
            loss.backward()
            optimizer.step()

            writer.add_scalar('info/loss', loss, iter_num)
            print(f'iter: {iter_num}, loss: {loss.item():.6f}')

            if iter_num % 500 == 0:
                model_post.eval()
                with torch.no_grad():
                    pred_all, gt_all, imgs_all, img_names_all = find_landmark_all_mssr_hand(
                        extractor, device, model_post, img_path_temp,
                        dataloader_test, landmarks_temp, mlmf_config,
                        args.load_size, args.bin,
                        template_orig_size=template_orig_size, topk=args.topk,
                        feat_cache_dir=feat_cache_dir, shot_mem_cache=shot_mem_cache)
                save_root = Path(args.save_dir) / 'dino_s_mssr_hand'
                save_root.mkdir(exist_ok=True, parents=True)
                ev = Evaluater(pred_all, gt_all, args.eval_radius, save_root,
                               name=f'mssr_hand_id{args.id_shot}_iter{iter_num}',
                               spacing=[0.1, 0.1], imgs=imgs_all, img_names=img_names_all)
                ev.calculate()
                ev.cal_metrics()
                if ev.mre < best_mre:
                    best_mre = ev.mre
                    torch.save(model_post.state_dict(),
                               os.path.join(snapshot_path,
                                            f'model_post_mssr_iter_{iter_num}_{best_mre:.4f}.pth'))
                model_post.train()

            if iter_num % 1000 == 0:
                torch.save(model_post.state_dict(),
                           os.path.join(snapshot_path, f'model_post_mssr_iter_{iter_num}.pth'))

            if iter_num >= args.max_iterations:
                break
        if iter_num >= args.max_iterations:
            break

    print(f"Training completed. Best MRE: {best_mre:.4f}")
