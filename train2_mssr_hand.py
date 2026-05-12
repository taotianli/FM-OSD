# Fine-tune local branch of MLMF+MSSR coarse-to-fine network — Hand dataset
# Drop-in replacement for train2_mlmf_hand.py — only the model class changes.
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from tensorboardX import SummaryWriter
from PIL import Image

from extractor_gpu import ViTExtractor
from datasets.hand_train import HandTrainDataset
from datasets.hand import Hand_SSL_Infer
from torch.utils.data import DataLoader
from evaluation.eval import Evaluater
from post_net import Upnet_v3_MLMF_MSSR_CoarseToFine


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def _norm_lm(x):
    if hasattr(x[0], 'item'):
        return int(x[0].item()), int(x[1].item())
    return int(x[0]), int(x[1])


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


def find_landmark_all_mssr_local_hand(
        extractor, device, model_post, image_path1, dataloader, lab, mlmf_config,
        load_size=224, bin=True, template_orig_size=None, topk=3):
    """MSSR global + local; template fixed size, query uses per-image (H,W)."""
    if template_orig_size is None:
        im = Image.open(image_path1).convert('RGB')
        template_orig_size = [im.size[1], im.size[0]]
    T_H, T_W = template_orig_size[0], template_orig_size[1]

    with torch.no_grad():
        image1_batch, _ = extractor.preprocess(image_path1, load_size)
        desc1_list = extractor.extract_multi_layer_multi_facet_descriptors(
            image1_batch.to(device),
            layers=mlmf_config['layers'], facets=mlmf_config['facets'], bin=bin)
        num_p1, ls1 = extractor.num_patches, extractor.load_size
        descriptors1_post = model_post(desc1_list, num_p1, ls1, islocal=False)
        descriptors1_post_large = nn.functional.interpolate(
            descriptors1_post, template_orig_size, mode='bilinear')

        size_y, size_x = descriptors1_post.shape[-2:]
        lab_feature_all, lab_feature_all_local = [], []
        desc1_post_local_all, gt_local_all = [], []

        for i in range(len(lab)):
            ly = int(int(lab[i][0]) / T_H * size_y)
            lx = int(int(lab[i][1]) / T_W * size_x)
            lab_feature_all.append(
                descriptors1_post[0, :, ly, lx].unsqueeze(1).unsqueeze(2))

            img1_loc, _, gt_local, _, crop_feat = extractor.preprocess_local_withfeature(
                image_path1, load_size, [int(lab[i][0]), int(lab[i][1])],
                descriptors1_post_large)
            gt_local_all.append(gt_local)

            d1_loc_list = extractor.extract_multi_layer_multi_facet_descriptors(
                img1_loc.to(device),
                layers=mlmf_config['layers'], facets=mlmf_config['facets'], bin=bin)
            np1l, ls1l = extractor.num_patches, extractor.load_size
            d1_post_loc = model_post(d1_loc_list, np1l, ls1l, islocal=True)
            d1_post_loc = (nn.functional.normalize(d1_post_loc, dim=1)
                           + nn.functional.normalize(crop_feat, dim=1))
            desc1_post_local_all.append(d1_post_loc)
            lf_loc = d1_post_loc[0, :, gt_local[0], gt_local[1]].unsqueeze(1).unsqueeze(2)
            lab_feature_all_local.append(lf_loc)

        pred_all, gt_all = [], []
        for sample in tqdm(dataloader, desc="Eval-hand"):
            img_path_query, landmark_list, origin_size, _ = sample
            q_H = int(origin_size[0])
            q_W = int(origin_size[1])
            q_orig = [q_H, q_W]

            image2_batch, _ = extractor.preprocess(img_path_query, load_size)
            desc2_list = extractor.extract_multi_layer_multi_facet_descriptors(
                image2_batch.to(device),
                layers=mlmf_config['layers'], facets=mlmf_config['facets'], bin=bin)
            np2, ls2 = extractor.num_patches, extractor.load_size
            descriptors2_post = model_post(desc2_list, np2, ls2, islocal=False)
            descriptors2_post_large = nn.functional.interpolate(
                descriptors2_post, q_orig, mode='bilinear')
            size_y2, size_x2 = descriptors2_post.shape[-2:]

            points1, points2 = [], []
            for i in range(len(lab)):
                yi, xi = _norm_lm(landmark_list[i])
                points1.append([yi, xi])

                sim = nn.CosineSimilarity(dim=0)(lab_feature_all[i], descriptors2_post[0])
                h, w = sim.shape
                _, nn_k = torch.topk(sim.reshape(-1), k=topk, largest=True)

                dist_best, idx_best = 1e18, 0
                for k in range(topk):
                    iy, ix = nn_k[k] // w, nn_k[k] % w
                    sim_rev = nn.CosineSimilarity(dim=0)(
                        descriptors2_post[0, :, iy, ix].unsqueeze(1).unsqueeze(2),
                        descriptors1_post[0])
                    _, nn_rev = torch.max(sim_rev.reshape(-1), dim=-1)
                    ry, rx = nn_rev // size_x, nn_rev % size_x
                    y1s = ry / size_y * T_H
                    x1s = rx / size_x * T_W
                    d = (y1s - int(lab[i][0])) ** 2 + (x1s - int(lab[i][1])) ** 2
                    if d < dist_best:
                        dist_best, idx_best = d, k

                best_flat = nn_k[idx_best].item()
                iy2 = best_flat // size_x2
                ix2 = best_flat % size_x2
                y2g = int(round(iy2 / max(size_y2 - 1, 1) * q_H))
                x2g = int(round(ix2 / max(size_x2 - 1, 1) * q_W))

                img2_loc, _, _, offset2, crop_feat2 = extractor.preprocess_local_withfeature(
                    img_path_query, load_size, [y2g, x2g], descriptors2_post_large)
                d2_loc_list = extractor.extract_multi_layer_multi_facet_descriptors(
                    img2_loc.to(device),
                    layers=mlmf_config['layers'], facets=mlmf_config['facets'], bin=bin)
                np2l, ls2l = extractor.num_patches, extractor.load_size
                d2_post_loc = model_post(d2_loc_list, np2l, ls2l, islocal=True)
                d2_post_loc = (nn.functional.normalize(d2_post_loc, dim=1)
                               + nn.functional.normalize(crop_feat2, dim=1))

                sim_loc = nn.CosineSimilarity(dim=0)(
                    lab_feature_all_local[i], d2_post_loc[0])
                h2l, w2l = sim_loc.shape
                _, nn_kl = torch.topk(sim_loc.reshape(-1), k=topk, largest=True)

                dist_best_l, idx_best_l = 1e18, 0
                sy1l, sx1l = desc1_post_local_all[i].shape[-2:]
                for k in range(topk):
                    iyl, ixl = nn_kl[k] // w2l, nn_kl[k] % w2l
                    sim_rev_l = nn.CosineSimilarity(dim=0)(
                        d2_post_loc[0, :, iyl, ixl].unsqueeze(1).unsqueeze(2),
                        desc1_post_local_all[i][0])
                    _, nn_rvl = torch.max(sim_rev_l.reshape(-1), dim=-1)
                    ryl, rxl = nn_rvl // sx1l, nn_rvl % sx1l
                    d = (ryl - gt_local_all[i][0]) ** 2 + (rxl - gt_local_all[i][1]) ** 2
                    if d < dist_best_l:
                        dist_best_l, idx_best_l = d, k

                bf_l = nn_kl[idx_best_l].item()
                y2l, x2l = bf_l // w2l, bf_l % w2l
                points2.append([int(offset2[0]) + y2l, int(offset2[1]) + x2l])

            pred_all.append(points2)
            gt_all.append(points1)

    return pred_all, gt_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='/home/u6da/taotl.u6da/FM-OSD/output')
    parser.add_argument('--load_size', default=224, type=int)
    parser.add_argument('--stride', default=4, type=int)
    parser.add_argument('--model_type', default='dino_vits8')
    parser.add_argument('--bin', default='True', type=str2bool)
    parser.add_argument('--topk', default=3, type=int)

    parser.add_argument('--dataset_pth', default='/home/u6da/taotl.u6da/FM-OSD/data/Hand/hand/')
    parser.add_argument('--input_size', default=[2048, 2048])
    parser.add_argument('--id_shot', default=0, type=int)
    parser.add_argument('--eval_radius', default=[2, 2.5, 3, 4, 6, 8])

    parser.add_argument('--bs', default=4, type=int)
    parser.add_argument('--random_range', default=50, type=int)
    parser.add_argument('--local_coe', default=1.0, type=float)
    parser.add_argument('--max_epoch', default=300, type=int)
    parser.add_argument('--max_iterations', type=int, default=100)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--exp', default='local_mssr_hand')

    parser.add_argument('--mlmf_layers', default='5,8,11')
    parser.add_argument('--mlmf_facets', default='key,value')
    parser.add_argument('--mssr_d_state', default=8, type=int,
                        help='SSM state dim. Default 8 is 2x faster than 16.')
    parser.add_argument('--mssr_expand', default=1, type=int,
                        help='Channel expansion. Default 1 avoids doubling width.')
    parser.add_argument('--auto_input_size', default='True', type=str2bool)
    parser.add_argument('--global_ckpt', type=str, default='',
                        help='Path to train1_mssr_hand best checkpoint')

    args = parser.parse_args()

    mlmf_config = {
        'layers': [int(x) for x in args.mlmf_layers.split(',')],
        'facets': args.mlmf_facets.split(','),
        'num_sources': len(args.mlmf_layers.split(',')) * len(args.mlmf_facets.split(','))
    }

    random.seed(2022)
    np.random.seed(2022)
    torch.manual_seed(2022)
    torch.cuda.manual_seed(2022)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    landmarks_temp = [_norm_lm(t) for t in landmarks_temp]

    dataset_test = Hand_SSL_Infer(args.dataset_pth, mode='Test')

    def _coll(x):
        return x[0]

    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False,
                                 num_workers=2, collate_fn=_coll)

    image, _, _, _ = train_dataset.__getitem__(0)
    image_size = (image.shape[-2], image.shape[-1])

    model_post = Upnet_v3_MLMF_MSSR_CoarseToFine(
        size=image_size, in_channels=6528, out_channels=256,
        num_sources=mlmf_config['num_sources'], fusion_reduction=4,
        mssr_d_state=args.mssr_d_state, mssr_expand=args.mssr_expand,
        mssr_direction='forward',
    ).to(device)

    # Load global checkpoint — remap Upnet_v3_MLMF_MSSR keys to CoarseToFine keys
    if os.path.exists(args.global_ckpt):
        global_sd = torch.load(args.global_ckpt, map_location=device)
        model_sd = model_post.state_dict()
        remap = {}
        for k, v in global_sd.items():
            if k.startswith('fusion.'):
                remap['fusion_global.' + k[len('fusion.'):]] = v
            elif k.startswith('conv_out.'):
                remap['conv_out1.' + k[len('conv_out.'):]] = v
            elif k.startswith('mssr.'):
                remap[k] = v
            elif k in model_sd:
                remap[k] = v
        matched = {k: v for k, v in remap.items() if k in model_sd}
        model_sd.update(matched)
        model_post.load_state_dict(model_sd)
        print(f"Loaded global weights: {len(matched)} keys from {args.global_ckpt}")
    else:
        print(f"WARNING: global_ckpt missing ({args.global_ckpt}). Train global first.")

    # Freeze global branch (fusion_global, mssr, conv_out1); train local branch only
    for name, param in model_post.named_parameters():
        if any(n in name for n in ('fusion_global', 'mssr', 'conv_out1')):
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model_post.parameters()), lr=args.lr)

    snapshot_path = f'/home/u6da/taotl.u6da/FM-OSD/models/{args.exp}'
    os.makedirs(snapshot_path, exist_ok=True)
    writer = SummaryWriter(snapshot_path + '/log')
    extractor = ViTExtractor(args.model_type, args.stride, device=device)

    tpl = Image.open(img_path_temp).convert('RGB')
    template_orig_size = [tpl.size[1], tpl.size[0]]

    eval_dir = Path(args.save_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    best_mre = float('inf')
    iter_num = 0

    # Initial evaluation
    model_post.eval()
    pred_all, gt_all = find_landmark_all_mssr_local_hand(
        extractor, device, model_post, img_path_temp, dataloader_test,
        landmarks_temp, mlmf_config, args.load_size, args.bin,
        template_orig_size=template_orig_size, topk=args.topk)
    ev = Evaluater(pred_all, gt_all, args.eval_radius, eval_dir,
                   name='mssr_hand_local_init', spacing=[0.1, 0.1])
    ev.calculate()
    ev.cal_metrics()
    best_mre = ev.mre
    torch.save(model_post.state_dict(),
               os.path.join(snapshot_path, f'model_post_fine_iter_0_{best_mre:.4f}.pth'))
    model_post.train()

    print(f"Starting Hand local fine-tuning ({args.max_iterations} iters)...")

    for epoch in range(args.max_epoch):
        model_post.train()
        for images, labs, lab_smalls, image_paths in train_loader:
            iter_num += 1
            optimizer.zero_grad()

            for j in range(labs.shape[1]):
                image_local_list, lab_local_list = [], []
                for i in range(images.shape[0]):
                    img_local, _, gt_local, _ = extractor.preprocess_local_random_new(
                        image_paths[i], args.load_size,
                        [int(labs[i, j, 0]), int(labs[i, j, 1])],
                        args.random_range)
                    image_local_list.append(img_local)
                    lab_local_list.append(gt_local)
                image_local = torch.cat(image_local_list, dim=0)
                lab_local = torch.tensor(lab_local_list).unsqueeze(1)

                with torch.no_grad():
                    desc_local_list = extractor.extract_multi_layer_multi_facet_descriptors(
                        image_local.to(device),
                        layers=mlmf_config['layers'], facets=mlmf_config['facets'],
                        bin=args.bin)
                    np_l, ls_l = extractor.num_patches, extractor.load_size

                desc_post_local = model_post(desc_local_list, np_l, ls_l, islocal=True)
                loss_local = (args.local_coe * heatmap_mse_loss(
                    desc_post_local, lab_local, var=2.0) / labs.shape[1])
                loss_local.backward(retain_graph=True)
                writer.add_scalar(f'loss/local_{j}', loss_local.item(), iter_num)

            optimizer.step()
            print(f"[Iter {iter_num}] loss_local: {loss_local.item():.6f}")

            if iter_num % 20 == 0:
                model_post.eval()
                pred_all, gt_all = find_landmark_all_mssr_local_hand(
                    extractor, device, model_post, img_path_temp, dataloader_test,
                    landmarks_temp, mlmf_config, args.load_size, args.bin,
                    template_orig_size=template_orig_size, topk=args.topk)
                ev = Evaluater(pred_all, gt_all, args.eval_radius, eval_dir,
                               name=f'mssr_hand_local_{iter_num}', spacing=[0.1, 0.1])
                ev.calculate()
                ev.cal_metrics()
                if ev.mre < best_mre:
                    best_mre = ev.mre
                    ckpt = os.path.join(
                        snapshot_path, f'model_post_fine_iter_{iter_num}_{best_mre:.4f}.pth')
                    torch.save(model_post.state_dict(), ckpt)
                    print(f"  ** best MRE: {best_mre:.4f} → {ckpt}")
                model_post.train()

            if iter_num >= args.max_iterations:
                break
        if iter_num >= args.max_iterations:
            break

    torch.save(model_post.state_dict(), os.path.join(snapshot_path, 'model_post_final.pth'))
    print(f"Done. Best MRE: {best_mre:.4f}")
