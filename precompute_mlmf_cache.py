"""
Pre-compute MLMF+FM-OSD predictions on all 400 labeled images and cache to disk.

Uses Upnet_v3_MLMF_CoarseToFine with the best local-branch checkpoint
(model_post_fine_iter_20_1.8327.pth).

Cache format: data/tcgr_cache_mlmf/train/<id>.json  (ids 001-150)
              data/tcgr_cache_mlmf/test/<id>.json   (ids 151-400)

Each JSON: {"pred": [[y,x], ...], "gt": [[y,x], ...]}  (19 landmarks)
"""

import argparse, os, json, torch, torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from extractor_gpu import ViTExtractor
from post_net import Upnet_v3_MLMF_CoarseToFine
from datasets.head import Head_SSL_Infer


def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes','true','t','y','1'): return True
    if v.lower() in ('no','false','f','n','0'): return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def precompute_template(extractor, model_post, img_path_temp, landmarks_temp,
                        mlmf_config, load_size, original_size, bin_flag, device):
    """Compute all per-landmark template tensors once."""
    with torch.no_grad():
        img1_batch, _ = extractor.preprocess(img_path_temp, load_size)
        desc1_list = extractor.extract_multi_layer_multi_facet_descriptors(
            img1_batch.to(device),
            layers=mlmf_config['layers'], facets=mlmf_config['facets'], bin=bin_flag)
        desc1_post = model_post(desc1_list, extractor.num_patches, extractor.load_size, islocal=False)
        desc1_post_large = nn.functional.interpolate(desc1_post, original_size, mode='bilinear')

        size_y, size_x = desc1_post.shape[-2:]
        lab_feat_all, lab_feat_all_local = [], []
        desc1_post_local_all, gt_local_all = [], []

        for i in range(len(landmarks_temp)):
            ly = int(int(landmarks_temp[i][0]) / original_size[0] * size_y)
            lx = int(int(landmarks_temp[i][1]) / original_size[1] * size_x)
            lab_feat_all.append(desc1_post[0, :, ly, lx].unsqueeze(1).unsqueeze(2))

            img1_loc, _, gt_loc, _, crop_f = extractor.preprocess_local_withfeature(
                img_path_temp, load_size,
                [int(landmarks_temp[i][0]), int(landmarks_temp[i][1])],
                desc1_post_large)
            gt_local_all.append(gt_loc)

            d1_loc_list = extractor.extract_multi_layer_multi_facet_descriptors(
                img1_loc.to(device),
                layers=mlmf_config['layers'], facets=mlmf_config['facets'], bin=bin_flag)
            d1_post_loc = model_post(d1_loc_list, extractor.num_patches, extractor.load_size,
                                     islocal=True)
            d1_post_loc = (nn.functional.normalize(d1_post_loc, dim=1)
                           + nn.functional.normalize(crop_f, dim=1))
            desc1_post_local_all.append(d1_post_loc)
            lab_feat_all_local.append(
                d1_post_loc[0, :, gt_loc[0], gt_loc[1]].unsqueeze(1).unsqueeze(2))

    return (desc1_post, desc1_post_large, size_y, size_x,
            lab_feat_all, lab_feat_all_local, desc1_post_local_all, gt_local_all)


def predict_one_image(extractor, model_post, img_path, landmarks_temp,
                      mlmf_config, load_size, original_size, bin_flag, topk, device,
                      desc1_post, desc1_post_large, size_y, size_x,
                      lab_feat_all, lab_feat_all_local, desc1_post_local_all, gt_local_all):
    """Run MLMF global+local pipeline on a single query image."""
    with torch.no_grad():
        img2_batch, _ = extractor.preprocess(img_path, load_size)
        desc2_list = extractor.extract_multi_layer_multi_facet_descriptors(
            img2_batch.to(device),
            layers=mlmf_config['layers'], facets=mlmf_config['facets'], bin=bin_flag)
        desc2_post = model_post(desc2_list, extractor.num_patches, extractor.load_size, islocal=False)
        desc2_post_large = nn.functional.interpolate(desc2_post, original_size, mode='bilinear')
        size_y2, size_x2 = desc2_post.shape[-2:]

        preds = []
        for i in range(len(landmarks_temp)):
            # ---- Global matching with bidirectional verification ----
            sim = nn.CosineSimilarity(dim=0)(lab_feat_all[i], desc2_post[0])
            _, nn_k = torch.topk(sim.reshape(-1), k=topk, largest=True)

            dist_best, idx_best = 1e18, 0
            for k in range(topk):
                iy, ix = nn_k[k] // size_x2, nn_k[k] % size_x2
                sim_rev = nn.CosineSimilarity(dim=0)(
                    desc2_post[0, :, iy, ix].unsqueeze(1).unsqueeze(2), desc1_post[0])
                _, nn_rev = torch.max(sim_rev.reshape(-1), dim=-1)
                ry, rx = nn_rev // size_x, nn_rev % size_x
                x1s = rx / size_x * original_size[1]
                y1s = ry / size_y * original_size[0]
                d = (y1s - int(landmarks_temp[i][0]))**2 + (x1s - int(landmarks_temp[i][1]))**2
                if d < dist_best:
                    dist_best, idx_best = d, k

            best_flat = nn_k[idx_best].item()
            y2g = round(best_flat // size_x2 / size_y2 * original_size[0])
            x2g = round(best_flat % size_x2 / size_x2 * original_size[1])

            # ---- Local refinement ----
            img2_loc, _, _, offset2, crop_f2 = extractor.preprocess_local_withfeature(
                img_path, load_size, [int(y2g), int(x2g)], desc2_post_large)
            d2_loc_list = extractor.extract_multi_layer_multi_facet_descriptors(
                img2_loc.to(device),
                layers=mlmf_config['layers'], facets=mlmf_config['facets'], bin=bin_flag)
            d2_post_loc = model_post(d2_loc_list, extractor.num_patches, extractor.load_size,
                                     islocal=True)
            d2_post_loc = (nn.functional.normalize(d2_post_loc, dim=1)
                           + nn.functional.normalize(crop_f2, dim=1))

            sim_loc = nn.CosineSimilarity(dim=0)(lab_feat_all_local[i], d2_post_loc[0])
            h2l, w2l = sim_loc.shape
            _, nn_kl = torch.topk(sim_loc.reshape(-1), k=topk, largest=True)

            size_y1l, size_x1l = desc1_post_local_all[i].shape[-2:]
            dist_best_l, idx_best_l = 1e18, 0
            for k in range(topk):
                iyl, ixl = nn_kl[k] // w2l, nn_kl[k] % w2l
                sim_rev_l = nn.CosineSimilarity(dim=0)(
                    d2_post_loc[0, :, iyl, ixl].unsqueeze(1).unsqueeze(2),
                    desc1_post_local_all[i][0])
                _, nn_rev_l = torch.max(sim_rev_l.reshape(-1), dim=-1)
                ryl, rxl = nn_rev_l // size_x1l, nn_rev_l % size_x1l
                d = (ryl - gt_local_all[i][0])**2 + (rxl - gt_local_all[i][1])**2
                if d < dist_best_l:
                    dist_best_l, idx_best_l = d, k

            best_flat_l = nn_kl[idx_best_l].item()
            preds.append([float(int(offset2[0]) + best_flat_l // w2l),
                          float(int(offset2[1]) + best_flat_l % w2l)])

    return preds


def read_gt(dataset_pth, img_id):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_pth', default='/home/taotl/Desktop/FM-OSD/dataset/Cephalometric/')
    parser.add_argument('--input_size',  default=[2400, 1935])
    parser.add_argument('--id_shot',     default=125, type=int)
    parser.add_argument('--load_size',   default=224, type=int)
    parser.add_argument('--stride',      default=4, type=int)
    parser.add_argument('--model_type',  default='dino_vits8')
    parser.add_argument('--bin',         default='True', type=str2bool)
    parser.add_argument('--topk',        default=3, type=int)
    parser.add_argument('--mlmf_layers', default='5,8,11')
    parser.add_argument('--mlmf_facets', default='key,value')
    parser.add_argument('--ckpt',
        default='/home/taotl/Desktop/FM-OSD/models/local_mlmf/model_post_fine_iter_20_1.8327.pth')
    parser.add_argument('--cache_dir',
        default='/home/taotl/Desktop/FM-OSD/data/tcgr_cache_mlmf')
    parser.add_argument('--force', default='False', type=str2bool)
    args = parser.parse_args()

    mlmf_config = {
        'layers': [int(x) for x in args.mlmf_layers.split(',')],
        'facets': args.mlmf_facets.split(','),
        'num_sources': len(args.mlmf_layers.split(',')) * len(args.mlmf_facets.split(','))
    }
    print(f"MLMF Config: {mlmf_config}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    extractor = ViTExtractor(args.model_type, args.stride, device=device)
    image_size = (args.load_size, args.load_size)

    model_post = Upnet_v3_MLMF_CoarseToFine(
        size=image_size, in_channels=6528, out_channels=256,
        num_sources=mlmf_config['num_sources'], fusion_reduction=4
    ).to(device)

    ckpt_sd  = torch.load(args.ckpt, map_location=device)
    model_post.load_state_dict(ckpt_sd)
    model_post.eval()
    for p in model_post.parameters():
        p.requires_grad_(False)
    print(f"Loaded checkpoint from {args.ckpt}")

    # One-shot template
    one_shot = Head_SSL_Infer(args.dataset_pth, mode='Oneshot',
                               size=args.input_size, id_oneshot=args.id_shot)
    _, landmarks_temp, img_path_temp = one_shot.__getitem__(0)
    print(f"Template: {img_path_temp}")

    print("Computing template features...")
    template_tensors = precompute_template(
        extractor, model_post, img_path_temp, landmarks_temp,
        mlmf_config, args.load_size, args.input_size, args.bin, device)

    # Cache train (001-150) and test (151-400)
    splits = [
        ('train', range(1,   151),
         os.path.join(args.dataset_pth, 'RawImage', 'TrainingData')),
        ('test',  range(151, 401),
         os.path.join(args.dataset_pth, 'RawImage', 'Test1Data')),
    ]

    for split_name, id_range, img_dir in splits:
        out_dir = os.path.join(args.cache_dir, split_name)
        os.makedirs(out_dir, exist_ok=True)
        ids = [f"{i:03d}" for i in id_range]
        todo = [img_id for img_id in ids
                if args.force or not os.path.exists(os.path.join(out_dir, f"{img_id}.json"))]
        if not todo:
            print(f"{split_name}: all {len(ids)} already cached — skipping.")
            continue
        print(f"Caching {split_name}: {len(todo)}/{len(ids)} images...")
        for img_id in tqdm(todo):
            img_path = os.path.join(img_dir, img_id + '.bmp')
            if not os.path.exists(img_path):
                print(f"  WARNING: {img_path} not found, skipping.")
                continue
            preds = predict_one_image(
                extractor, model_post, img_path, landmarks_temp,
                mlmf_config, args.load_size, args.input_size, args.bin, args.topk, device,
                *template_tensors)
            gt = read_gt(args.dataset_pth, img_id)
            with open(os.path.join(out_dir, f"{img_id}.json"), 'w') as f:
                json.dump({'pred': preds, 'gt': gt}, f)
        print(f"{split_name} caching done.")

    print("All done.")
