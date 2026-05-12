"""
Pre-compute MLMF+FM-OSD predictions on Hand X-ray dataset (909 images, 37 landmarks).
Output: data/tcgr_cache_hand/{train,test}/<id>.json

JSON format: {"pred": [[y,x]×37], "gt": [[y,x]×37]}
"""
import argparse, os, json, torch, torch.nn as nn
import numpy as np, pandas as pd
from tqdm import tqdm
from extractor_gpu import ViTExtractor
from post_net import Upnet_v3_MLMF_CoarseToFine


HAND_PATH = '/home/u6da/taotl.u6da/FM-OSD/data/Hand/hand'


def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ('yes','true','t','y','1'): return True
    if v.lower() in ('no','false','f','n','0'): return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def load_hand_split(hand_path, mode='Train', id_shot=0):
    """Return list of (img_path, gt_37×2) for the requested split."""
    img_dir = os.path.join(hand_path, 'jpg')
    labels  = pd.read_csv(os.path.join(hand_path, 'all.csv'),
                          header=None, index_col=0)
    index_set = set(labels.index)
    files = [i[:-4] for i in sorted(os.listdir(img_dir)) if i.endswith('.jpg')]
    files = [f for f in files if int(f) in index_set]
    n = len(files)
    train_num, val_num = 550, 59
    if mode == 'Train':
        split_files = files[:train_num]
    elif mode == 'Val':
        split_files = files[train_num:train_num + val_num]
    elif mode == 'Test':
        split_files = files[train_num + val_num:]
    elif mode == 'Oneshot':
        split_files = files[id_shot:id_shot + 1]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    records = []
    for fid in split_files:
        img_path = os.path.join(img_dir, fid + '.jpg')
        row = labels.loc[int(fid)].values.tolist()
        # CSV: x1,y1,x2,y2,... → convert to [y,x] pairs
        gt = [[int(row[2*i+1]), int(row[2*i])] for i in range(len(row)//2)]
        records.append((fid, img_path, gt))
    return records


def precompute_template(extractor, model_post, img_path_temp, landmarks_temp,
                        mlmf_config, load_size, bin_flag, device):
    """Compute all per-landmark template tensors once."""
    from PIL import Image as PILImage
    img = PILImage.open(img_path_temp).convert('L')
    original_size = [img.size[1], img.size[0]]  # [H, W]

    with torch.no_grad():
        img1_batch, _ = extractor.preprocess(img_path_temp, load_size)
        desc1_list = extractor.extract_multi_layer_multi_facet_descriptors(
            img1_batch.to(device),
            layers=mlmf_config['layers'], facets=mlmf_config['facets'], bin=bin_flag)
        desc1_post = model_post(desc1_list, extractor.num_patches, extractor.load_size,
                                islocal=False)
        desc1_post_large = nn.functional.interpolate(desc1_post, original_size, mode='bilinear')
        size_y, size_x = desc1_post.shape[-2:]

        lab_feat_all, lab_feat_all_local = [], []
        desc1_post_local_all, gt_local_all = [], []

        for lm in landmarks_temp:
            ly = int(int(lm[0]) / original_size[0] * size_y)
            lx = int(int(lm[1]) / original_size[1] * size_x)
            lab_feat_all.append(desc1_post[0, :, ly, lx].unsqueeze(1).unsqueeze(2))

            img1_loc, _, gt_loc, _, crop_f = extractor.preprocess_local_withfeature(
                img_path_temp, load_size, [int(lm[0]), int(lm[1])], desc1_post_large)
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

    return (original_size, desc1_post, desc1_post_large, size_y, size_x,
            lab_feat_all, lab_feat_all_local, desc1_post_local_all, gt_local_all)


def predict_one_image(extractor, model_post, img_path, landmarks_temp,
                      mlmf_config, load_size, bin_flag, topk, device,
                      original_size_temp,
                      desc1_post, desc1_post_large, size_y, size_x,
                      lab_feat_all, lab_feat_all_local,
                      desc1_post_local_all, gt_local_all):
    """MLMF global+local matching for one query Hand image."""
    from PIL import Image as PILImage
    img = PILImage.open(img_path).convert('L')
    original_size = [img.size[1], img.size[0]]  # [H, W]

    with torch.no_grad():
        img2_batch, _ = extractor.preprocess(img_path, load_size)
        desc2_list = extractor.extract_multi_layer_multi_facet_descriptors(
            img2_batch.to(device),
            layers=mlmf_config['layers'], facets=mlmf_config['facets'], bin=bin_flag)
        desc2_post = model_post(desc2_list, extractor.num_patches, extractor.load_size,
                                islocal=False)
        desc2_post_large = nn.functional.interpolate(desc2_post, original_size, mode='bilinear')
        size_y2, size_x2 = desc2_post.shape[-2:]

        preds = []
        for i, lm in enumerate(landmarks_temp):
            # ---- Global matching ----
            sim = nn.CosineSimilarity(dim=0)(lab_feat_all[i], desc2_post[0])
            _, nn_k = torch.topk(sim.reshape(-1), k=topk, largest=True)
            dist_best, idx_best = 1e18, 0
            for k in range(topk):
                iy, ix = nn_k[k] // size_x2, nn_k[k] % size_x2
                sim_rev = nn.CosineSimilarity(dim=0)(
                    desc2_post[0, :, iy, ix].unsqueeze(1).unsqueeze(2), desc1_post[0])
                _, nn_rev = torch.max(sim_rev.reshape(-1), dim=-1)
                ry, rx = nn_rev // size_x, nn_rev % size_x
                x1s = rx / size_x * original_size_temp[1]
                y1s = ry / size_y * original_size_temp[0]
                d = (y1s - int(lm[0]))**2 + (x1s - int(lm[1]))**2
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hand_path', default=HAND_PATH)
    parser.add_argument('--load_size', default=224, type=int)
    parser.add_argument('--stride',    default=4,   type=int)
    parser.add_argument('--model_type', default='dino_vits8')
    parser.add_argument('--bin',    default='True', type=str2bool)
    parser.add_argument('--topk',   default=3,   type=int)
    parser.add_argument('--mlmf_layers', default='5,8,11')
    parser.add_argument('--mlmf_facets', default='key,value')
    parser.add_argument('--ckpt',
        default='/home/u6da/taotl.u6da/FM-OSD/models/local_mlmf_hand/model_post_final.pth',
        help='Hand-trained Upnet_v3_MLMF_CoarseToFine (after train2_mlmf_hand.py)')
    parser.add_argument('--cache_dir',
        default='/projects/u6da/fmosd_cache/tcgr_cache_hand')
    parser.add_argument('--oneshot_idx', default=0, type=int,
        help='Index of template image in training split')
    parser.add_argument('--force', default='False', type=str2bool)
    args = parser.parse_args()

    mlmf_config = {
        'layers': [int(x) for x in args.mlmf_layers.split(',')],
        'facets': args.mlmf_facets.split(','),
        'num_sources': len(args.mlmf_layers.split(',')) * len(args.mlmf_facets.split(','))
    }
    print(f"MLMF Config: {mlmf_config}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTExtractor(args.model_type, args.stride, device=device)

    model_post = Upnet_v3_MLMF_CoarseToFine(
        size=(args.load_size, args.load_size),
        in_channels=6528, out_channels=256,
        num_sources=mlmf_config['num_sources'], fusion_reduction=4
    ).to(device)
    ckpt_sd = torch.load(args.ckpt, map_location=device)
    model_post.load_state_dict(ckpt_sd)
    model_post.eval()
    for p in model_post.parameters():
        p.requires_grad_(False)
    print(f"Loaded checkpoint from {args.ckpt}")

    # One-shot template: first training image
    train_recs = load_hand_split(args.hand_path, 'Train')
    tmpl_id, tmpl_path, tmpl_gt = train_recs[args.oneshot_idx]
    print(f"Template: {tmpl_path}  ({len(tmpl_gt)} landmarks)")

    print("Computing template features...")
    template_tensors = precompute_template(
        extractor, model_post, tmpl_path, tmpl_gt,
        mlmf_config, args.load_size, args.bin, device)
    original_size_temp = template_tensors[0]

    # Process train (excluding template) + val + test
    splits = [
        ('train', [r for r in train_recs if r[0] != tmpl_id]),
        ('test',  load_hand_split(args.hand_path, 'Test')),
    ]

    for split_name, records in splits:
        out_dir = os.path.join(args.cache_dir, split_name)
        os.makedirs(out_dir, exist_ok=True)
        todo = [r for r in records
                if args.force or not os.path.exists(
                    os.path.join(out_dir, f"{r[0]}.json"))]
        if not todo:
            print(f"{split_name}: all {len(records)} already cached.")
            continue
        print(f"Caching {split_name}: {len(todo)}/{len(records)} images...")
        for fid, img_path, gt in tqdm(todo):
            preds = predict_one_image(
                extractor, model_post, img_path, tmpl_gt,
                mlmf_config, args.load_size, args.bin, args.topk, device,
                original_size_temp, *template_tensors[1:])
            with open(os.path.join(out_dir, f"{fid}.json"), 'w') as f:
                json.dump({'pred': preds, 'gt': gt}, f)
        print(f"{split_name} caching done.")

    print("Hand cache complete.")
