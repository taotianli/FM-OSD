# Fine-tune the local branch of Upnet_v3_MLMF_CoarseToFine
# Adapts train2.py to use MLMF multi-layer multi-facet features
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

from extractor_gpu import ViTExtractor
from datasets.head_train import TrainDataset
from datasets.head import Head_SSL_Infer
from torch.utils.data import DataLoader
from evaluation.eval import Evaluater
from post_net import Upnet_v3_MLMF_CoarseToFine


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def make_heatmap(landmark, size, var=5.0):
    length, width = size
    x, y = torch.meshgrid(torch.arange(0, length), torch.arange(0, width))
    p = torch.stack([x, y], dim=2).float()
    inner_factor = -1 / (2 * (var ** 2))
    mean = torch.as_tensor(landmark).float()
    heatmap = (p - mean).pow(2).sum(dim=-1)
    heatmap = torch.exp(heatmap * inner_factor)
    return heatmap


def heatmap_mse_loss(features, landmarks, var=5.0, criterion=nn.MSELoss()):
    lab = []
    for i in range(len(landmarks)):
        labels = landmarks[i]
        labtemp = []
        for l in range(labels.shape[0]):
            labtemp.append(make_heatmap(
                labels[l], [features.shape[-2], features.shape[-1]], var=var))
        lab.append(torch.stack(labtemp, dim=0))
    label = torch.stack(lab, dim=0).to(features.device)

    pred = []
    for i in range(len(landmarks)):
        feature_temp = features[i]
        pred_temp = []
        for j in range(labels.shape[0]):
            gt = feature_temp[:, landmarks[i, j, 0], landmarks[i, j, 1]].unsqueeze(1).unsqueeze(2)
            similarity = torch.nn.CosineSimilarity(dim=0)(gt, feature_temp).unsqueeze(0)
            pred_temp.append(similarity)
        pred_temp = torch.cat(pred_temp, dim=0).unsqueeze(0)
        pred.append(pred_temp)
    pred = torch.cat(pred, dim=0)
    return criterion(pred, label)


def find_landmark_all_mlmf_local(extractor, device, model_post, image_path1,
                                  dataloader, lab, mlmf_config,
                                  load_size=224, layer=8, facet='key',
                                  bin=True, original_size=[2400, 1935], topk=3):
    """Full MLMF evaluation: global matching + local refinement pipeline."""
    with torch.no_grad():
        # ---- Template processing ----
        image1_batch, _ = extractor.preprocess(image_path1, load_size)
        desc1_list = extractor.extract_multi_layer_multi_facet_descriptors(
            image1_batch.to(device),
            layers=mlmf_config['layers'], facets=mlmf_config['facets'], bin=bin)
        num_patches1, load_size1 = extractor.num_patches, extractor.load_size
        descriptors1_post = model_post(desc1_list, num_patches1, load_size1, islocal=False)
        descriptors1_post_large = torch.nn.functional.interpolate(
            descriptors1_post, original_size, mode='bilinear')

        size_y, size_x = descriptors1_post.shape[-2:]
        lab_feature_all        = []
        lab_feature_all_local  = []
        desc1_post_local_all   = []
        gt_local_all           = []

        for i in range(len(lab)):
            ly = int(int(lab[i][0]) / original_size[0] * size_y)
            lx = int(int(lab[i][1]) / original_size[1] * size_x)
            lf = descriptors1_post[0, :, ly, lx].unsqueeze(1).unsqueeze(2)
            lab_feature_all.append(lf)

            # Template local crop
            img1_local, _, gt_local, offset, crop_feat = \
                extractor.preprocess_local_withfeature(
                    image_path1, load_size,
                    [int(lab[i][0]), int(lab[i][1])],
                    descriptors1_post_large)
            gt_local_all.append(gt_local)

            desc1_loc_list = extractor.extract_multi_layer_multi_facet_descriptors(
                img1_local.to(device),
                layers=mlmf_config['layers'], facets=mlmf_config['facets'], bin=bin)
            num_patches1_loc, load_size1_loc = extractor.num_patches, extractor.load_size
            desc1_post_loc = model_post(desc1_loc_list, num_patches1_loc,
                                        load_size1_loc, islocal=True)
            desc1_post_loc = (nn.functional.normalize(desc1_post_loc, dim=1)
                              + nn.functional.normalize(crop_feat, dim=1))
            desc1_post_local_all.append(desc1_post_loc)

            lf_loc = desc1_post_loc[0, :, gt_local[0], gt_local[1]].unsqueeze(1).unsqueeze(2)
            lab_feature_all_local.append(lf_loc)

        # ---- Query images ----
        pred_all, gt_all = [], []
        for image, landmark_list, img_path_query in tqdm(dataloader, desc="Eval"):
            image2_batch, _ = extractor.preprocess(img_path_query[0], load_size)
            desc2_list = extractor.extract_multi_layer_multi_facet_descriptors(
                image2_batch.to(device),
                layers=mlmf_config['layers'], facets=mlmf_config['facets'], bin=bin)
            num_patches2, load_size2 = extractor.num_patches, extractor.load_size
            descriptors2_post = model_post(desc2_list, num_patches2, load_size2, islocal=False)
            descriptors2_post_large = torch.nn.functional.interpolate(
                descriptors2_post, original_size, mode='bilinear')
            size_y2, size_x2 = descriptors2_post.shape[-2:]

            points1, points2 = [], []
            for i in range(len(lab)):
                points1.append([landmark_list[i][0].item(), landmark_list[i][1].item()])

                # Global matching with bidirectional verification
                sim = torch.nn.CosineSimilarity(dim=0)(lab_feature_all[i], descriptors2_post[0])
                h, w = sim.shape
                sim_flat = sim.reshape(-1)
                _, nn_k = torch.topk(sim_flat, k=topk, largest=True)

                dist_best, idx_best = 1e18, 0
                for k in range(topk):
                    iy, ix = nn_k[k] // w, nn_k[k] % w
                    sim_rev = torch.nn.CosineSimilarity(dim=0)(
                        descriptors2_post[0, :, iy, ix].unsqueeze(1).unsqueeze(2),
                        descriptors1_post[0])
                    _, nn_rev = torch.max(sim_rev.reshape(-1), dim=-1)
                    ry, rx = nn_rev // size_x, nn_rev % size_x
                    x1s = rx / size_x * original_size[1]
                    y1s = ry / size_y * original_size[0]
                    d = (y1s - int(lab[i][0])) ** 2 + (x1s - int(lab[i][1])) ** 2
                    if d < dist_best:
                        dist_best, idx_best = d, k

                best_flat = nn_k[idx_best].item()
                y2_global = round(best_flat // size_x2 / size_y2 * original_size[0])
                x2_global = round(best_flat % size_x2 / size_x2 * original_size[1])

                # Local refinement
                img2_local, _, gt_local2, offset2, crop_feat2 = \
                    extractor.preprocess_local_withfeature(
                        img_path_query[0], load_size,
                        [int(y2_global), int(x2_global)],
                        descriptors2_post_large)
                desc2_loc_list = extractor.extract_multi_layer_multi_facet_descriptors(
                    img2_local.to(device),
                    layers=mlmf_config['layers'], facets=mlmf_config['facets'], bin=bin)
                num_patches2_loc, load_size2_loc = extractor.num_patches, extractor.load_size
                desc2_post_loc = model_post(desc2_loc_list, num_patches2_loc,
                                            load_size2_loc, islocal=True)
                desc2_post_loc = (nn.functional.normalize(desc2_post_loc, dim=1)
                                  + nn.functional.normalize(crop_feat2, dim=1))

                sim_loc = torch.nn.CosineSimilarity(dim=0)(
                    lab_feature_all_local[i], desc2_post_loc[0])
                h2l, w2l = sim_loc.shape
                _, nn_kl = torch.topk(sim_loc.reshape(-1), k=topk, largest=True)

                dist_best_l, idx_best_l = 1e18, 0
                size_y1l, size_x1l = desc1_post_local_all[i].shape[-2:]
                for k in range(topk):
                    iyl, ixl = nn_kl[k] // w2l, nn_kl[k] % w2l
                    sim_rev_l = torch.nn.CosineSimilarity(dim=0)(
                        desc2_post_loc[0, :, iyl, ixl].unsqueeze(1).unsqueeze(2),
                        desc1_post_local_all[i][0])
                    _, nn_rev_l = torch.max(sim_rev_l.reshape(-1), dim=-1)
                    ryl, rxl = nn_rev_l // size_x1l, nn_rev_l % size_x1l
                    d = (ryl - gt_local_all[i][0]) ** 2 + (rxl - gt_local_all[i][1]) ** 2
                    if d < dist_best_l:
                        dist_best_l, idx_best_l = d, k

                best_flat_l = nn_kl[idx_best_l].item()
                y2_loc = best_flat_l // w2l
                x2_loc = best_flat_l % w2l
                y2_final = int(offset2[0]) + y2_loc
                x2_final = int(offset2[1]) + x2_loc
                points2.append([y2_final, x2_final])

            pred_all.append(points2)
            gt_all.append(points1)

    return pred_all, gt_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune MLMF local branch.')
    parser.add_argument('--save_dir', type=str, default='/home/u6da/taotl.u6da/FM-OSD/output')
    parser.add_argument('--load_size', default=224, type=int)
    parser.add_argument('--stride', default=4, type=int)
    parser.add_argument('--model_type', default='dino_vits8', type=str)
    parser.add_argument('--facet', default='key', type=str)
    parser.add_argument('--layer', default=8, type=int)
    parser.add_argument('--bin', default='True', type=str2bool)
    parser.add_argument('--topk', default=3, type=int)

    parser.add_argument('--dataset_pth', type=str,
                        default='/home/u6da/taotl.u6da/FM-OSD/data/Cephalometric/')
    parser.add_argument('--input_size', default=[2400, 1935])
    parser.add_argument('--id_shot', default=125, type=int)
    parser.add_argument('--eval_radius', default=[2, 2.5, 3, 4, 6, 8])

    parser.add_argument('--bs', default=4, type=int)
    parser.add_argument('--random_range', default=50, type=int)
    parser.add_argument('--local_coe', default=1.0, type=float)
    parser.add_argument('--max_epoch', default=300, type=int)
    parser.add_argument('--max_iterations', type=int, default=50)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--exp', default='local_mlmf')

    parser.add_argument('--mlmf_layers', default='5,8,11', type=str)
    parser.add_argument('--mlmf_facets', default='key,value', type=str)

    # Path to pretrained MLMF global checkpoint
    parser.add_argument('--global_ckpt', type=str,
                        default='/home/u6da/taotl.u6da/FM-OSD/models/global_mlmf/model_post_best.pth')

    args = parser.parse_args()

    mlmf_config = {
        'layers': [int(x) for x in args.mlmf_layers.split(',')],
        'facets': args.mlmf_facets.split(','),
        'num_sources': len(args.mlmf_layers.split(',')) * len(args.mlmf_facets.split(','))
    }
    print(f"MLMF Config: {mlmf_config}")

    random_seed = 2022
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    train_dataset = TrainDataset(istrain=0, original_size=args.input_size, load_size=args.load_size)
    train_loader  = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=2)

    one_shot_loader = Head_SSL_Infer(
        pathDataset=args.dataset_pth, mode='Oneshot',
        size=args.input_size, id_oneshot=args.id_shot)
    _, landmarks_temp, img_path_temp = one_shot_loader.__getitem__(0)

    dataset_test = Head_SSL_Infer(args.dataset_pth, mode='Test', size=args.input_size)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=2)

    image, _, _, _ = train_dataset.__getitem__(0)
    image_size = (image.shape[-2], image.shape[-1])

    model_post = Upnet_v3_MLMF_CoarseToFine(
        size=image_size, in_channels=6528, out_channels=256,
        num_sources=mlmf_config['num_sources'], fusion_reduction=4
    ).to(device)

    # Load global branch checkpoint and transfer weights.
    # Upnet_v3_MLMF uses keys: fusion.*, conv_out.*
    # Upnet_v3_MLMF_CoarseToFine uses: fusion_global.*, conv_out1.*
    # We remap them explicitly.
    if os.path.exists(args.global_ckpt):
        global_sd = torch.load(args.global_ckpt, map_location=device)
        model_sd  = model_post.state_dict()

        remap = {}
        for k, v in global_sd.items():
            if k.startswith('fusion.'):
                remap['fusion_global.' + k[len('fusion.'):]] = v
            elif k.startswith('conv_out.'):
                remap['conv_out1.' + k[len('conv_out.'):]] = v
            elif k in model_sd:
                remap[k] = v

        matched = {k: v for k, v in remap.items() if k in model_sd}
        model_sd.update(matched)
        model_post.load_state_dict(model_sd)
        print(f"Transferred {len(matched)}/{len(model_sd)} keys from {args.global_ckpt}")
        print(f"  Mapped keys: {list(matched.keys())[:5]} ...")
    else:
        print(f"WARNING: global checkpoint not found at {args.global_ckpt}. Training from scratch.")

    # Freeze global branch, only train local branch
    for name, param in model_post.named_parameters():
        if 'fusion_global' in name or 'conv_out1' in name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)

    trainable_params = sum(p.numel() for p in model_post.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model_post.parameters()), lr=args.lr)

    snapshot_path = f'/home/u6da/taotl.u6da/FM-OSD/models/{args.exp}'
    os.makedirs(snapshot_path, exist_ok=True)
    writer = SummaryWriter(snapshot_path + '/log')

    extractor = ViTExtractor(args.model_type, args.stride, device=device)

    best_mre = float('inf')
    iter_num  = 0

    # Initial evaluation
    model_post.eval()
    pred_all, gt_all = find_landmark_all_mlmf_local(
        extractor, device, model_post, img_path_temp, dataloader_test,
        landmarks_temp, mlmf_config, args.load_size, args.layer, args.facet, args.bin,
        args.input_size, args.topk)
    eval_dir = Path(args.save_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    evaluater = Evaluater(pred_all, gt_all, args.eval_radius, eval_dir,
                          name='mlmf_local_init', spacing=[0.1, 0.1])
    evaluater.calculate()
    evaluater.cal_metrics()
    best_mre = evaluater.mre
    torch.save(model_post.state_dict(),
               os.path.join(snapshot_path, f'model_post_fine_iter_0_{best_mre:.4f}.pth'))
    model_post.train()

    print(f"Starting MLMF local-branch fine-tuning for {args.max_iterations} iterations...")

    for epoch in range(args.max_epoch):
        model_post.train()
        for images, labs, lab_smalls, image_paths in train_loader:
            iter_num += 1
            optimizer.zero_grad()

            for j in range(labs.shape[1]):  # iterate landmarks
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
                        layers=mlmf_config['layers'],
                        facets=mlmf_config['facets'],
                        bin=args.bin)
                    num_patches_local, load_size_local = extractor.num_patches, extractor.load_size

                desc_post_local = model_post(
                    desc_local_list, num_patches_local, load_size_local, islocal=True)

                loss_local = (args.local_coe
                              * heatmap_mse_loss(desc_post_local, lab_local, var=2.0)
                              / labs.shape[1])
                loss_local.backward(retain_graph=True)
                writer.add_scalar(f'loss/local_{j}', loss_local.item(), iter_num)

            optimizer.step()
            print(f"[Iter {iter_num}] loss_local: {loss_local.item():.6f}")

            if iter_num % 20 == 0:
                model_post.eval()
                pred_all, gt_all = find_landmark_all_mlmf_local(
                    extractor, device, model_post, img_path_temp, dataloader_test,
                    landmarks_temp, mlmf_config, args.load_size, args.layer,
                    args.facet, args.bin, args.input_size, args.topk)
                evaluater = Evaluater(pred_all, gt_all, args.eval_radius, eval_dir,
                                      name=f'mlmf_local_{iter_num}', spacing=[0.1, 0.1])
                evaluater.calculate()
                evaluater.cal_metrics()
                mre = evaluater.mre
                writer.add_scalar('eval/mre', mre, iter_num)
                if mre < best_mre:
                    best_mre = mre
                    ckpt = os.path.join(snapshot_path,
                                        f'model_post_fine_iter_{iter_num}_{best_mre:.4f}.pth')
                    torch.save(model_post.state_dict(), ckpt)
                    print(f"  ** New best MRE: {best_mre:.4f} mm — saved to {ckpt}")
                model_post.train()

            if iter_num % 100 == 0:
                torch.save(model_post.state_dict(),
                           os.path.join(snapshot_path,
                                        f'model_post_fine_iter_{iter_num}.pth'))

            if iter_num >= args.max_iterations:
                break
        if iter_num >= args.max_iterations:
            break

    final_path = os.path.join(snapshot_path, 'model_post_final.pth')
    torch.save(model_post.state_dict(), final_path)
    print(f"Done. Best MRE: {best_mre:.4f} mm. Final: {final_path}")
