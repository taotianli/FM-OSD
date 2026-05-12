# MLMF global branch training — Hand X-ray dataset (37 landmarks, variable image size)
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
from post_net import Upnet_v3_MLMF
import torch.optim as optim
from tensorboardX import SummaryWriter
import os
import random


def _norm_lab_entry(t):
    if hasattr(t[0], 'item'):
        return int(t[0].item()), int(t[1].item())
    return int(t[0]), int(t[1])


def find_landmark_all_mlmf_hand(
        extractor, device, model_post, image_path1, dataloader, lab,
        mlmf_config, load_size=224, layer=9, facet='key', bin=True,
        thresh=0.05, model_type='dino_vits8', stride=4,
        template_orig_size=None, topk=5):
    """Global matching only; per-query `original_size` from Hand loader."""
    if template_orig_size is None:
        im = Image.open(image_path1).convert('RGB')
        template_orig_size = [im.size[1], im.size[0]]

    image1_batch, _ = extractor.preprocess(image_path1, load_size)
    descriptors1_list = extractor.extract_multi_layer_multi_facet_descriptors(
        image1_batch.to(device),
        layers=mlmf_config['layers'],
        facets=mlmf_config['facets'],
        bin=bin)
    num_patches1, load_size1 = extractor.num_patches, extractor.load_size
    descriptors1_post = model_post(descriptors1_list, num_patches1)

    lab_feature_all = []
    for i in range(len(lab)):
        ly = int(lab[i][0] / template_orig_size[0] * descriptors1_post.shape[-2])
        lx = int(lab[i][1] / template_orig_size[1] * descriptors1_post.shape[-1])
        lf = descriptors1_post[0, :, ly, lx].unsqueeze(1).unsqueeze(2)
        lab_feature_all.append(lf)

    pred_all, gt_all, imgs_all, img_names_all = [], [], [], []

    for sample in dataloader:
        img_path_query, landmark_list, origin_size, _ = sample
        original_size_q = [int(origin_size[0]), int(origin_size[1])]

        image2_batch, image2_pil = extractor.preprocess(img_path_query, load_size)
        descriptors2_list = extractor.extract_multi_layer_multi_facet_descriptors(
            image2_batch.to(device),
            layers=mlmf_config['layers'],
            facets=mlmf_config['facets'],
            bin=bin)
        num_patches2, load_size2 = extractor.num_patches, extractor.load_size
        descriptors2_post = model_post(descriptors2_list, num_patches2)

        points1, points2 = [], []
        imgs_all.append(None)
        img_names_all.append([img_path_query])

        for i in range(len(lab)):
            yi, xi = _norm_lab_entry(landmark_list[i])
            points1.append([yi, xi])

            similarities = torch.nn.CosineSimilarity(dim=0)(
                lab_feature_all[i], descriptors2_post[0])
            h2, w2 = similarities.shape
            similarities = similarities.reshape(1, -1).squeeze(0)
            _, nn_k = torch.topk(similarities, k=topk, dim=-1, largest=True)

            distance_best = 1e18
            index_best = 0
            for idx in range(topk):
                i_y = nn_k[idx] // w2
                i_x = nn_k[idx] % w2
                sim_rev = torch.nn.CosineSimilarity(dim=0)(
                    descriptors2_post[0, :, i_y, i_x].unsqueeze(1).unsqueeze(2),
                    descriptors1_post[0])
                h1, w1 = sim_rev.shape
                sim_rev = sim_rev.reshape(-1)
                _, nn_1 = torch.max(sim_rev, dim=-1)
                img1_y = nn_1 // w1
                img1_x = nn_1 % w1
                sz_y, sz_x = descriptors1_post.shape[-2:]
                x1_show = img1_x / sz_x * template_orig_size[1]
                y1_show = img1_y / sz_y * template_orig_size[0]
                d = (y1_show - lab[i][0]) ** 2 + (x1_show - lab[i][1]) ** 2
                if d < distance_best:
                    distance_best = d
                    index_best = idx

            img2_idx = nn_k[index_best].cpu().item()
            sz_y2, sz_x2 = descriptors2_post.shape[-2:]
            y_patch = img2_idx // sz_x2
            x_patch = img2_idx % sz_x2
            y2 = np.round(y_patch / sz_y2 * original_size_q[0])
            x2 = np.round(x_patch / sz_x2 * original_size_q[1])
            points2.append([float(y2), float(x2)])

        pred_all.append(points2)
        gt_all.append(points1)

    return pred_all, gt_all, imgs_all, img_names_all


def get_feature_mlmf(extractor, device, image1_batch, mlmf_config,
                     load_size=224, layer=9, facet='key', bin=True,
                     thresh=0.05, model_type='dino_vits8', stride=4, topk=5):
    descriptors_list = extractor.extract_multi_layer_multi_facet_descriptors(
        image1_batch.to(device),
        layers=mlmf_config['layers'],
        facets=mlmf_config['facets'],
        bin=bin)
    num_patches, load_size = extractor.num_patches, extractor.load_size
    return descriptors_list, num_patches, load_size


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
    heatmap = torch.exp(heatmap * inner_factor)
    return heatmap


def heatmap_mse_loss(features, landmarks, var=5.0, criterion=torch.nn.MSELoss()):
    lab = []
    for i in range(len(landmarks)):
        labels = landmarks[i]
        labtemp = []
        for l in range(labels.shape[0]):
            labtemp.append(make_heatmap(
                labels[l], [features.shape[-2], features.shape[-1]], var=var))
        labtemp2 = torch.stack(labtemp, dim=0)
        lab.append(labtemp2)

    label = torch.stack(lab, dim=0)
    label = label.to(features.device)

    pred = []
    for i in range(len(landmarks)):
        feature_temp = features[i]
        pred_temp = []
        n_lm = landmarks[i].shape[0]
        for j in range(n_lm):
            gt = feature_temp[:, landmarks[i, j, 0], landmarks[i, j, 1]].unsqueeze(1).unsqueeze(2)
            similarity = torch.nn.CosineSimilarity(dim=0)(gt, feature_temp).unsqueeze(0)
            pred_temp.append(similarity)

        pred_temp = torch.cat(pred_temp, dim=0).unsqueeze(0)
        pred.append(pred_temp)

    pred = torch.cat(pred, dim=0)
    loss = criterion(pred, label)
    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MLMF global on Hand (37 landmarks).')
    parser.add_argument('--save_dir', type=str, default='/home/u6da/taotl.u6da/FM-OSD/output')
    parser.add_argument('--load_size', default=224, type=int)
    parser.add_argument('--stride', default=4, type=int)
    parser.add_argument('--model_type', default='dino_vits8', type=str)
    parser.add_argument('--facet', default='key', type=str)
    parser.add_argument('--layer', default=8, type=int)
    parser.add_argument('--bin', default='True', type=str2bool)
    parser.add_argument('--thresh', default=0.05, type=float)
    parser.add_argument('--topk', default=5, type=int)

    parser.add_argument('--dataset_pth', type=str,
                        default='/home/u6da/taotl.u6da/FM-OSD/data/Hand/hand/')
    parser.add_argument('--input_size', default=[2048, 2048],
                        help='Template H,W for heatmap resize (match data/hand augmented crop)')
    parser.add_argument('--id_shot', default=0, type=int)
    parser.add_argument('--eval_radius', default=[2, 2.5, 3, 4, 6, 8])

    parser.add_argument('--bs', default=4, type=int)
    parser.add_argument('--max_epoch', default=300, type=int)
    parser.add_argument('--max_iterations', type=int, default=8000)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--exp', default='global_mlmf_hand')

    parser.add_argument('--mlmf_layers', default='5,8,11', type=str)
    parser.add_argument('--mlmf_facets', default='key,value', type=str)
    parser.add_argument('--fusion_uniform', default='False', type=str2bool)

    parser.add_argument('--auto_input_size', default='True', type=str2bool,
                        help='Set input_size from first data/hand training image')

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

    train_dataset = HandTrainDataset(istrain=0, original_size=args.input_size,
                                     load_size=args.load_size)
    if args.auto_input_size and len(train_dataset.data) > 0:
        im = Image.open(train_dataset.data[0][0])
        args.input_size = [im.size[1], im.size[0]]
        train_dataset.original_size = args.input_size
        print(f"auto_input_size: template aug crop H,W = {args.input_size}")

    train_dataloaders = DataLoader(
        train_dataset, batch_size=args.bs, shuffle=True, num_workers=2)

    one_shot_loader_val = Hand_SSL_Infer(
        pathDataset=args.dataset_pth, mode='Oneshot', id_oneshot=args.id_shot)

    _, landmarks_temp_val, img_path_temp, _ = one_shot_loader_val.__getitem__(0)
    landmarks_temp_val = [_norm_lab_entry(t) for t in landmarks_temp_val]

    dataset_test = Hand_SSL_Infer(args.dataset_pth, mode='Test')


    def _collate_infer(x):
        return x[0]

    dataloader_test = DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=_collate_infer)

    image, _, _, _ = train_dataset.__getitem__(0)
    image_size = (image.shape[-2], image.shape[-1])

    in_channels = 6528
    model_post = Upnet_v3_MLMF(
        size=image_size,
        in_channels=in_channels,
        out_channels=256,
        num_sources=mlmf_config['num_sources'],
        fusion_reduction=4,
        fusion_uniform=args.fusion_uniform
    ).cuda()
    model_post.train()

    optimizer = optim.Adam(model_post.parameters(), lr=args.lr)

    snapshot_path = '/home/u6da/taotl.u6da/FM-OSD/models/' + args.exp
    os.makedirs(snapshot_path, exist_ok=True)
    writer = SummaryWriter(snapshot_path + '/log')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTExtractor(args.model_type, args.stride, device=device)

    tpl_im = Image.open(img_path_temp).convert('RGB')
    template_orig_size = [tpl_im.size[1], tpl_im.size[0]]

    best_performance = 10000.0
    iter_num = 0

    for epoch in np.arange(0, args.max_epoch) + 1:
        model_post.train()
        for images, labs, lab_smalls, image_paths in train_dataloaders:
            iter_num += 1
            with torch.no_grad():
                descriptors_list, num_patches, _ = get_feature_mlmf(
                    extractor, device, images, mlmf_config,
                    args.load_size, args.layer, args.facet, args.bin,
                    args.thresh, args.model_type, args.stride, args.topk)

            descriptors_post = model_post(descriptors_list, num_patches)
            optimizer.zero_grad()
            loss = heatmap_mse_loss(descriptors_post, lab_smalls)
            loss.backward()
            optimizer.step()

            writer.add_scalar('info/loss', loss, iter_num)
            print('iter: {}, loss: {}'.format(iter_num, loss))

            if iter_num % 500 == 0:
                model_post.eval()
                with torch.no_grad():
                    pred_all, gt_all, imgs_all, img_names_all = find_landmark_all_mlmf_hand(
                        extractor, device, model_post, img_path_temp,
                        dataloader_test, landmarks_temp_val, mlmf_config,
                        args.load_size, args.layer, args.facet, args.bin,
                        args.thresh, args.model_type, args.stride,
                        template_orig_size=template_orig_size, topk=args.topk)

                save_root = Path(args.save_dir) / 'dino_s_mlmf_hand'
                save_root.mkdir(exist_ok=True, parents=True)
                ev = Evaluater(
                    pred_all, gt_all, args.eval_radius, save_root,
                    name=f'stride4_224_mlmf_hand_id{args.id_shot}_iter{iter_num}',
                    spacing=[0.1, 0.1], imgs=imgs_all, img_names=img_names_all)
                ev.calculate()
                ev.cal_metrics()

                performance = ev.mre
                if performance < best_performance:
                    best_performance = performance
                    save_best = os.path.join(
                        snapshot_path,
                        f'model_post_mlmf_iter_{iter_num}_{best_performance}.pth')
                    torch.save(model_post.state_dict(), save_best)
                model_post.train()

            if iter_num % 1000 == 0:
                torch.save(model_post.state_dict(),
                           os.path.join(snapshot_path,
                                        f'model_post_mlmf_iter_{iter_num}.pth'))

            if iter_num >= args.max_iterations:
                break
        if iter_num >= args.max_iterations:
            break

    print(f"Training completed. Best MRE: {best_performance}")
