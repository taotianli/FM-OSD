# train the global branch with MLMF (Multi-Layer Multi-Facet) feature fusion
import argparse
import torch
from pathlib import Path
from extractor_gpu import ViTExtractor
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import List, Tuple

from datasets.head_train import *
from datasets.head import *
from torch.utils.data import DataLoader
from evaluation.eval import *
from post_net import Upnet_v3_MLMF
import torch.nn as nn

import torch.optim as optim
from tensorboardX import SummaryWriter
import os
import random


def find_landmark_all_mlmf(extractor, device, model_post, image_path1: str, dataloader, lab, 
                           mlmf_config: dict, load_size: int = 224, layer: int = 9,
                           facet: str = 'key', bin: bool = True, thresh: float = 0.05, 
                           model_type: str = 'dino_vits8', stride: int = 4, 
                           original_size=[2400, 1935], topk=5):
    """
    Find landmarks using MLMF feature extraction.
    """
    image1_batch, image1_pil = extractor.preprocess(image_path1, load_size)
    
    # MLMF: Extract multi-layer multi-facet descriptors
    descriptors1_list = extractor.extract_multi_layer_multi_facet_descriptors(
        image1_batch.to(device), 
        layers=mlmf_config['layers'], 
        facets=mlmf_config['facets'],
        bin=bin
    )
    num_patches1, load_size1 = extractor.num_patches, extractor.load_size
    
    # Use MLMF fusion network
    descriptors1_post = model_post(descriptors1_list, num_patches1)
    
    # Obtain template features for each landmark
    lab_feature_all = []
    for i in range(len(lab)):
        lab_y = int(lab[i][0])
        lab_x = int(lab[i][1])
        size_y, size_x = descriptors1_post.shape[-2:]
        lab_y = int(lab_y / original_size[0] * size_y)
        lab_x = int(lab_x / original_size[1] * size_x)
        
        lab_feature = descriptors1_post[0, :, lab_y, lab_x]
        lab_feature = lab_feature.unsqueeze(1).unsqueeze(2)
        lab_feature_all.append(lab_feature)
    
    pred_all = []
    gt_all = []
    imgs_all = []
    img_names_all = []
    
    # Iterate over all testing images
    for image, landmark_list, img_path_query in tqdm(dataloader):
        image2_batch, image2_pil = extractor.preprocess(img_path_query[0], load_size)
        
        # MLMF extraction for query image
        descriptors2_list = extractor.extract_multi_layer_multi_facet_descriptors(
            image2_batch.to(device),
            layers=mlmf_config['layers'],
            facets=mlmf_config['facets'],
            bin=bin
        )
        num_patches2, load_size2 = extractor.num_patches, extractor.load_size
        
        descriptors2_post = model_post(descriptors2_list, num_patches2)
        
        points1 = []
        points2 = []
        
        imgs_all.append(image)
        img_names_all.append(img_path_query)
        
        # Iterate over each landmark
        for i in range(len(lab)):
            points1.append([landmark_list[i][0].item(), landmark_list[i][1].item()])
            
            # Compute cosine similarity
            similarities = torch.nn.CosineSimilarity(dim=0)(lab_feature_all[i], descriptors2_post[0])
            
            h2, w2 = similarities.shape
            similarities = similarities.reshape(1, -1).squeeze(0)
            sim_k, nn_k = torch.topk(similarities, k=topk, dim=-1, largest=True)
            
            # Bidirectional verification
            distance_best = 1000000000
            index_best = 0
            for index in range(topk):
                i_y = nn_k[index] // w2
                i_x = nn_k[index] % w2
                similarities_reverse = torch.nn.CosineSimilarity(dim=0)(
                    descriptors2_post[0, :, i_y, i_x].unsqueeze(1).unsqueeze(2), 
                    descriptors1_post[0]
                )
                h1, w1 = similarities_reverse.shape
                similarities_reverse = similarities_reverse.reshape(-1)
                _, nn_1 = torch.max(similarities_reverse, dim=-1)
                img1_y_to_show = nn_1 // w1
                img1_x_to_show = nn_1 % w1
                
                size_y, size_x = descriptors1_post.shape[-2:]
                x1_show = img1_x_to_show / size_x * original_size[1]
                y1_show = img1_y_to_show / size_y * original_size[0]
                
                distance_temp = pow(y1_show - int(lab[i][0]), 2) + pow(x1_show - int(lab[i][1]), 2)
                if distance_temp < distance_best:
                    distance_best = distance_temp
                    index_best = index
            
            img2_indices_to_show = nn_k[index_best:index_best+1].cpu().item()
            
            size_y, size_x = descriptors2_post.shape[-2:]
            y2_show = img2_indices_to_show // size_x
            x2_show = img2_indices_to_show % size_x
            
            y2_show = np.round(y2_show / size_y * original_size[0])
            x2_show = np.round(x2_show / size_x * original_size[1])
            points2.append([y2_show, x2_show])
        
        pred_all.append(points2)
        gt_all.append(points1)
    
    return pred_all, gt_all, imgs_all, img_names_all


def get_feature_mlmf(extractor, device, image1_batch, mlmf_config: dict,
                     load_size: int = 224, layer: int = 9, facet: str = 'key', 
                     bin: bool = True, thresh: float = 0.05, model_type: str = 'dino_vits8',
                     stride: int = 4, original_size=[2400, 1935], topk=5):
    """Extract MLMF features."""
    descriptors_list = extractor.extract_multi_layer_multi_facet_descriptors(
        image1_batch.to(device),
        layers=mlmf_config['layers'],
        facets=mlmf_config['facets'],
        bin=bin
    )
    num_patches, load_size = extractor.num_patches, extractor.load_size
    
    return descriptors_list, num_patches, load_size


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
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
            labtemp.append(make_heatmap(labels[l], [features.shape[-2], features.shape[-1]], var=var))
        labtemp2 = torch.stack(labtemp, dim=0)
        lab.append(labtemp2)
    
    label = torch.stack(lab, dim=0)
    label = label.to(features.device)
    
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
    
    loss = criterion(pred, label)
    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train global branch with MLMF fusion.')
    parser.add_argument('--save_dir', type=str, default='output', required=False)
    parser.add_argument('--load_size', default=224, type=int, help='load size of the input image.')
    parser.add_argument('--stride', default=4, type=int)
    parser.add_argument('--model_type', default='dino_vits8', type=str)
    parser.add_argument('--facet', default='key', type=str)
    parser.add_argument('--layer', default=8, type=int)
    parser.add_argument('--bin', default='True', type=str2bool)
    parser.add_argument('--thresh', default=0.05, type=float)
    parser.add_argument('--topk', default=5, type=int)
    
    parser.add_argument('--dataset_pth', type=str, default='dataset/Cephalometric/', required=False)
    parser.add_argument('--input_size', default=[2400, 1935])
    parser.add_argument('--id_shot', default=125, type=int)
    parser.add_argument('--eval_radius', default=[2, 2.5, 3, 4, 6, 8])
    
    parser.add_argument('--bs', default=4, type=int)
    parser.add_argument('--max_epoch', default=300, type=int)
    parser.add_argument('--max_iterations', type=int, default=8000)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--exp', default='global_mlmf', help='exp name.')
    
    # MLMF specific arguments
    parser.add_argument('--mlmf_layers', default='5,8,11', type=str, help='ViT layers for MLMF')
    parser.add_argument('--mlmf_facets', default='key,value', type=str, help='Facets for MLMF')
    parser.add_argument('--fusion_uniform', default='False', type=str2bool,
                        help='Ablation A5: use uniform equal weights instead of learned attention')

    args = parser.parse_args()
    
    # Parse MLMF config
    mlmf_config = {
        'layers': [int(x) for x in args.mlmf_layers.split(',')],
        'facets': args.mlmf_facets.split(','),
        'num_sources': len(args.mlmf_layers.split(',')) * len(args.mlmf_facets.split(','))
    }
    print(f"MLMF Config: {mlmf_config}")
    
    # Random seed
    random_seed = 2022
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    
    train_dataset = TrainDataset(istrain=0, original_size=args.input_size, load_size=args.load_size)
    train_dataloaders = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=2)
    
    one_shot_loader_val = Head_SSL_Infer(
        pathDataset=args.dataset_pth,
        mode='Oneshot', size=args.input_size, id_oneshot=args.id_shot
    )
    
    _, landmarks_temp_val, img_path_temp = one_shot_loader_val.__getitem__(0)
    
    dataset_test = Head_SSL_Infer(args.dataset_pth, mode='Test', size=args.input_size)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=2)
    
    image, _, _, _ = train_dataset.__getitem__(0)
    
    image_size = (image.shape[-2], image.shape[-1])
    
    # Each source is a 6528-dim log-bin descriptor (384 embed × 17 bins)
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
    
    # Model saving path
    snapshot_path = 'models/' + args.exp
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    writer = SummaryWriter(snapshot_path + '/log')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    extractor = ViTExtractor(args.model_type, args.stride, device=device)
    
    best_performance = 10000
    
    iter_num = 0
    max_iterations = args.max_iterations
    
    # Skip initial eval (save time; first eval happens at iter 500)
    model_post.train()
    
    # Training loop
    for epoch in np.arange(0, args.max_epoch) + 1:
        model_post.train()
        for images, labs, lab_smalls, image_paths in train_dataloaders:
            iter_num = iter_num + 1
            with torch.no_grad():
                descriptors_list, num_patches, load_size = get_feature_mlmf(
                    extractor, device, images, mlmf_config,
                    args.load_size, args.layer, args.facet, args.bin, 
                    args.thresh, args.model_type, args.stride, topk=args.topk
                )
            
            descriptors_post = model_post(descriptors_list, num_patches)
            
            optimizer.zero_grad()
            loss = heatmap_mse_loss(descriptors_post, lab_smalls)
            loss.backward()
            optimizer.step()
            
            writer.add_scalar('info/loss', loss, iter_num)
            print('iter: {}, loss: {}'.format(iter_num, loss))
            
            # Regular testing and saving
            if iter_num % 500 == 0:
                model_post.eval()
                with torch.no_grad():
                    pred_all, gt_all, imgs_all, img_names_all = find_landmark_all_mlmf(
                        extractor, device, model_post, img_path_temp, dataloader_test,
                        landmarks_temp_val, mlmf_config, args.load_size, args.layer,
                        args.facet, args.bin, args.thresh, args.model_type, args.stride, topk=args.topk
                    )
                
                test_name = 'dino_s_mlmf'
                save_root = args.save_dir + '/' + test_name
                save_root = Path(save_root)
                save_root.mkdir(exist_ok=True, parents=True)
                evaluater = Evaluater(
                    pred_all, gt_all, args.eval_radius, save_root,
                    name='stride4_224_mse_mlmf_layer_' + str(args.layer) + '_' + str(args.id_shot),
                    spacing=[0.1, 0.1], imgs=imgs_all, img_names=img_names_all
                )
                evaluater.calculate()
                evaluater.cal_metrics()
                
                performance = evaluater.mre
                if performance < best_performance:
                    best_performance = performance
                    save_best = os.path.join(snapshot_path, 'model_post_mlmf_iter_{}_{}.pth'.format(iter_num, best_performance))
                    torch.save(model_post.state_dict(), save_best)
                model_post.train()
            
            if iter_num % 1000 == 0:
                save_path = os.path.join(snapshot_path, 'model_post_mlmf_iter_{}.pth'.format(iter_num))
                torch.save(model_post.state_dict(), save_path)
            
            if iter_num >= max_iterations:
                break
        
        if iter_num >= max_iterations:
            break
    
    print(f"Training completed. Best MRE: {best_performance}")
