# Testing with MLMF (Multi-Layer Multi-Facet) + TCGR (Topology-Constrained Graph Refinement)
import argparse
import torch
from pathlib import Path
from extractor_gpu import ViTExtractor
from tqdm import tqdm
import numpy as np

from datasets.head_train import *
from datasets.head import *
from torch.utils.data import DataLoader
from evaluation.eval import *
from post_net import Upnet_v3_MLMF, Upnet_v3_MLMF_CoarseToFine
from landmark_graph import TCGRModule, normalize_coordinates, denormalize_coordinates
import torch.nn as nn

import torch.optim as optim
import os
import json


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def find_landmark_all_mlmf_tcgr(
    extractor, device, model_post, tcgr_module,
    image_path1: str, dataloader, lab, 
    mlmf_config: dict,
    load_size: int = 224, layer: int = 9,
    facet: str = 'key', bin: bool = True, 
    thresh: float = 0.05, model_type: str = 'dino_vits8',
    stride: int = 4, original_size=[2400, 1935], topk=5,
    use_tcgr: bool = True
):
    """
    Find landmarks using MLMF feature extraction + TCGR graph refinement.
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
    scores_all = []  # Store similarity scores for TCGR
    
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
        sim_scores = []  # Similarity scores for this image
        local_features = []  # Local features for TCGR
        
        imgs_all.append(image)
        img_names_all.append(img_path_query)
        
        # Iterate over each landmark
        for i in range(len(lab)):
            points1.append([landmark_list[i][0].item(), landmark_list[i][1].item()])
            
            # Compute cosine similarity
            similarities = torch.nn.CosineSimilarity(dim=0)(lab_feature_all[i], descriptors2_post[0])
            
            h2, w2 = similarities.shape
            similarities_flat = similarities.reshape(1, -1).squeeze(0)
            sim_k, nn_k = torch.topk(similarities_flat, k=topk, dim=-1, largest=True)
            
            # Bidirectional verification
            distance_best = 1000000000
            index_best = 0
            best_score = 0
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
                    best_score = sim_k[index].item()
            
            img2_indices_to_show = nn_k[index_best:index_best+1].cpu().item()
            
            size_y, size_x = descriptors2_post.shape[-2:]
            y2_show = img2_indices_to_show // size_x
            x2_show = img2_indices_to_show % size_x
            
            # Extract local feature at predicted location for TCGR
            local_feat = descriptors2_post[0, :, y2_show, x2_show].detach()
            local_features.append(local_feat)
            
            y2_show = np.round(y2_show / size_y * original_size[0])
            x2_show = np.round(x2_show / size_x * original_size[1])
            points2.append([y2_show, x2_show])
            sim_scores.append(best_score)
        
        # Apply TCGR refinement if enabled
        if use_tcgr and tcgr_module is not None:
            # Prepare inputs for TCGR
            coords = torch.tensor(points2, dtype=torch.float32).unsqueeze(0).to(device)  # [1, N, 2]
            coords_norm = normalize_coordinates(coords, original_size)
            
            scores = torch.tensor(sim_scores, dtype=torch.float32).unsqueeze(0).to(device)  # [1, N]
            
            # Stack local features
            local_feats = torch.stack(local_features, dim=0).unsqueeze(0)  # [1, N, C]
            # Reduce feature dimension if needed
            if local_feats.shape[-1] > 64:
                local_feats = local_feats[..., :64]
            
            # TCGR forward pass
            with torch.no_grad():
                refined_coords_norm, offsets = tcgr_module(coords_norm, scores, local_feats)
            
            # Denormalize back to pixel space
            refined_coords = denormalize_coordinates(refined_coords_norm, original_size)
            
            # Update points2 with refined coordinates
            points2 = refined_coords[0].cpu().numpy().tolist()
        
        pred_all.append(points2)
        gt_all.append(points1)
        scores_all.append(sim_scores)
    
    return pred_all, gt_all, imgs_all, img_names_all, scores_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test with MLMF + TCGR.')
    parser.add_argument('--save_dir', type=str, default='output', required=False)
    parser.add_argument('--load_size', default=224, type=int)
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
    
    # MLMF arguments
    parser.add_argument('--mlmf_layers', default='5,8,11', type=str)
    parser.add_argument('--mlmf_facets', default='key,value', type=str)
    
    # Model paths
    parser.add_argument('--model_post_path', type=str, default='models/global_mlmf/model_post_mlmf_best.pth')
    parser.add_argument('--tcgr_path', type=str, default='models/tcgr/tcgr_best.pth')
    
    # TCGR options
    parser.add_argument('--use_tcgr', default='True', type=str2bool)
    parser.add_argument('--tcgr_hidden_dim', default=128, type=int)
    parser.add_argument('--tcgr_num_layers', default=2, type=int)
    parser.add_argument('--tcgr_use_attention', default='True', type=str2bool)
    
    # Output
    parser.add_argument('--json_output', type=str, default='results/predictions.json')
    
    args = parser.parse_args()
    
    # Parse MLMF config
    mlmf_config = {
        'layers': [int(x) for x in args.mlmf_layers.split(',')],
        'facets': args.mlmf_facets.split(','),
        'num_sources': len(args.mlmf_layers.split(',')) * len(args.mlmf_facets.split(','))
    }
    print(f"MLMF Config: {mlmf_config}")
    print(f"TCGR Enabled: {args.use_tcgr}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load one-shot template
    one_shot_loader = Head_SSL_Infer(
        pathDataset=args.dataset_pth,
        mode='Oneshot', size=args.input_size, id_oneshot=args.id_shot
    )
    _, landmarks_temp, img_path_temp = one_shot_loader.__getitem__(0)
    
    # Load test dataset
    dataset_test = Head_SSL_Infer(args.dataset_pth, mode='Test', size=args.input_size)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=2)
    
    # Initialize extractor
    extractor = ViTExtractor(args.model_type, args.stride, device=device)
    
    # Initialize MLMF model
    image_size = (args.load_size, args.load_size)
    model_post = Upnet_v3_MLMF(
        size=image_size,
        in_channels=6528,
        out_channels=256,
        num_sources=mlmf_config['num_sources'],
        fusion_reduction=4
    ).to(device)
    
    # Load MLMF weights if available
    if os.path.exists(args.model_post_path):
        model_post.load_state_dict(torch.load(args.model_post_path, map_location=device))
        print(f"Loaded MLMF model from {args.model_post_path}")
    else:
        print(f"MLMF model path not found: {args.model_post_path}, using random weights")
    model_post.eval()
    
    # Initialize TCGR module
    tcgr_module = None
    if args.use_tcgr:
        tcgr_module = TCGRModule(
            num_landmarks=19,
            coord_dim=2,
            score_dim=1,
            feature_dim=64,
            hidden_dim=args.tcgr_hidden_dim,
            num_layers=args.tcgr_num_layers,
            use_attention=args.tcgr_use_attention
        ).to(device)
        
        # Load TCGR weights if available
        if os.path.exists(args.tcgr_path):
            tcgr_module.load_state_dict(torch.load(args.tcgr_path, map_location=device))
            print(f"Loaded TCGR model from {args.tcgr_path}")
        else:
            print(f"TCGR model path not found: {args.tcgr_path}, using random weights")
        tcgr_module.eval()
    
    # Run inference
    print("Running MLMF + TCGR inference...")
    with torch.no_grad():
        pred_all, gt_all, imgs_all, img_names_all, scores_all = find_landmark_all_mlmf_tcgr(
            extractor, device, model_post, tcgr_module,
            img_path_temp, dataloader_test, landmarks_temp,
            mlmf_config, args.load_size, args.layer, args.facet, args.bin,
            args.thresh, args.model_type, args.stride, 
            topk=args.topk, use_tcgr=args.use_tcgr
        )
    
    # Evaluate
    test_name = 'mlmf_tcgr' if args.use_tcgr else 'mlmf_only'
    save_root = args.save_dir + '/' + test_name
    save_root = Path(save_root)
    save_root.mkdir(exist_ok=True, parents=True)
    
    evaluater = Evaluater(
        pred_all, gt_all, args.eval_radius, save_root,
        name=f'{test_name}_layer_{args.layer}_{args.id_shot}',
        spacing=[0.1, 0.1], imgs=imgs_all, img_names=img_names_all
    )
    evaluater.calculate()
    evaluater.cal_metrics()
    
    # Save predictions to JSON
    json_dir = os.path.dirname(args.json_output)
    if json_dir and not os.path.exists(json_dir):
        os.makedirs(json_dir)
    
    results = {
        'predictions': pred_all,
        'ground_truth': gt_all,
        'mre': evaluater.mre,
        'sdr': evaluater.sdr,
        'config': {
            'mlmf_layers': mlmf_config['layers'],
            'mlmf_facets': mlmf_config['facets'],
            'use_tcgr': args.use_tcgr,
            'topk': args.topk
        }
    }
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        return obj
    
    results = convert_to_serializable(results)
    
    with open(args.json_output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.json_output}")
    
    print(f"\nFinal Results ({test_name}):")
    print(f"  MRE: {evaluater.mre:.4f} mm")
    print(f"  SDR: {evaluater.sdr}")
