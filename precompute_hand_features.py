"""
Precompute MLMF backbone features for all Hand X-ray pipeline images.

Saves float16 descriptor lists to:
  cache/feat_hand/<model_type>/shot_<id>.pt
  cache/feat_hand/<model_type>/test_<stem>.pt
  cache/feat_hand/<model_type>/train_<stem>.pt

Each file:
  {'desc_list': [Tensor(float16), ...],   # len = num_layers × num_facets
   'num_patches': (H_patches, W_patches)}

Usage:
  python precompute_hand_features.py --model_type dino_vits8
"""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from extractor_gpu import ViTExtractor
from datasets.hand import Hand_SSL_Infer


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def extract_mlmf_and_save(extractor, img_path, save_path, load_size,
                           layers, facets, bin_flag, device):
    """Extract MLMF descriptors for one image and save as float16 .pt file."""
    if save_path.exists():
        return
    img_batch, _ = extractor.preprocess(str(img_path), load_size)
    with torch.no_grad():
        desc_list = extractor.extract_multi_layer_multi_facet_descriptors(
            img_batch.to(device), layers=layers, facets=facets, bin=bin_flag)
    num_patches = extractor.num_patches
    torch.save(
        {'desc_list': [d.cpu().half() for d in desc_list],
         'num_patches': num_patches},
        save_path
    )


def main():
    parser = argparse.ArgumentParser(
        description='Precompute MLMF backbone features for Hand X-ray dataset.')
    parser.add_argument('--model_type',  default='dino_vits8')
    parser.add_argument('--load_size',   default=224,    type=int)
    parser.add_argument('--stride',      default=4,      type=int)
    parser.add_argument('--bin',         default='True', type=str2bool)
    parser.add_argument('--mlmf_layers', default='5,8,11')
    parser.add_argument('--mlmf_facets', default='key,value')
    parser.add_argument('--dataset_pth',
                        default='/home/taotl/Desktop/FM-OSD/dataset/Hand/hand/')
    parser.add_argument('--train_patch_dir',
                        default='/home/taotl/Desktop/FM-OSD/data/hand/image')
    parser.add_argument('--cache_dir',
                        default='/home/taotl/Desktop/FM-OSD/cache/feat_hand')
    parser.add_argument('--id_shot', default=0, type=int)
    parser.add_argument('--device',  default='cuda')
    args = parser.parse_args()

    layers = [int(x) for x in args.mlmf_layers.split(',')]
    facets = args.mlmf_facets.split(',')

    out_dir = Path(args.cache_dir) / args.model_type
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[precompute_hand] model={args.model_type}  stride={args.stride}')
    print(f'[precompute_hand] MLMF layers={layers}  facets={facets}  bin={args.bin}')
    print(f'[precompute_hand] output → {out_dir}')

    extractor = ViTExtractor(args.model_type, args.stride, device=args.device)

    # ── 1. One-shot template ──────────────────────────────────────────────
    shot_ds = Hand_SSL_Infer(
        pathDataset=args.dataset_pth, mode='Oneshot', id_oneshot=args.id_shot)
    img_path_shot, _, _, _ = shot_ds.__getitem__(0)
    shot_save = out_dir / f'shot_{args.id_shot}.pt'
    print(f'[precompute_hand] Shot: {img_path_shot}')
    extract_mlmf_and_save(extractor, img_path_shot, shot_save,
                          args.load_size, layers, facets, args.bin, args.device)
    print(f'  → {shot_save}')

    # ── 2. Test images ────────────────────────────────────────────────────
    test_ds = Hand_SSL_Infer(pathDataset=args.dataset_pth, mode='Test')
    print(f'[precompute_hand] Test images: {len(test_ds)}')
    for idx in tqdm(range(len(test_ds)), desc='test'):
        img_path, _, _, _ = test_ds.__getitem__(idx)
        stem = Path(img_path).stem
        save_path = out_dir / f'test_{stem}.pt'
        extract_mlmf_and_save(extractor, img_path, save_path,
                              args.load_size, layers, facets, args.bin, args.device)

    # ── 3. Training patches (data/hand/image/) ────────────────────────────
    patch_dir = Path(args.train_patch_dir)
    patch_files = sorted(patch_dir.glob('*.png')) + sorted(patch_dir.glob('*.jpg'))
    print(f'[precompute_hand] Train patches: {len(patch_files)}')
    for pth in tqdm(patch_files, desc='train_patch'):
        save_path = out_dir / f'train_{pth.stem}.pt'
        extract_mlmf_and_save(extractor, pth, save_path,
                              args.load_size, layers, facets, args.bin, args.device)

    n_files = len(list(out_dir.glob('*.pt')))
    total_bytes = sum(f.stat().st_size for f in out_dir.glob('*.pt'))
    print(f'\n[precompute_hand] Done: {n_files} files, {total_bytes / 1e9:.2f} GB → {out_dir}')


if __name__ == '__main__':
    main()
