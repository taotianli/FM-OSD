"""
Precompute ViT backbone features for all fixed images (no augmentation):
  - One-shot / template image (1 image)
  - Test images (mode='Test', 250 images)
  - Training patches in data/head/image/ (500 patches)

Features are saved as float16 .pt files under:
  cache/feat/<model_type>/shot_<id>.pt
  cache/feat/<model_type>/test_<stem>.pt
  cache/feat/<model_type>/train_<stem>.pt

Each file: {'desc': Tensor(float16), 'num_patches': (H, W)}

Usage:
  python precompute_backbone_features.py --model_type dino_vitb8
  python precompute_backbone_features.py --model_type dinov2_vits14
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from extractor_gpu import ViTExtractor, make_extractor
from datasets.head import Head_SSL_Infer


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def extract_and_save(extractor, img_path: str, save_path: Path,
                     load_size: int, layer: int, facet: str, use_bin: bool,
                     device: str):
    """Extract descriptors for one image and save as float16 .pt file."""
    if save_path.exists():
        return  # already done

    image_batch, _ = extractor.preprocess(img_path, load_size)
    with torch.no_grad():
        desc = extractor.extract_descriptors(
            image_batch.to(device), layer, facet, use_bin
        )
    num_patches = extractor.num_patches  # (H_patches, W_patches)

    torch.save(
        {'desc': desc.cpu().half(), 'num_patches': num_patches},
        save_path
    )


def main():
    parser = argparse.ArgumentParser(
        description='Precompute backbone features for backbone ablation.')
    parser.add_argument('--model_type', default='dino_vits8', type=str)
    parser.add_argument('--load_size', default=224, type=int)
    parser.add_argument('--layer', default=8, type=int)
    parser.add_argument('--facet', default='key', type=str)
    parser.add_argument('--bin', default='True', type=str2bool)
    parser.add_argument('--stride', default=None, type=int,
                        help='Override stride. None = auto.')
    parser.add_argument('--dataset_pth',
                        default='/home/u6da/taotl.u6da/FM-OSD/data/Cephalometric/',
                        type=str)
    parser.add_argument('--train_patch_dir',
                        default='/projects/u6da/fmosd_cache/head/image',
                        type=str)
    parser.add_argument('--cache_dir',
                        default='/projects/u6da/fmosd_cache/feat',
                        type=str)
    parser.add_argument('--id_shot', default=125, type=int,
                        help='One-shot template image ID.')
    parser.add_argument('--input_size', default=[2400, 1935])
    parser.add_argument('--device', default='cuda', type=str)

    args = parser.parse_args()

    if args.stride is None:
        args.stride = ViTExtractor.get_default_stride(args.model_type)

    device = args.device
    out_dir = Path(args.cache_dir) / args.model_type
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[precompute] model_type={args.model_type}, stride={args.stride}, '
          f'layer={args.layer}, facet={args.facet}, bin={args.bin}')
    print(f'[precompute] output dir: {out_dir}')

    extractor = make_extractor(args.model_type, args.stride, device=device)

    # ── 1. One-shot template image ─────────────────────────────────────────
    oneshot_ds = Head_SSL_Infer(
        pathDataset=args.dataset_pth,
        mode='Oneshot',
        size=args.input_size,
        id_oneshot=args.id_shot
    )
    _, _, img_path_shot = oneshot_ds.__getitem__(0)

    shot_save = out_dir / f'shot_{args.id_shot}.pt'
    print(f'[precompute] Shot image: {img_path_shot}')
    extract_and_save(extractor, img_path_shot, shot_save,
                     args.load_size, args.layer, args.facet, args.bin, device)
    print(f'  → saved {shot_save}')

    # ── 2. Test images ─────────────────────────────────────────────────────
    test_ds = Head_SSL_Infer(
        pathDataset=args.dataset_pth,
        mode='Test',
        size=args.input_size
    )
    print(f'[precompute] Test images: {len(test_ds)} total')
    for idx in tqdm(range(len(test_ds)), desc='test'):
        _, _, pth_img = test_ds.__getitem__(idx)
        # pth_img is a path string to the .bmp file
        stem = Path(pth_img).stem
        save_path = out_dir / f'test_{stem}.pt'
        extract_and_save(extractor, pth_img, save_path,
                         args.load_size, args.layer, args.facet, args.bin, device)

    # ── 3. Training patches ────────────────────────────────────────────────
    patch_dir = Path(args.train_patch_dir)
    patch_files = sorted(patch_dir.glob('*.png')) + sorted(patch_dir.glob('*.jpg'))
    print(f'[precompute] Train patches: {len(patch_files)} files')
    for pth in tqdm(patch_files, desc='train_patch'):
        save_path = out_dir / f'train_{pth.stem}.pt'
        extract_and_save(extractor, str(pth), save_path,
                         args.load_size, args.layer, args.facet, args.bin, device)

    # ── Summary ────────────────────────────────────────────────────────────
    n_files = len(list(out_dir.glob('*.pt')))
    total_bytes = sum(f.stat().st_size for f in out_dir.glob('*.pt'))
    print(f'\n[precompute] Done. {n_files} files, '
          f'{total_bytes / 1e9:.2f} GB total in {out_dir}')


if __name__ == '__main__':
    main()
