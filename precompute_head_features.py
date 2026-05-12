"""
Precompute MLMF backbone features for all Cephalometric (head) pipeline images.

Saves float16 descriptor lists to:
  cache/feat_head/<model_type>/shot_<id_shot>.pt
  cache/feat_head/<model_type>/test_<stem>.pt
  cache/feat_head/<model_type>/train_<stem>.pt

Each file:
  {'desc_list': [Tensor(float16), ...],   # len = num_layers × num_facets
   'num_patches': (H_patches, W_patches)}

Usage:
  python precompute_head_features.py
  python precompute_head_features.py --model_type dino_vits8 --mlmf_layers 5,8,11 --mlmf_facets key,value
"""

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from extractor_gpu import ViTExtractor
from datasets.head import Head_SSL_Infer


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
    """Extract MLMF descriptors for one image and save as float16 .pt file.
    Skips if file already exists (safe to re-run after interruption).
    """
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
        description='Precompute MLMF backbone features for Cephalometric dataset.')
    parser.add_argument('--model_type',   default='dino_vits8')
    parser.add_argument('--load_size',    default=224,    type=int)
    parser.add_argument('--stride',       default=4,      type=int)
    parser.add_argument('--bin',          default='True', type=str2bool)
    parser.add_argument('--mlmf_layers',  default='5,8,11')
    parser.add_argument('--mlmf_facets',  default='key,value')
    parser.add_argument('--dataset_pth',
                        default='/home/taotl/Desktop/FM-OSD/dataset/Cephalometric/')
    parser.add_argument('--train_patch_dir',
                        default='/home/taotl/Desktop/FM-OSD/data/head/image')
    parser.add_argument('--cache_dir',
                        default='/home/taotl/Desktop/FM-OSD/cache/feat_head')
    parser.add_argument('--id_shot',  default=125, type=int)
    parser.add_argument('--input_size', default=[2400, 1935])
    parser.add_argument('--device',   default='cuda')
    args = parser.parse_args()

    layers = [int(x) for x in args.mlmf_layers.split(',')]
    facets = args.mlmf_facets.split(',')

    out_dir = Path(args.cache_dir) / args.model_type
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[precompute_head] model={args.model_type}  stride={args.stride}')
    print(f'[precompute_head] MLMF layers={layers}  facets={facets}  bin={args.bin}')
    print(f'[precompute_head] output → {out_dir}')

    extractor = ViTExtractor(args.model_type, args.stride, device=args.device)

    # ── 1. One-shot template ──────────────────────────────────────────────
    shot_ds = Head_SSL_Infer(
        pathDataset=args.dataset_pth, mode='Oneshot',
        size=args.input_size, id_oneshot=args.id_shot)
    _, _, img_path_shot = shot_ds.__getitem__(0)
    shot_save = out_dir / f'shot_{args.id_shot}.pt'
    print(f'[precompute_head] Shot: {img_path_shot}')
    extract_mlmf_and_save(extractor, img_path_shot, shot_save,
                          args.load_size, layers, facets, args.bin, args.device)
    print(f'  → {shot_save}')

    # ── 2. Test images (RawImage/Test1Data, 001-400 .bmp) ─────────────────
    test_ds = Head_SSL_Infer(pathDataset=args.dataset_pth, mode='Test',
                              size=args.input_size)
    print(f'[precompute_head] Test images: {len(test_ds)}')
    for idx in tqdm(range(len(test_ds)), desc='test'):
        _, _, img_path = test_ds.__getitem__(idx)
        stem = Path(img_path).stem
        save_path = out_dir / f'test_{stem}.pt'
        extract_mlmf_and_save(extractor, img_path, save_path,
                              args.load_size, layers, facets, args.bin, args.device)

    # ── 3. Training patches — skipped (too large, ~150 GB for 500 patches)
    #    Train loop uses live encode for patches; only eval needs cache.
    print('[precompute_head] Train patches: skipped (cache shot+test only)')

    n_files = len(list(out_dir.glob('*.pt')))
    total_bytes = sum(f.stat().st_size for f in out_dir.glob('*.pt'))
    print(f'\n[precompute_head] Done: {n_files} files, '
          f'{total_bytes / 1e9:.2f} GB → {out_dir}')


if __name__ == '__main__':
    main()
