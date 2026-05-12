# Offline augmentation for Hand one-shot template (→ data/hand/)
import argparse
import os
import numpy as np

from datasets.hand import Hand_SSL_Infer_SSLv1_generate


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_pth', type=str,
                        default='/home/u6da/taotl.u6da/FM-OSD/data/Hand/hand/')
    parser.add_argument('--id_shot', default=0, type=int,
                        help='Index into sorted training IDs for one-shot template')
    parser.add_argument('--max_iter', default=500, type=int)
    args = parser.parse_args()

    one_shot = Hand_SSL_Infer_SSLv1_generate(
        pathDataset=args.dataset_pth, id_oneshot=args.id_shot)

    snapshot_path = '/projects/u6da/fmosd_cache/hand/'
    os.makedirs(snapshot_path, exist_ok=True)
    image_root = os.path.join(snapshot_path, 'image/')
    label_root = os.path.join(snapshot_path, 'label/')
    os.makedirs(image_root, exist_ok=True)
    os.makedirs(label_root, exist_ok=True)

    tpl_id = one_shot.list[0]['ID']
    for iter_num in range(args.max_iter):
        image, landmarks_temp, _ = one_shot.__getitem__(0)
        image_path = os.path.join(image_root, f'{tpl_id}_{iter_num}.png')
        label_path = os.path.join(label_root, f'{tpl_id}_{iter_num}.npy')
        image.save(image_path)
        np.save(label_path, np.array(landmarks_temp, dtype=np.int64))

    print(f"Saved {args.max_iter} pairs under {snapshot_path}")
