# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import lmdb
import os
from pathlib import Path
import pickle
import torchvision.datasets as dset
from tqdm import tqdm, trange
import shutil
from torch.utils.data import DataLoader
import io
import torch 

def dump_obj(obj):
    return pickle.dumps(obj)


def main(split, img_path, lmdb_path):
    assert split in {"train", "valid", "test"}
    # create target directory
    if not os.path.exists(lmdb_path):
        os.makedirs(lmdb_path, exist_ok=True)

    lmdb_split = split
    lmdb_path = os.path.join(lmdb_path, '%s.lmdb' % lmdb_split)
    
    if os.path.exists(lmdb_path):
        shutil.rmtree(lmdb_path)
        os.makedirs(lmdb_path, exist_ok=True)

    # if you don't have this will download the data
    data = dset.celeba.CelebA(root=img_path, split=split, target_type='attr', transform=None, download=True)

    # create lmdb
    env = lmdb.open(lmdb_path, map_size=int(1e12))
    dataset_size = 0
    
    with env.begin(write=True) as txn:
        for i in trange(len(data)):
            file_path = os.path.join(data.root, data.base_folder, "img_align_celeba", data.filename[i])
            attr = data.attr[i, :].numpy()
            
            with open(file_path, 'rb') as f:
                file_data = f.read()

            txn.put(str(i).encode(), dump_obj((file_data, attr)))
            dataset_size+=1
    
    file_path = Path(lmdb_path).joinpath(f'{split}_dsize.pkl')
    if os.path.exists(file_path):
        os.remove(file_path)
    
    with open(file_path, 'wb') as f:
        pickle.dump(dataset_size, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('CelebA 64 LMDB creator.')
    # experimental results
    parser.add_argument('--img_path', type=str, default='./data/',
                        help='location of images for CelebA dataset')
    parser.add_argument('--lmdb_path', type=str, default='./data/celeba_lmdb',
                        help='target location for storing lmdb files')
    parser.add_argument('--split', type=str, default='train',
                        help='training or validation split', choices=["train", "valid", "test"])
    args = parser.parse_args()
    main(args.split, args.img_path, args.lmdb_path)
