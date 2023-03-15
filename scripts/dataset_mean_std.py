import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import Sentinel2Dataset


dataset_path = '/data/datasets/sentinel'
split = 0.9
scale = None
tile_height, tile_width = 1022, 1022
tile_stride_y, tile_stride_x = 1022, 1022
batch_size = 16
loader_num_workers = 4

train_dataset = Sentinel2Dataset(dataset_path, train=True, split=split, tile_height=tile_height, tile_width=tile_width, 
        stride_y=tile_stride_y, stride_x=tile_stride_x, scale=scale)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=loader_num_workers)

psum = torch.tensor([0.0, 0.0, 0.0, 0.0])
psum_sq = torch.tensor([0.0, 0.0, 0.0, 0.0])
for it in tqdm(train_loader):
    tile, target = it
    psum += tile.sum(axis=[0, 2, 3])
    psum_sq += (tile**2).sum(axis=[0, 2, 3])

count = len(train_dataset) * tile_height * tile_width
total_mean = psum / count
total_std = torch.sqrt((psum_sq / count) - (total_mean**2))

print(f'mean: {total_mean}')
print(f'std: {total_std}')

