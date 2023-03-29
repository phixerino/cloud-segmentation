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
batch_size = 1
loader_num_workers = 4

train_dataset = Sentinel2Dataset(dataset_path, train=True, split=split, tile_height=tile_height, tile_width=tile_width, 
        stride_y=tile_stride_y, stride_x=tile_stride_x, scale=scale)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=loader_num_workers)

ones = torch.tensor(0)
zeros = torch.tensor(0)
for it in tqdm(train_loader):
    tile, target = it
    ones += target.sum((2,3)).type(torch.LongTensor).squeeze()
    zeros += (target==0.).sum((2,3)).type(torch.LongTensor).squeeze()
total = ones + zeros
print(f'Clear: {zeros} ({(zeros/total)*100:.2f}%), Cloud: {ones} ({(ones/total)*100:.2f}%)')

