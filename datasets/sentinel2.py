import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import shutil
import time


class Sentinel2Dataset(Dataset):
    def __init__(self, dataset_path, train=True, split=0.9, transform=None, bands=None,
            tile_height=512, tile_width=512, stride_x=512, stride_y=512,
            subscene_height=1022, subscene_width=1022, scale=None, dataset_limit=None, debug=False, binary=False):
        
        self.transform = transform
        self.debug = debug
        self.tile_height = tile_height
        self.tile_width = tile_width
        self.dataset_limit = dataset_limit  # load only first X subscenes to train set to test overfitting
        self.subscene_height, self.subscene_width = subscene_height, subscene_width  # resize subscenes and masks to this shape
        self.binary = binary
    
        default_bands = [3, 2, 1, 7]  # R, G, B, NIR
        self.bands = bands if bands else default_bands
        
        # debug=True will save tiles and targets to disk 
        if self.debug:
            self.debug_folder = 'temp'
            if os.path.isdir(self.debug_folder):
                shutil.rmtree(self.debug_folder)
            os.makedirs(self.debug_folder)

        # optional scaling for the already resized images, so that tiles fit the entire image
        extra_width = (self.subscene_width - self.tile_width) % stride_x
        extra_height = (self.subscene_height - self.tile_height) % stride_y
        if scale:
            if scale == 'up':
                self.subscene_height += stride_y - extra_height
                self.subscene_width += stride_x - extra_width
            elif scale == 'down':
                self.subscene_height -= extra_height
                self.subscene_width -= extra_width
            else:
                raise Exception(f'Wrong scale option {scale}. Choose "None", "up" or "down".')
            print(f'Images in {"training" if train else "validation"} set will be scaled {scale} to ({self.subscene_height}, {self.subscene_width})')
        else:
            print(f'Extra pixels after tiling in {"training" if train else "validation"} set - width: {extra_height}, height: {extra_width}')

        # load dataset paths
        subscenes_path = os.path.join(dataset_path, 'subscenes')
        masks_path = os.path.join(dataset_path, 'masks')
        subscenes_filepath = []
        masks_filepath = []
        file_names = sorted(os.listdir(subscenes_path))  # sort so that train and val sets are reproducible on different file systems
        for file_name in file_names:
            subscene_filepath = os.path.join(subscenes_path, file_name)
            mask_filepath = os.path.join(masks_path, file_name)
            if os.path.isfile(subscene_filepath) and os.path.isfile(mask_filepath):
                subscenes_filepath.append(subscene_filepath)
                masks_filepath.append(mask_filepath)
        assert len(subscenes_filepath) == len(masks_filepath)

        # split
        if train:
            start = 0
            end = round(split * len(subscenes_filepath))
        else:
            start = round(split * len(subscenes_filepath))
            end = len(subscenes_filepath)
        print(f'{end-start} subscenes and masks in {"training" if train else "validation"} set.')

        # calculate tiles
        tiles_filepath, targets_filepath = [], []
        tiles_start = []
        for i in range(start, end):
            if i == self.dataset_limit:
                break
            tile_x, tile_y = 0, 0
            while tile_y + self.tile_height <= self.subscene_height:
                tiles_filepath.append(subscenes_filepath[i])
                targets_filepath.append(masks_filepath[i])
                tiles_start.append((tile_y, tile_x))
                tile_x += stride_x
                if tile_x + self.tile_width > self.subscene_width:
                    tile_x = 0
                    tile_y += stride_y
        assert len(tiles_filepath) == len(targets_filepath) == len(tiles_start)
        print(f'{len(tiles_filepath)} tiles in {"training" if train else "validation"} set.')

        # convert lists to numpy to prevent copy-on-access problem with multiple DataLoader workers https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        self.tiles_filepath = np.array(tiles_filepath)
        self.targets_filepath = np.array(targets_filepath)
        self.tiles_start = np.array(tiles_start)


    def __len__(self):
        return len(self.tiles_filepath)


    def __getitem__(self, idx):
        subscene_filepath = self.tiles_filepath[idx]
        mask_filepath = self.targets_filepath[idx]
        tile_y, tile_x = self.tiles_start[idx]

        subscene = np.load(subscene_filepath)
        mask = np.load(mask_filepath)
        assert subscene.shape[:2] == mask.shape[:2]

        resize = False if (self.subscene_height, self.subscene_width) == subscene.shape[:2] else True 
        if resize:
            # select channels
            subscene = subscene[..., self.bands]
            if self.binary: 
                mask = mask[..., 1]  # merging CLEAR and CLOUD_SHADOW classes to one
            mask = mask * 1.

            # resize
            #print(f'Resizing from {subscene.shape[:2]} to {(self.subscene_height, self.subscene_width)}')
            subscene = cv2.resize(subscene, (self.subscene_width, self.subscene_height), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (self.subscene_width, self.subscene_height), interpolation=cv2.INTER_NEAREST)
            
            # create patch
            tile = subscene[tile_y:tile_y+self.tile_height, tile_x:tile_x+self.tile_width, :]
            if self.binary:
                target = mask[tile_y:tile_y+self.tile_height, tile_x:tile_x+self.tile_width]
            else:
                target = mask[tile_y:tile_y+self.tile_height, tile_x:tile_x+self.tile_width, :]
                target = torch.argmax(torch.from_numpy(target), dim=-1).numpy()
        else:
            # select channels and create patch
            tile = subscene[tile_y:tile_y+self.tile_height, tile_x:tile_x+self.tile_width, self.bands]
            if self.binary:
                target = mask[tile_y:tile_y+self.tile_height, tile_x:tile_x+self.tile_width, 1]
            else:
                mask = mask * 1.
                target = mask[tile_y:tile_y+self.tile_height, tile_x:tile_x+self.tile_width, :]
                target = torch.argmax(torch.from_numpy(target), dim=-1).numpy()

        #print("Tile:", tile.shape)
        #print("Target", target.shape)

        if self.debug:
            tile_debug = np.copy(tile)
            tile_debug = tile_debug[...,:3]  # RGB
            tile_debug = tile_debug[...,::-1]  # RGB to BGR
            tile_debug = cv2.normalize(tile_debug, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # normalize to 0-255 values
            target_debug = target * 127  # to 0/255
            print(idx, subscene_filepath)
            cv2.imwrite(os.path.join(self.debug_folder, f'{idx}_tile.jpg'), tile_debug)
            cv2.imwrite(os.path.join(self.debug_folder, f'{idx}_target.jpg'), target_debug)
        
        if target.dtype == bool:
            target = target * 1.  # boolean to 0/1

        if self.transform:
            augmented = self.transform(image=tile, mask=target)
            tile = augmented['image']
            target = augmented['mask']
            if self.binary:
                target = target.unsqueeze(0)
        else:
            tile = tile.transpose(2, 0, 1)  # HWC to CHW, already done by ToTensorV2() in self.transform
            target = np.expand_dims(target, axis=0)
        
        return tile, target.long()

