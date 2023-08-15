import os
import argparse
import json
import time
import math
import functools
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import wandb

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import segmentation_models_pytorch as smp

from datasets import Sentinel2Dataset
from utils.metrics import mean_intersection_over_union
from utils.general import AttrDict
from utils.schedulers import CosineDecay, LinearDecay, NoneDecay



def train_epoch(model, data_loader, loss_fn, optimizer, scaler, scheduler, dataset_len, device):
    model.train()
    
    running_loss = 0.
    for (tiles, labels) in tqdm(data_loader):
        tiles = tiles.to(device)
        labels = labels.to(device)

        scheduler.step()

        with torch.cuda.amp.autocast():
            preds = model(tiles)
            loss = loss_fn(preds, labels)
            running_loss += loss.item() * tiles.size(0)
            
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()  # loss.backward()
        scaler.step(optimizer)  # optimizer.step()
        scaler.update()

    epoch_loss = running_loss / dataset_len

    return epoch_loss


def val_epoch(model, data_loader, loss_fn, num_classes, metric_fn, dataset_len, device):
    model.eval()
    
    running_loss = 0.
    running_metric = 0.
    with torch.no_grad():
        for (tiles, labels) in tqdm(data_loader):
            tiles = tiles.to(device)
            labels = labels.to(device)
            preds = model(tiles)
            loss = loss_fn(preds, labels)
            running_loss += loss * tiles.size(0)

            if num_classes == 1:
                threshold = 0.0  # expects logit outputs
                preds_mask = ((preds>threshold).float())
            else:
                preds_mask = torch.argmax(preds, dim=1)
                preds_mask = F.one_hot(preds_mask, num_classes=num_classes).transpose(1, 3)  # Same shape as model output
                labels = F.one_hot(labels, num_classes=num_classes).transpose(1, 3)     # Same shape as preds
            
            metric = metric_fn(preds_mask, labels).item()
            running_metric += metric * labels.size(0)

    epoch_loss = running_loss / dataset_len
    epoch_metric = running_metric / dataset_len

    return epoch_loss, epoch_metric


def train(epochs_num, model, train_loader, val_loader, loss_fn, metric_fn, optimizer, scaler, scheduler, num_classes, train_dataset_len, val_dataset_len,
        master_process=True, ddp=False, wandb_log=False, early_stop_patience=0, model_filepath='model.pt', config=None, device='cuda:0'):
    
    loss_name, metric_name = 'loss', 'metric'
    if isinstance(loss_fn, dict):
        loss_name = loss_fn['loss_name'] + ' loss'
        loss_fn = loss_fn['loss_fn']
    if isinstance(metric_fn, dict):
        metric_name = metric_fn['metric_name']
        metric_fn = metric_fn['metric_fn']

    checkpoint = None
    best_val_metric = 0
    best_epoch = 0
    early_stop_counter = 0
    if not early_stop_patience: 
        early_stop_patience = epochs_num
    for epoch in range(epochs_num):
        if master_process:
            print(f'Epoch {epoch+1}')
        
        if ddp:
            train_loader.sampler.set_epoch(epoch)
        epoch_loss = train_epoch(model, train_loader, loss_fn, optimizer, scaler, scheduler, train_dataset_len, device)

        if master_process:
            val_epoch_loss, val_epoch_metric = val_epoch(model, val_loader, loss_fn, num_classes, metric_fn, val_dataset_len, device)

            print(f'Train {loss_name}: {epoch_loss:.4f}')
            print(f'Validation {loss_name}: {val_epoch_loss:.4f}, {metric_name}: {val_epoch_metric:.4f}')

            if wandb_log:
                for param_group in optimizer.param_groups:
                    current_lr = param_group['lr']
                wandb.log({
                    'epoch': epoch+1,
                    'train_loss': epoch_loss,
                    'val_loss': val_epoch_loss,
                    f'val_{metric_name}': val_epoch_metric,
                    'lr': current_lr
                    })           

            if val_epoch_metric > best_val_metric:
                best_val_metric = val_epoch_metric
                best_epoch = epoch
                early_stop_counter = 0
                checkpoint = deepcopy(model.state_dict())
            else:
                early_stop_counter += 1
                if early_stop_counter == early_stop_patience:
                    print('Early stopping')
                    break

    if master_process:
        print(f'Best metric in epoch {best_epoch+1} {best_val_metric:.4f}')
        
        if checkpoint:
            if not wandb_log:
                model_filename, model_ext = os.path.splitext(model_filepath)
                val_metric_str = "{:4f}".format(best_val_metric).replace(".", ",")
                model_filepath = f'{model_filename}_{val_metric_str}{model_ext}'
            torch.save(checkpoint, model_filepath)
            print(f'Model saved to {model_filepath}')
            if config:
                config_filepath = f'{os.path.splitext(model_filepath)[0]}.json'
                with open(config_filepath, 'w') as f:
                    json.dump(config, f, indent=4)

    if ddp:
        torch.distributed.destroy_process_group()


def main(config):
    # DDP
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.distributed.init_process_group(backend='nccl')
        #device = f'cuda:{ddp_local_rank}'
        device = ddp_local_rank
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        master_process = True

    # transforms
    assert len(config.bands) == len(config.normalize_mean) == len(config.normalize_std)
    train_transform = A.Compose([
        A.Rotate(limit=config.rotate_limit, p=config.rotate_prob),
        A.HorizontalFlip(p=config.horizontal_flip_prob),
        A.VerticalFlip(p=config.vertical_flip_prob),
        A.Normalize(mean=config.normalize_mean, std=config.normalize_std, max_pixel_value=1.0),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Normalize(mean=config.normalize_mean, std=config.normalize_std, max_pixel_value=1.0),
        ToTensorV2()
    ])

    # dataset loader
    train_dataset = Sentinel2Dataset(config.dataset_path, train=True, split=config.split, transform=train_transform, tile_height=config.tile_height, tile_width=config.tile_width,
            stride_y=config.train_tile_stride_y, stride_x=config.train_tile_stride_x, scale=config.train_scale, subscene_width=config.subscene_width, subscene_height=config.subscene_height,
            dataset_limit=config.dataset_limit, debug=config.debug)
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if ddp else None
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=config.loader_num_workers, pin_memory=True)

    val_dataset = Sentinel2Dataset(config.dataset_path, train=False, split=config.split, transform=val_transform, tile_height=config.tile_height, tile_width=config.tile_width,
            stride_y=config.val_tile_stride_y, stride_x=config.val_tile_stride_x, scale=config.val_scale, subscene_width=config.subscene_width, subscene_height=config.subscene_height)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.loader_num_workers)

    # model
    '''
    if the encoder is pretrained and number of input channels isnt 3, then weights of first convolutional layer are reused like this:
    pretrained weights in (i%3)-th channel are copied to i-th channel and then all weights are scaled: w * 3 / in_channels
    '''
    available_archs = [attribute for attribute in dir(smp) if isinstance(getattr(smp, attribute), type)]
    if config.decoder_name in available_archs:
        model_class = getattr(smp, config.decoder_name)
    else:
        raise Exception(f'Decoder with name {config.decoder_name} isnt supported. Choose one of the following: {available_archs}')
    model = model_class(encoder_name=config.encoder_name, encoder_weights='imagenet' if config.pretrained_encoder else None, in_channels=len(config.bands), classes=config.num_classes)
    
    model.to(device)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # loss function - all loss functions expect logits
    available_losses = ['CE', 'Dice']
    if config.loss_fn_name == 'CE':
        if config.num_classes == 1:
            loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
    elif config.loss_fn_name == 'Dice':
        mode = 'binary' if config.num_classes == 1 else 'multiclass'
        loss_fn = smp.losses.DiceLoss(mode, from_logits=True)
    else:
        raise Exception(f'Loss function with name {config.loss_func_name} isnt supported. Choose one of the following: {available_losses}')
    loss_dict = {'loss_name': config.loss_fn_name, 'loss_fn': loss_fn}

    # metric
    print((config.num_classes==1))
    metric_fn = functools.partial(mean_intersection_over_union, binary=(config.num_classes==1))  # freeze binary argument to keep metric_fn general
    metric_dict = {'metric_name': 'mIoU', 'metric_fn': metric_fn}

    # optimizer
    available_optimizers = ['Adam', 'AdamW', 'AMSGrad', 'SGD', 'Nesterov']
    if config.optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(config.momentum, 0.999), weight_decay=config.weight_decay)
    elif config.optimizer_name == 'AdamW' or config.optimizer_name == 'AMSGrad':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, betas=(config.momentum, 0.999), weight_decay=config.weight_decay, amsgrad=(config.optimizer_name=='AMSGrad'))
    elif config.optimizer_name == 'SGD' or config.optimizer_name == 'Nesterov':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay, nesterov=(config.optimizer_name=='Nesterov'))
    else:
        raise Exception(f'Optimizer with name {config.optimizer_name} isnt supported. Choose one of the following: {available_optimizers}')

    # automatic mixed precision
    scaler = torch.cuda.amp.GradScaler()

    # lr scheduler
    epoch_iterations = len(train_loader)
    warmup_iterations = epoch_iterations * config.warmup_epochs
    decay_iterations = epoch_iterations * config.epochs_num
    available_schedulers = ['cos', 'linear', None]
    if config.lr_scheduler_name == 'cos':  # cosine with linear warmup
        scheduler = CosineDecay(optimizer, config.learning_rate, warmup_iterations, decay_iterations, config.lr_start_coeff, config.lr_final_coeff)
    elif config.lr_scheduler_name == 'linear':  # linear with linear warmup
        scheduler = LinearDecay(optimizer, config.learning_rate, warmup_iterations, decay_iterations, config.lr_start_coeff, config.lr_final_coeff)
    elif config.lr_scheduler_name is None:
        scheduler = NoneDecay(optimizer, config.learning_rate, lr_start_coeff=1.0, lr_final_coeff=1.0)
    else:
        raise Exception(f'Learning rate scheduler {config.lr_scheduler_name} isnt supported. Choose one of the following: {available_schedulers}')

    # save and logging
    os.makedirs(config.model_folder, exist_ok=True)
    model_name = f'{config.decoder_name}_{config.encoder_name}.pt'  # default name, if wandb logging is enabled, then current time is added to the name, else validation metric will be added
    if config.wandb_log and master_process:
        wandb_run = f'{config.decoder_name}_{config.encoder_name}_{int(time.time())}'
        model_name = f'{wandb_run}.pt'
        wandb.init(project=config.wandb_project, name=wandb_run, config=config)
    model_filepath = os.path.join(config.model_folder, model_name)

    # training
    train(config.epochs_num, model, train_loader, val_loader, loss_dict, metric_dict, optimizer, scaler, scheduler, config.num_classes, len(train_dataset), len(val_dataset),
            master_process, ddp, config.wandb_log, config.early_stop_patience, model_filepath, config, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # config
    parser.add_argument('--config_filepath', '--config', type=str, default='cfg/config.json', help='Config with settings.')

    # dataset settings
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--split', type=float, help='Train/validation split, e.g. 0.9 means 9:1 split.')
    parser.add_argument('--dataset_limit', type=int, help='Subcenes limit for training dataset. This is meant for testing overfitting. None to load entire training dataset.')
    parser.add_argument('--debug', action='store_true', default=None, help='Save tiles and their masks from DataLoader as JPEG images to temp folder.')
    parser.add_argument('--no_debug', action='store_false', dest='debug', help='Disable debug')
    parser.add_argument('--subscene_width', type=int, help='Resize subscenes and masks before tiling to this width.')
    parser.add_argument('--subscene_height', type=int, help='Resize subscenes and masks before tiling to this height.')
    parser.add_argument('--tile_height', type=int)
    parser.add_argument('--tile_width', type=int)
    parser.add_argument('--train_tile_stride_y', type=int, help='Stride while creating train tiles. Lower stride than tile_height will lead to overlap.')
    parser.add_argument('--train_tile_stride_x', type=int, help='Stride while creating train tiles. Lower stride than tile_width will lead to overlap.')
    parser.add_argument('--val_tile_stride_y', type=int, help='Stride while creating validation tiles. Lower stride than tile_height will lead to overlap.')
    parser.add_argument('--val_tile_stride_x', type=int, help='Stride while creating validation tiles. Lower stride than tile_width will lead to overlap.')
    parser.add_argument('--train_scale', choices=[None, 'up', 'down'], help='Scale train subscenes and masks so that tiles fit the entire image. This is done after optional resizing with' \
                         'subscene_width and subscene_height. None will not do any scaling, "up" will scale to the nearest bigger resolution and "down" to the nearest smaller resolution.')
    parser.add_argument('--val_scale', choices=[None, 'up', 'down'], help='Scale validation subscenes and masks so that tiles fit the entire image. This is done after optional resizing with' \
                        'subscene_width and subscene_height. None will not do any scaling, "up" will scale to the nearest bigger resolution and "down" to the nearest smaller resolution.')
    parser.add_argument('--bands', nargs='+', type=int, help='Subscene bands. Usage: --bands 3 2 1 7')
    parser.add_argument('--normalize_mean', nargs='+', type=float)
    parser.add_argument('--normalize_std', nargs='+', type=float)
    parser.add_argument('--rotate_limit', type=int, help='Rotate degrees.')
    parser.add_argument('--rotate_prob', type=float, help='Rotate probability.')
    parser.add_argument('--horizontal_flip_prob', type=float, help='Horizontal flip probability.')
    parser.add_argument('--vertical_flip_prob', type=float, help='Vertical flip probability.')
    
    # loader settings
    parser.add_argument('--batch_size', '--batch', type=int)
    parser.add_argument('--loader_num_workers', type=int)

    # training settings
    parser.add_argument('--epochs_num', '--epochs', type=int)
    parser.add_argument('--learning_rate', '--lr', type=float)
    parser.add_argument('--momentum', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--optimizer_name', '--optimizer', type=str, choices=['Adam', 'AdamW', 'AMSGrad', 'SGD', 'Nesterov'])
    parser.add_argument('--lr_scheduler_name', '--scheduler', type=str, choices=['linear', 'cos', None])
    parser.add_argument('--lr_final_coeff', type=float, help='If lr scheduler is enabled, then learning rate will reach learning_rate*lr_final_coeff at the last epoch.')
    parser.add_argument('--lr_start_coeff', type=float, help='If warmup is enabled, then learning rate will begin at learning_rate*lr_start_coeff and then increase linearly to learning_rate.')
    parser.add_argument('--warmup_epochs', type=int, help='Linear learning rate warmup')
    parser.add_argument('--early_stop_patience', type=int)

    # model settings
    parser.add_argument('--decoder_name', '--decoder', type=str, help='Model architecture, e.g. "Unet", "UnetPlusPlus", "DeepLabV3", "DeepLabV3Plus"')
    parser.add_argument('--encoder_name', '--encoder', type=str, help='Architecture of encoder, e.g. "resnet50". All available architectures can be found through segmentation_models_pytorch module:' \
                                                        '"segmentation_models_pytorch.encoders.encoders.keys()"')
    parser.add_argument('--num_classes', type=int, help='1 class is binary output, so 2 classes')
    parser.add_argument('--loss_fn_name', '--loss_name', '--loss', type=str, choices=['CE', 'Dice'])
    parser.add_argument('--pretrained_encoder', action='store_true', default=None, help='Use pretrained encoder on ImageNet')
    parser.add_argument('--no_pretrained_encoder', action='store_false', dest='pretrained_encoder', help='Dont use pretrained encoder')

    # logging settings
    parser.add_argument('--wandb_log', '--wandb', action='store_true', default=None)
    parser.add_argument('--no_wandb_log', '--no_wandb', action='store_false', dest='wandb_log')
    parser.add_argument('--wandb_project', type=str)
    
    # save settings
    parser.add_argument('--model_folder', type=str, help='Folder for saving model.')
    args = parser.parse_args()

    # load config
    with open(args.config_filepath, 'r') as f:
        config_dict = json.load(f)
    config = AttrDict(config_dict)
    
    # override config settings with command-line arguments
    args_dict = vars(args)
    for key, value in args_dict.items():
        if args_dict[key] is not None and key != 'config_filepath':
            config[key] = value
    print(f'Config:\n{json.dumps(config, indent=4)}')

    main(config)

