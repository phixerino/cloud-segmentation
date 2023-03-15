# cloud-segmentation

Cloud segmentation on satellite images from [Sentinel-2 Cloud Mask Catalogue](https://zenodo.org/record/4172871), written in PyTorch. DataLoader is implemented for
binary segmentation with CLOUD and CLEAR/CLOUD_SHADOW classes. Training script is also capable of multiclass segmentation, but DataLoader would have to be modified.
By default this repo assumes 4 (RGB + NIR) channels, but this can be changed through arguments/config.


## Install

```
pip install -r requirements.txt
```

or build docker image and run container with:

```
make build
make
```
You might have to change base image according to your CUDA and cuDNN versions. If you want to do training, then also change *LOCAL_DATASET* in Makefile to your
dataset path.


## Inference

Inference is accelerated with ONNX Runtime. You can download model [here](https://drive.google.com/file/d/1HLpewT9vKwMc9Vy4IJ9f3OteqqAy_oi5/view?usp=share_link)
and then run inference:

```
python inference.py --model DeepLabV3Plus_resnet101_1678896432.onnx --source data/test_subscene.npy --save --out_folder data/preds/
```

Predicted masks will be saved in *out_folder*. You can also plot masks with *--show*. 

Short example of how to do inference in Google Colab is in 
[notebooks/sentinel_segmentation_inference.ipynb](https://github.com/phixerino/cloud-segmentation/blob/main/notebooks/sentinel_segmentation_inference.ipynb)


## Training
Download and unzip [subscenes.zip](https://zenodo.org/record/4172871/files/subscenes.zip?download=1) and
[masks.zip](https://zenodo.org/record/4172871/files/masks.zip?download=1) to your dataset path. This path can be changed in
[cfg/config.json](https://github.com/phixerino/cloud-segmentation/blob/main/cfg/config.json) under *dataset_path*.

### Data loading

Images can be loaded with different tiling strategies by changing these settings:
- subscene_width, subscene_height - manually resize subscenes and masks before tiling
- train_tile_stride_x, train_tile_stride_y, val_tile_stride_x, val_tile_stride_y - lower stride than tile height/width will lead to overlap
- train_scale, val_scale - automatically scale subscenes and masks so that tiles fit the entire image. This is done after manual resizing with subscene_width/subscene_height. Allowed values are [None, 'down', 'up']

### Train

Modify [cfg/config.json](https://github.com/phixerino/cloud-segmentation/blob/main/cfg/config.json) to your liking and run:
```
python train.py
```
You can also override any setting through command-line arguments without modifying the config file, for example:
```
python3 train.py --epochs 100 --batch_size 128 --lr 0.01 --optimizer AdamW --scheduler cos --warmup_epochs 5 --decoder_name UnetPlusPlus --encoder_name resnet50 --loss CE --no_wandb_log
```

Example of training progress:

<img src="https://github.com/phixerino/cloud-segmentation/blob/main/data/W%26B%20Chart%203_15_2023%2C%2010_21_04%20PM.png" width="375" height="225"> <img src="https://github.com/phixerino/cloud-segmentation/blob/main/data/W%26B%20Chart%203_15_2023%2C%2010_20_05%20PM.png" width="375" height="225"> <img src="https://github.com/phixerino/cloud-segmentation/blob/main/data/W%26B%20Chart%203_15_2023%2C%2010_20_19%20PM.png" width="375" height="225"> <img src="https://github.com/phixerino/cloud-segmentation/blob/main/data/W%26B%20Chart%203_15_2023%2C%2010_22_12%20PM.png" width="375" height="225">


### Results

| Model | mIoU |
| --- | --- |
| DeepLabV3+ with ResNet101 | 86.43 |
| Unet++ with ResNet101 | 84.55 |

The [DeepLabV3+ model with ResNet101 encoder](https://drive.google.com/file/d/1HLpewT9vKwMc9Vy4IJ9f3OteqqAy_oi5/view?usp=share_link) was trained with these settings:
- pretrained encoder on ImageNet
- 9:1 train/val split
- 50 epochs
- 20 epochs early stop
- 64 batch size
- AdamW optimizezr
- 0.0005 learning rate
- 0.005 weight decay
- 3 linear warmup epochs
- linear lr scheduler
- Dice loss
- mean IoU val metric
- augmentations: rotation (max 60 degrees, probability 0.5), horizontal flip (probability 0.5), vertical flip (probability 0.5)


### TODO
- Resume training
- Automated hyperparameter tuning
- DistributedDataParallel (DDP) training

DDP can be enabled, but it's not working properly for now:
```
torchrun --standalone --nproc_per_node 2 train.py
```


## Export

After training, the PyTorch model can be exported to ONNX with:
```
python export.py --model_file weights/my_model.pt
```
