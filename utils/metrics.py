import numpy as np

from utils.general import add_dim, concat

def intersection_over_union(preds, labels, smooth=1e-1):
    intersection = (preds * labels).sum((2, 3))
    union = (preds + labels).sum((2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    class_iou = iou.mean(axis=0)
    return class_iou

def process_intersection_over_union(preds, labels, smooth=1e-1, binary=False):
    dims_num = [3, 4]  # BHW or BCHW
    assert len(preds.shape) in dims_num and len(labels.shape) in dims_num
    
    if len(preds.shape) == 3:
        preds = add_dim(preds, dim=1)
    if len(labels.shape) == 3:
        labels = add_dim(labels, dim=1)

    class_iou = intersection_over_union(preds, labels, smooth=smooth)

    if binary:
        class2_iou = intersection_over_union(1 - preds, 1 - labels, smooth=smooth)
        class_iou = concat((class_iou, class2_iou))

    return class_iou

def mean_intersection_over_union(preds, labels, smooth=1e-1, binary=False):
    iou = process_intersection_over_union(preds, labels, smooth, binary)
    return iou.mean()

