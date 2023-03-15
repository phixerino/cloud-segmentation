import numpy as np


def mean_intersection_over_union(preds, labels, smooth=1e-1, binary=True):
    dims_num = [3, 4]  # BHW or BCHW
    assert len(preds.shape) in dims_num and len(labels.shape) in dims_num
    
    if len(preds.shape) == 3:
        preds = np.expand_dims(preds, axis=1)
    if len(labels.shape) == 3:
        labels = np.expand_dims(labels, axis=1)

    def intersection_over_union(preds, labels):
        intersection = (preds * labels).sum((2, 3))
        union = (preds + labels).sum((2, 3)) - intersection
        iou = (intersection + smooth) / (union + smooth)
        class_iou = iou.mean(axis=0)
        return class_iou
    
    class_iou = intersection_over_union(preds, labels)
    
    if binary:
        preds_rev = np.where(preds==0., 1., 0.)
        labels_rev = np.where(labels==0., 1., 0.)
        class2_iou = intersection_over_union(preds_rev, labels_rev)
        class_iou = np.concatenate((class_iou, class2_iou))
    
    miou = class_iou.mean()

    return miou

