import os
import argparse
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

def rect_iou(rect1, rect2):
    
    assert rect1.shape == rect2.shape
    rects_inter = _intersection(rect1, rect2)
    areas_inter = np.prod(rects_inter[..., 2:], axis=1)
    
    areas1 = np.prod(rect1[..., 2:], axis=-1)
    areas2 = np.prod(rect2[..., 2:], axis=-1)
    areas_union = areas1 + areas2 - areas_inter

    ious = areas_inter / areas_union
    ious = np.clip(ious, 0.0, 1.0)

    return ious

def _intersection(rect1, rect2):
    assert rect1.shape == rect2.shape
    x1 = np.maximum(rect1[..., 0], rect2[..., 0])
    y1 = np.maximum(rect1[..., 1], rect2[..., 1])
    x2 = np.minimum(rect1[..., 0] + rect1[..., 2],
                    rect2[..., 0] + rect2[..., 2])
    y2 = np.minimum(rect1[..., 1] + rect1[..., 3],
                    rect2[..., 1] + rect2[..., 3])
    
    w = np.maximum(x2 - x1, 0)
    h = np.maximum(y2 - y1, 0)

    return np.stack([x1, y1, w, h]).T

def evaluate(ious):
    ao = np.mean(ious) * 100        # area overlap
    sr = np.mean(ious > 0.5) * 100  # success rate when iou > 0.5
    return ao, sr

def main():
    parser = argparse.ArgumentParser(description='Get file from GT and Prediction')
    parser.add_argument('--gt', type=str, default='groundtruth_rect.txt', help='groundtruth file')
    parser.add_argument('--pd', type=str, default='tracking_rect.txt', help='predict file')
    
    args = parser.parse_args()
    
    gt_boxes = np.loadtxt(args.gt, delimiter=',')
    tr_boxes = np.loadtxt(args.pd, delimiter=',')
    assert (gt_boxes.shape == tr_boxes.shape)

    # Calculate IoU to find AO and SR
    seq_ious = rect_iou(tr_boxes, gt_boxes)
    ao, sr = evaluate(seq_ious)
    print("Average Overlap (AO): {:.2f} %".format(ao))
    print("Success (SR): {:.2f} %".format(sr))
    

if __name__ == '__main__':
    main()