from connectomics.utils.evaluate import confusion_matrix_visualize
from connectomics.utils.evaluate import cremi_distance
import argparse
from connectomics.data.utils.data_io import readvol
import tifffile as tf
import numpy as np


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='path to gt and prediction')
    parser.add_argument('--gt_path', type=str, help='gt file path')
    parser.add_argument('--pre_path', type=str, help='prediction file path')
    args = parser.parse_args()

    gt = readvol(args.gt_path)
    pred = readvol(args.pre_path)[0]
    b_gt = gt > 0
    vis = confusion_matrix_visualize(pred, b_gt)
    vis = vis.astype(np.uint8)
    tf.imsave(args.pre_path.split('.')[0]+'_vis'+'.tif', vis)
    print('Computing cremi distance')
    b_pred = pred > 0.5
    cremi_distance(b_pred, b_gt)