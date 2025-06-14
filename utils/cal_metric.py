import os
import glob
import numpy as np
import torch
import argparse


def calc_diversity(feats):
    feat_array = np.array(feats)
    n, c = feat_array.shape
    diff = np.array([feat_array] * n) - feat_array.reshape(n, 1, c)
    return np.sqrt(np.sum(diff**2, axis=2)).sum() / n / (n-1)

def calculate_avg_distance(feature_list, mean=None, std=None):
    feature_list = np.stack(feature_list)
    n = feature_list.shape[0]
    # normalize the scale
    if (mean is not None) and (std is not None):
        feature_list = (feature_list - mean) / std
    dist = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist += np.linalg.norm(feature_list[i] - feature_list[j])
    dist /= (n * n - n) / 2
    return dist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--param_path", type=str, default="/cfs2/user/chenliyang/Research/ICLR2024/method_compare/AdaMesh/face_param/M007_front_happy_level_3_007_predict.pt")
    args = parser.parse_args()

    statiscs_path = "/cfs2/user/chenliyang/Data/deca_param_statistics.npy"
    stcs_dict = np.load(statiscs_path, allow_pickle=True).item()
    # import pdb; pdb.set_trace()
    mean = np.concatenate((stcs_dict['mean'][:50], stcs_dict['mean'][53:56]))
    std = np.concatenate((stcs_dict['std'][:50], stcs_dict['std'][53:56]))

    param = torch.load(args.param_path)
    expression = param["exp"].numpy()
    pose = param["pose"].numpy()
    if pose.shape[1] > 3:
        pose = param["pose"].numpy()[:, 3:]

    motion = np.concatenate((expression, pose), axis=1)
    motion = (motion - mean) / std
    div_exp = calc_diversity(motion[:53])
    print(f"expression diversity {div_exp}.")
