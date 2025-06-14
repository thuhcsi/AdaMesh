import os
import glob
import numpy as np
import torch
import argparse
from rotation_converter import batch_euler2axis


def calc_diversity(feats):
    feat_array = np.array(feats)
    n, c = feat_array.shape
    diff = np.array([feat_array] * n) - feat_array.reshape(n, 1, c)
    return np.sqrt(np.sum(diff**2, axis=2)).sum() / n / (n-1)

def calculate_avg_distance(feature_list, mean=None, std=None):
    # feature_list = np.stack(feature_list)
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

def normalize(feat, feat2):
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)
    
    return (feat - mean) / (std + 1e-10), (feat2 - mean) / (std + 1e-10)

def calc_standard_deviation(pose):
    """[T, 3]"""
    list_std = [pose[:, i].std() for i in range(pose.shape[1])]
    return np.array(list_std).mean()

def load_pose(pose_path):
    ext = os.path.splitext(pose_path)[-1]
    if ext == ".npy":
        pose = np.load(pose_path)
        pi = torch.Tensor([3.14159265358979323846])
        pose = torch.FloatTensor(pose) / 180. * pi
        pose = batch_euler2axis(pose).numpy()
    elif args.extension == ".pt":
        pose = torch.load(pose_path)["pose"][:, :3].numpy()
    return pose


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--param_path", type=str, default="/cfs2/user/chenliyang/Research/ICLR2024/method_compare/AdaMesh/pose/id01224#bhPid-mDK38#00227.npy")
    parser.add_argument("--param_dir", type=str, default="/cfs2/user/chenliyang/Project/HeadPoseEstimate/HeadPoseEstimate/ablation/mel_hubert_randid")
    parser.add_argument("--extension", type=str, default=".npy")
    args = parser.parse_args()

    list_pose_path = glob.glob(os.path.join(args.param_dir, f"*{args.extension}"))

    # list_metric = []
    # for pose_path in list_pose_path:
    #     if args.extension == ".npy":
    #         pose = np.load(pose_path)
    #     elif args.extension == ".pt":
    #         pose = torch.load(pose_path)["pose"][:, :3].numpy()
    #     div_pose = calculate_avg_distance(pose)
    # list_metric.append(div_pose)

    # print(f"pose diversity {np.array(list_metric).mean()}.")

    list_pose = [load_pose(pose_path) for pose_path in list_pose_path if os.path.basename(pose_path)[:4] != "M007"]
    list_pose = np.concatenate(list_pose, axis=0)
    div_metric = calc_diversity(list_pose)
    lsd_metric = calc_standard_deviation(list_pose)
    print(f"pose diversity {div_metric}, lsd {lsd_metric}.")