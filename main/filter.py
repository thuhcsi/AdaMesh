import os
import glob
import argparse
import numpy as np
import torch


def read_txt(txt_path):
    """Read .txt

    Args:
        txt-path (str): Save path of .txt.
    Return:
        content (list[str]): Each element in list corresponds to a single line in .txt.
    """
    with open(txt_path) as f:
        content = [line.strip() for line in f]
    return content


def write_txt(content, txt_path):
    """Write to .txt

    Args:
        content (list[str]): Each element in list corresponds to a single line in .txt.
        txt-path (str): Save path of .txt.
    """
    with open(txt_path, "w") as f:
        f.write("\n".join(content))

def load_param(param_path):
    param = torch.load(param_path)
    expression = param["exp"].numpy()
    pose = param["pose"].numpy()
    cam = param["cam"].numpy()
    cam[:, 0] = 1 / cam[:, 0]
    motion = np.concatenate((expression, pose, cam), axis=1)

    return motion


def stcs_cal(video_feats, stcs_path=None, dim=None):
    '''
    Use this if normalizing data is needed, but input shape is different in a batch
    Args:
        video_feats: numpy.ndarray or list [[T1, 53], [T2, 53], ...] 
        stcs_path: given statistics {min, max, mean, std}
    Return:
        res: stcs_dict, normalized data
    '''
    if dim is None:
        if isinstance(video_feats, list) or len(video_feats.shape) > 2:
            dim = video_feats[0].shape[1]
        else:
            dim = video_feats.shape[1]

    if stcs_path is None:
        # min, max, mean, std calculation
        mus, sigmas, mins, maxs = [], [], [], []
        for i in range(dim):
            i_feat = [v[:, i] for v in video_feats]  # [[T1,] [T2,], ...] each is numpy.ndarray
            i_feat = np.concatenate(i_feat, axis=0)

            mins.append(i_feat.min())
            maxs.append(i_feat.max())
            mus.append(i_feat.mean())
            sigmas.append(i_feat.std())

        # store this if needed
        stcs_dict = {
            'min': mins,
            'max': maxs,
            'mean': mus,
            'std': sigmas
        }
    else:
        stcs_dict = np.load(stcs_path, allow_pickle=True).item()
        mus = stcs_dict['mean']
        sigmas = stcs_dict['std']
        
    # normalize data
    normalized_feats = []
    if isinstance(video_feats, list) or len(video_feats.shape) > 2:
        for f in video_feats:
            feat = (f - mus) / sigmas  # ([T,53] - [1,53]) / [1,53]
            normalized_feats.append(feat)
        normalized_feats = np.array(normalized_feats, dtype='object')
    else:
        normalized_feats = (video_feats - mus) / sigmas

    return stcs_dict, normalized_feats


def inv_norm(video_feats, stcs_path=None):
    '''
    inverse data normalization according to the given statistics
    '''
    stcs_dict = np.load(stcs_path, allow_pickle=True).item()
    mus = stcs_dict['mean']
    sigmas = stcs_dict['std']

    # inv-normalize data
    inv_feats = sigmas * video_feats + mus  # [1,53] * [T,53] + [1,53]

    return inv_feats


def filter_data(data_dir, file_path, save_path):
    list_res = []

    list_itemname = read_txt(file_path)
    for itemname in list_itemname:
        motion_path = os.path.join(data_dir, "deca_feature", itemname, itemname+"_3dparams.pt")
        speech_path = os.path.join(data_dir, "hubert_feature", itemname+"_hubert.npy")
        
        motion = load_param(motion_path)
        speech_feat = np.load(speech_path)
        
        if abs(speech_feat.shape[0] // 2 - motion.shape[0]) < 5:
            list_res.append(itemname)
        else:
            print(f"wrong length {itemname}")
    write_txt(list_res, save_path)



def main(raw_args=None):
    data_dir = "/cfs2/user/chenliyang/Data/MEAD_processed"
    train_path = f"{data_dir}/deca_feature/file_list_train.txt"
    test_path = f"{data_dir}/deca_feature/file_list_test.txt"

    filter_train_path = f"{data_dir}/deca_feature/file_list_train_filter.txt"
    filter_test_path = f"{data_dir}/deca_feature/file_list_test_filter.txt"

    filter_data(data_dir, train_path, filter_train_path)
    filter_data(data_dir, test_path, filter_test_path)



if __name__ == '__main__':
    '''
    Caculate statistical infomation and save.
    '''

    main()