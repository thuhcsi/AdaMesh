import os
import glob
import argparse
import numpy as np
import torch


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


def main(raw_args=None):
    args = parser(raw_args)

    list_param_path = glob.glob(os.path.join(args.param_dir, "**/*_3dparams.pt"), recursive=True)
    print("Detected %d files." % (len(list_param_path)))
    # exit()

    param_data = []
    for param_path in list_param_path:
        param = load_param(param_path)
        param_data.append(param)
    print("Loaded %d files." % (len(param_data)))
    stcs_dict, _ = stcs_cal(param_data, dim=param_data[0].shape[-1])

    np.save(args.save_path, stcs_dict)
    


def parser(raw_args):
    parser = argparse.ArgumentParser(description='Caculate statistical info and save.')
    parser.add_argument('--param_dir', type=str, required=True,
                        default='/disk3/liyangchen/Data/weiwei/avena_exp_full',
                        help='Folder contains data to be normalized.')
    parser.add_argument('--save_path', type=str, required=True,
                        default='/disk3/liyangchen/Data/weiwei/statistics_avena_exp_full.npy',
                        help='The path to save result.')        
    args = parser.parse_args(raw_args)
    
    return args


if __name__ == '__main__':
    '''
    Caculate statistical infomation and save.
    '''

    main()