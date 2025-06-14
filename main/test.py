#!/usr/bin/env python
import os
import glob
import torch
import numpy as np
import cv2
from tqdm import tqdm

import soundfile as sf
from base.utilities import get_parser, get_logger
from models import get_model
from base.baseTrainer import load_state_dict
from stcs_save import inv_norm
from torch.utils.data import Dataset

from main.process_audio_hubert import get_hubert_from_16k_speech


class TestDataset(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, args):        
        self.list_audio_path = glob.glob(os.path.join(args.test_data_dir, "**/*.wav"), recursive=True)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        audio_path = self.list_audio_path[index]
        speech_16k, _ = sf.read(audio_path)
        hubert_hidden = get_hubert_from_16k_speech(speech_16k, device="cuda")
        input_len = hubert_hidden.shape[0] // 2 * 2

        hubert_feature = torch.FloatTensor(hubert_hidden[:input_len].detach().cpu().numpy()).unsqueeze(0)
        input_len = torch.LongTensor([input_len])

        return {"audio_path": audio_path, "hubert_feature": hubert_feature, "input_len": input_len}

    def __len__(self):
        return len(self.list_audio_path)



def main():
    cfg = get_parser()
    os.makedirs(cfg.test_save_dir, exist_ok=True)
    logger = get_logger(cfg.test_save_dir)
    logger.info(cfg)
    logger.info("=> creating model ...")
    model = get_model(cfg)
    model = model.cuda()

    if os.path.isfile(cfg.model_path):
        checkpoint = torch.load(cfg.model_path, map_location=lambda storage, loc: storage.cpu())
        load_state_dict(model, checkpoint['state_dict'])
        logger.info("=> loaded checkpoint '{}'".format(cfg.model_path))
    else:
        raise RuntimeError("=> no checkpoint flound at '{}'".format(cfg.model_path))

    test_loader = TestDataset(cfg)

    test(model, test_loader, cfg)


def test(model, test_loader, cfg):
    model.eval()
    save_folder = cfg.test_save_dir
    os.makedirs(save_folder, exist_ok=True)

    stcs_dict = np.load(cfg.statiscs_path, allow_pickle=True).item()
    mean, std = stcs_dict['mean'], stcs_dict['std']

    mean = np.concatenate([stcs_dict["mean"][:50], stcs_dict["mean"][53:56]])
    std = np.concatenate([stcs_dict["std"][:50], stcs_dict["std"][53:56]])

    result = {}
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader)):
            audio_path = data["audio_path"]
            hubert_feature = data["hubert_feature"].cuda()
            input_len = data["input_len"].cuda()
            # import pdb; pdb.set_trace()
            _, motion = model(hubert_feature, input_len)
            motion = motion.squeeze(0).cpu().numpy()
            motion = torch.FloatTensor(motion * std + mean)

            result= {
                "exp": motion[:, :50],
                "pose": motion[:, 50:],
                }

            itemname = os.path.splitext(os.path.basename(audio_path))[0]
            save_path = os.path.join(save_folder, itemname + "_predict.pt")
            torch.save(result, save_path)

            # save_huber_feature_path = os.path.join(save_folder, itemname + "_hubert.npy")
            # hubert_feature = hubert_feature.squeeze(0).cpu().numpy()
            # np.save(save_huber_feature_path, hubert_feature)
                


if __name__ == '__main__':
    main()

    # CUDA_VISIBLE_DEVICES=7 PYTHONPATH=./ python main/test.py --config=./config/train/s2e_base.yaml model_path /cfs2/user/chenliyang/Project/ArtFace/Speech2Landmark/RUN/speech2expression_obama_yuhangclip/model/model_10.pth.tar test_data_dir ./TEST/mobtts/prediction test_save_dir ./TEST/mobtts/debug