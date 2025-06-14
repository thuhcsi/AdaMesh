import os
import cv2
import numpy as np
import torch
from torch import nn
from moviepy.editor import AudioFileClip, ImageSequenceClip

from tools.DECA.decalib.utils.config import cfg
from tools.DECA.decalib.models.FLAME import FLAME
from tools.DECA.decalib.utils import util


class FLAMEWrapper(nn.Module):
    def __init__(self):
        super(FLAMEWrapper, self).__init__()
        self.flame = FLAME(cfg.model)

        # self.mouth_index = np.concatenate([np.arange(3, 14), np.arange(48, 68)])
        self.mouth_index = np.concatenate([np.arange(5, 12), np.arange(48, 68)])

        template_path = "/cfs2/user/chenliyang/Data/Obama/deca_feature/OB0000_0001/OB0000_0001_3dparams.pt"
        template = torch.load(template_path)

        self.shape = template["shape"][77].unsqueeze(0)
        self.pose = torch.zeros([1, 6], dtype=torch.float32, requires_grad=False)
        self.cam = template["cam"][77].unsqueeze(0)
    
    def forward(self, target_codedict, only_mouth=True):
        batch_size = target_codedict["exp"].shape[0]
        codedict = {}

        codedict["shape"] = self.shape.expand(batch_size, -1).clone()
        codedict["cam"] = self.cam.expand(batch_size, -1).clone()
        codedict["pose"] = self.pose.expand(batch_size, -1).clone()

        # import pdb; pdb.set_trace()
        codedict["pose"][:, 3:] = target_codedict["pose"][:, 3:]
        codedict["exp"] = target_codedict["exp"]

        _, landmarks2d, landmarks3d = self.flame(shape_params=codedict['shape'], expression_params=codedict['exp'], pose_params=codedict['pose'])
        
        ## projection
        landmarks2d = util.batch_orth_proj(landmarks2d, codedict['cam'])[:,:,:2]; landmarks2d[:,:,1:] = -landmarks2d[:,:,1:]#; landmarks2d = landmarks2d*self.image_size/2 + self.image_size/2
        # landmarks3d = util.batch_orth_proj(landmarks3d, codedict['cam']); landmarks3d[:,:,1:] = -landmarks3d[:,:,1:] #; landmarks3d = landmarks3d*self.image_size/2 + self.image_size/2

        # landmarks2d = transform_points(landmarks2d, tform, points_scale, [h, w])
        # landmarks3d = transform_points(landmarks3d, tform, points_scale, [h, w])
        landmarks2d = landmarks2d * 0.5 + 0.5  # normalize to [0, 1]

        if only_mouth:
            landmarks2d = landmarks2d[:, self.mouth_index, :]

        return landmarks2d
    

    def _save_ldmk(self, ldmk, audio_path, save_path):
        list_img = []
        for i in range(ldmk.shape[0]):
            bg = np.zeros([256,256, 3])
            for p in ldmk[i]:
                x, y = p
                cv2.circle(bg, (int(x * 256), int(y * 256)), 1, (0, 255, 0), -1)
            list_img.append(bg)
        
        audio_sample_frequency = 16000
        target_audio_clip = AudioFileClip(audio_path, fps=audio_sample_frequency)
        os.makedirs(os.path.split(save_path)[0], exist_ok=True)
        imgseqclip = ImageSequenceClip(list_img, fps=25)
        temp = imgseqclip.set_audio(target_audio_clip)
        temp.write_videofile(save_path, logger=None)
        
