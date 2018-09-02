# import os, sys
from cv2 import cv2

import os
from os import path as osp
from scipy.io import loadmat
import numpy as np

video_base_path = '../data/videos'
frames_base_path = '../data/frames'
hmaps_base_path = '../data/hmaps'
preds_base_path = '../data/preds'
sk_frames_base_path = '../data/sk_frames'
sk_videos_base_path = '../data/sk_videos'
cropped_frames_base_path = '../data/cropped_frames'


class VideoProject(object):
    def __init__(self, video_name):
        self.video_name = video_name
        self.project_name = osp.basename(video_name.split('.')[0])
        self.video_path = osp.join(video_base_path, video_name)
        self.frames_dir_path = osp.join(frames_base_path, self.project_name)
        self.preds_dir_path = osp.join(preds_base_path, self.project_name)
        self.hmaps_dir_path = osp.join(hmaps_base_path, self.project_name)
        self.cropped_frames_dir_path = osp.join(cropped_frames_base_path, self.project_name)
        self.sk_frames_dir_path = osp.join(sk_frames_base_path, self.project_name)
        self.sk_video_path = osp.join(sk_videos_base_path, 'sk_%s.avi' % self.project_name)

    def __str__(self):
        return self.project_name

    def get_frame_path(self, img_id):
        return osp.join(self.frames_dir_path, 'frame%04d.png' % img_id)

    def get_cropped_frame_path(self, img_id):
        return osp.join(self.cropped_frames_dir_path, 'cropped_frame%04d.png' % img_id)

    def get_hmap_path(self, img_id):
        return osp.join(self.hmaps_dir_path, 'hmap_frame%04d.png' % img_id)

    def get_pred_path(self, img_id=None):
        if img_id is None:
            return osp.join(preds_base_path, self.project_name + '_output.mat')
        else:
            return osp.join(self.preds_dir_path, 'pred_frame%04d.mat' % img_id)

    def get_sk_frame_path(self, img_id):
        return osp.join(self.sk_frames_dir_path, 'sk_frame%04d.png' % img_id)

    def get_hmap_img(self, img_id):
        return cv2.imread(self.get_hmap_path(img_id))[:, :, [2, 1, 0]]

    def get_preds(self, img_id=None):
        return loadmat(self.get_pred_path(img_id))

    def get_hmap_imgs(self):
        return np.array([self.get_hmap_img(img_id) for img_id in xrange(self.num_frames)])

    @property
    def num_frames(self):
        num_frames = 0
        try:
            num_frames = len(os.listdir(self.frames_dir_path))
        except:
            pass
        return num_frames