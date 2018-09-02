from scipy.io import loadmat
import os

import skeleton_data as sk_data
import numpy as np

vnect_to_prior_perm = [14, 1, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 19, 11, 12, 13, 20]
vnect_to_prior_perm = np.array(vnect_to_prior_perm).astype(int)

limb_parents = [0, 0, 1, 2, 3, 1, 5, 6, 1, 0, 9, 10, 11, 0, 13, 14, 15]

# -1: Left, 0: Middle/Center 1: Right
lr_flags = [
    0, 0,
    1, 1, 1,
    -1, -1, -1,
    0,
    1, 1, 1, 1,
    -1, -1, -1, -1,
]

joint_names = [
    'pelvis', 'neck',  # 2
    'right_shoulder', 'right_elbow', 'right_wrist',  # 5
    'left_shoulder', 'left_elbow', 'left_wrist',  # 8
    'head_top',  # 9
    'right_hip', 'right_knee', 'right_ankle', 'right_foot',  # 13
    'left_hip', 'left_knee', 'left_ankle', 'left_foot'  # 17
]

joint_names = np.array(joint_names)

head_limb = 8

limb_root = 0

limb_ratios = sk_data.limb_ratios[vnect_to_prior_perm]
limb_ratios[1] = sk_data.limb_ratios[1] + sk_data.limb_ratios[15]
limb_ratios = np.array(limb_ratios)
limb_ratios /= limb_ratios[head_limb]

n_joints = len(limb_parents)
n_limbs = n_joints

limb_graph = sk_data.gen_limb_graph(limb_parents)
limb_order = sk_data.bfs_order(limb_graph, limb_root)

joint_map = {joint_names[i]: i for i in xrange(n_joints)}

torso_joints = [0, 1, 2, 5, 9, 13]

upper_limbs = {3, 6, 8, 10, 14}

neck_joints = {3, 6, 8}

child = [3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16]

static_pose_file = os.path.join(os.path.dirname(__file__), 'staticPose.mat')
D = loadmat(static_pose_file)['di']

arbitary_vec = loadmat(static_pose_file)['a'].squeeze()


def get_joint_index(joint_name):
    return joint_map.get(joint_name, -1)
