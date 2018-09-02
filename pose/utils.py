import os
import numpy as np
from matplotlib import pyplot as plt
from commons import transform_util as tr_utils, vis_util
from commons import skeleton_utils as sk_utils


def get_exp_name():
    return os.path.basename(os.path.abspath('.'))


def unit_norm(v, axis=0):
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(norm, 1e-9)


def copy_weights(iter_no, label='iter'):
    """
    Copied the Weights from weights/ to pretrained_weights/ given iteration number and label: 'iter' or 'best'
    """
    files = os.listdir('pose/weights/')
    match_substr = '%s-%d' % (label, iter_no)
    files = [f for f in files if match_substr in f]
    for f in files:
        cmd = 'cp pose/weights/%s pose/pretrained_weights/' % f
        print (cmd)
        os.system(cmd)


def get_most_recent_iteration(label='best', dir='weights'):
    """
    Gets the most recent iteration number from weights/ dir of given label: ('best' or 'iter')
    """
    files = os.listdir(dir)
    files = [f for f in files if label in f]
    numbers = {int(f[f.index('-') + 1:f.index('.')]) for f in files}
    return max(numbers)


def copy_latest(type='best'):
    latest_iter = get_most_recent_iteration(type)
    copy_weights(latest_iter, type)
    return latest_iter


def local_2_global_batch_skeleton(x_batch):
    x_batch = unit_norm(x_batch, axis=2)
    x_batch = sk_utils.scale_local_skeleton(x_batch)
    x_batch_global = np.array([tr_utils.prior_local2global(x_batch[i]) for i in range(x_batch.shape[0])])
    return sk_utils.fit_skeleton_frames(x_batch_global)


def get_global_skeleton(x):
    x = unit_norm(x, axis=1)
    x = sk_utils.scale_local_skeleton(x)
    x = tr_utils.prior_local2global(x)
    x = sk_utils.fit_skeleton_frame(x)
    return x


def plot_local_skeleton(x, title=""):
    x = get_global_skeleton(x)
    return vis_util.plot_skeleton(x, title=title)


def plot_global_skeleton(x, title=""):
    return vis_util.plot_skeleton(x, title=title)


def compare_global_skeletons(x_original, x_reconstructed):
    fig = plt.figure(frameon=False, figsize=(14, 7))
    titles = ['Original Skeleton', 'Reconstructed Skeleton']
    skeletons = [x_original, x_reconstructed]
    for i in range(2):
        ax = fig.add_subplot(1, 2, i + 1, projection='3d')
        ax.set_axis_off()
        ax.clear()
        ax.view_init(azim=-90, elev=10)
        ax.set_xlim(-800, 800)
        ax.set_ylim(-800, 800)
        ax.set_zlim(-800, 800)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(titles[i])
        vis_util.draw_limbs_3d_plt(skeletons[i] * 100, ax)
    return fig


def compare_multiple_global_skeletons(skeletons):
    num_skeletons = len(skeletons)
    height = 6
    width = num_skeletons * 6
    fig = plt.figure(frameon=False, figsize=(width, height))
    titles = ['Skeleton: %d' % i for i in range(num_skeletons)]
    for i in range(num_skeletons):
        ax = fig.add_subplot(1, num_skeletons, i + 1, projection='3d')
        ax.set_axis_off()
        ax.clear()
        ax.view_init(azim=-90, elev=10)
        ax.set_xlim(-800, 800)
        ax.set_ylim(-800, 800)
        ax.set_zlim(-800, 800)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(titles[i])
        vis_util.draw_limbs_3d_plt(skeletons[i] * 100, ax)
    return fig


def compare_local_skeletons(x_original, x_recon):
    x_original = get_global_skeleton(x_original)
    x_recon = get_global_skeleton(x_recon)
    return compare_global_skeletons(x_original, x_recon)


def compare_multiple_local_skeletons(skeletons):
    global_skeletons = [get_global_skeleton(sk) for sk in skeletons]
    return compare_multiple_global_skeletons(global_skeletons)


def plot_2_skeletons(x_original, x_reconstructed):
    plot_local_skeleton(x_original, 'Original skeleton')
    plot_local_skeleton(x_reconstructed, 'Reconstructed Skeleton')

# copy_weights(859501)
