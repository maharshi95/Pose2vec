from scipy.signal import savgol_filter
import numpy as np

# from . import skeleton_data as sk_data
from . import prior_sk_data as psk_data

sk_data = psk_data


def smoothen(preds, window=11, order=3, axis=0):
    return savgol_filter(preds, window, order, axis=axis)


def get_parent_relative_joint_locations(joints_xyz, axis=0, sk_data=sk_data):
    """
    Assumed shape: [#limbs, X, ...]
    """
    limb_parents = sk_data.limb_parents
    joints_xyz = joints_xyz.swapaxes(0, axis)
    limb_rel_preds = joints_xyz - joints_xyz[limb_parents]
    return limb_rel_preds.swapaxes(0, axis)


def get_abs_joint_locations(rel_joints_xyz, axis=0, sk_data=sk_data):
    """
    Assumed shape: [#limbs, X, ...]
    """
    limb_parents, limb_order = sk_data.limb_parents, sk_data.limb_order

    rel_joints_xyz = rel_joints_xyz.swapaxes(0, axis)
    abs_preds = np.zeros_like(rel_joints_xyz, dtype=np.float64)
    abs_preds[limb_order[0]] = rel_joints_xyz[limb_order[0]]
    for l in limb_order[1:]:
        p = limb_parents[l]
        abs_preds[l] = abs_preds[p] + rel_joints_xyz[l]
    return abs_preds.swapaxes(0, axis)


def cart2pol(xyz, axis=0):
    """
    Assumed shape: [3, X, ...]
    """
    xyz = xyz.swapaxes(0, axis)
    rtp = np.zeros_like(xyz, dtype=np.float64)
    rtp[0] = np.linalg.norm(xyz, axis=0)
    rtp[1] = np.arctan2(xyz[2], np.sqrt(xyz[0] ** 2 + xyz[1] ** 2))
    rtp[2] = np.arctan2(xyz[1], xyz[0])
    return rtp.swapaxes(0, axis)


def pol2cart(rtp, axis=0):
    """
    Assumed shape: [3, X, ...]
    """
    rtp = rtp.swapaxes(0, axis)
    xyz = np.zeros_like(rtp, dtype=np.float64)
    xyz[0] = rtp[0] * np.cos(rtp[1]) * np.cos(rtp[2])
    xyz[1] = rtp[0] * np.cos(rtp[1]) * np.sin(rtp[2])
    xyz[2] = rtp[0] * np.sin(rtp[1])
    return xyz.swapaxes(0, axis)


def fit_skeleton_frame(preds, head_length=2.0, sk_data=sk_data):
    """
    :param preds: joint values of a skeleton or skeleton vector
    :shape: [#limbs, 3, X, X, ...]
    """

    rel_preds = get_parent_relative_joint_locations(preds, sk_data=sk_data)

    pol_rel_preds = cart2pol(rel_preds, axis=1)

    fixed_limb_lengths = head_length * sk_data.limb_ratios

    pol_rel_preds[:, 0].T[:] = fixed_limb_lengths * (pol_rel_preds[:, 0].T[:] != 0.)

    cart_rel_preds = pol2cart(pol_rel_preds, axis=1)

    cart_abs_preds = get_abs_joint_locations(cart_rel_preds, sk_data=sk_data)

    return cart_abs_preds


def fit_skeleton_frames(frames, head_length=2.0, sk_data=sk_data):
    orig_shape = frames.shape
    frames = frames.reshape(-1, *orig_shape[-2:])
    fit_frames = fit_skeleton_frame(frames.transpose([1, 2, 0]), head_length, sk_data).transpose([2, 0, 1])
    return fit_frames.reshape(orig_shape)


def scale_local_skeleton(preds, head_length=2.0):
    fixed_limb_lengths = np.expand_dims(head_length * sk_data.limb_ratios, axis=1)
    return preds * fixed_limb_lengths
