from __future__ import absolute_import
import tensorflow as tf

from pose.hyperparams import Hyperparams as H
from . import prior_sk_data as sk_data

dtype = H.dtype

a = tf.constant(sk_data.arbitary_vec, dtype=dtype)
D = tf.constant(sk_data.D, dtype=dtype)
eps = tf.constant(1e-9, dtype=dtype)
parents = sk_data.limb_parents


def tf_unit_norm(vec, axis=-1):
    return vec / (tf.norm(vec, axis=axis, keep_dims=True) + 1e-15)


def tf_proj(u, v, axis=-1):
    return tf.reduce_sum(u * v, axis=axis, keep_dims=True) * v


def tf_gram_schmidt(V):
    rank = len(V.shape)
    if rank > 3 or rank < 2:
        raise Exception("gram_schmidt: invalid rank: %d" % rank)
    if rank == 2:
        V = V[None, :, :]
    # basis [B, 3, 3]
    u0 = tf_unit_norm(V[:, 0])
    u1 = tf_unit_norm(V[:, 1] - tf_proj(V[:, 1], u0))
    u2 = tf_unit_norm(V[:, 2] - tf_proj(V[:, 2], u0) - tf_proj(V[:, 2], u1))
    U = tf.concat([u0[:, None, :], u1[:, None, :], u2[:, None, :]], axis=1)
    if rank == 2:
        U = U[0]
    return U


def tf_prior_get_normal(u, v, w):
    get_wu = lambda: tf.cross(w, u)
    get_vw = lambda: tf.cross(v, w)
    cond = tf.logical_and(tf.norm(w - v) < eps, tf.norm(w + v) < eps)
    normal = tf.cond(cond, get_wu, get_vw)
    return tf_unit_norm(normal)


def tf_global2local(skeleton_batch):
    batch_size = tf.shape(skeleton_batch)[0]

    ap = a[None, :, None]
    ap = tf.tile(ap, multiples=[batch_size, 1, 1])

    dp = tf.gather(D, parents)[:, None, :]
    dp = tf.tile(dp, multiples=[1, batch_size, 1])

    # [17, B, 3]
    skeleton_batch = tf.transpose(skeleton_batch, [1, 0, 2])

    # [17, B, 3]
    dS = tf.gather(skeleton_batch, parents) - skeleton_batch

    # [B, 3]
    shldr = dS[sk_data.joint_map['left_shoulder']] - dS[sk_data.joint_map['right_shoulder']]
    hip = dS[sk_data.joint_map['left_hip']] - dS[sk_data.joint_map['right_hip']]

    dSl = [None] * 17

    for i in range(17):
        if i not in sk_data.child:
            dSl[i] = dS[i]
            continue

        if i in sk_data.upper_limbs:
            u = shldr if i in sk_data.neck_joints else hip
            u = tf_unit_norm(u, axis=-1)
            v = tf_unit_norm(dS[1], axis=-1)
        else:
            u = dS[parents[i]]
            u = tf_unit_norm(u, axis=-1)

            RT = tf.transpose(R, [0, 2, 1])
            x1 = tf.matmul(RT, dp[i][:, :, None])[:, :, 0]
            x2 = tf.matmul(RT, ap)[:, :, 0]
            x3 = u
            v = tf_prior_get_normal(x1, x2, x3)

        # u: [B, 3]
        # v: [B, 3]
        # w: [B, 3]
        w = tf.cross(u, v)
        w = tf_unit_norm(w, axis=-1)

        # [B, 3, 3]
        basis = tf.concat([u[:, None, :], v[:, None, :], w[:, None, :]], axis=1)

        # [B, 3, 3]
        R = tf_gram_schmidt(basis)
        dSl[i] = tf.matmul(R, dS[i][:, :, None])[:, :, 0]

    dSl = tf.concat([d[:, None, :] for d in dSl], axis=1)
    return dSl


def prior_estimate_absolute_positions(dS):
    # dS: [[B, 3], [B, 3], ..., [B, 3]] (17 times)
    batch_size = tf.shape(dS[0])[0]
    S = [None] * 17
    limb_order = sk_data.limb_order
    parent = sk_data.limb_parents
    S[limb_order[0]] = tf.zeros((batch_size, 3), dtype=dtype)

    for i in limb_order[1:]:
        S[i] = S[parent[i]] - dS[i]
    return tf.concat([s[None, :, :] for s in S], axis=0)


def tf_local2global(dS_local):
    # dS_local = [B, 17, 3]
    batch_size = tf.shape(dS_local)[0]

    ap = a[None, :, None]
    ap = tf.tile(ap, multiples=[batch_size, 1, 1])

    dp = tf.gather(D, parents)[:, None, :]
    dp = tf.tile(dp, multiples=[1, batch_size, 1])

    # [17, B, 3]
    dS_local = tf.transpose(dS_local, [1, 0, 2])

    shldr = dS_local[sk_data.joint_map['left_shoulder']] - dS_local[sk_data.joint_map['right_shoulder']]
    hip = dS_local[sk_data.joint_map['left_hip']] - dS_local[sk_data.joint_map['right_hip']]

    dS = [None] * 17

    for i in range(17):

        if i in sk_data.torso_joints:
            dS[i] = dS_local[i]
            continue

        if i in sk_data.upper_limbs:
            u = shldr if i in sk_data.neck_joints else hip
            u = tf_unit_norm(u, axis=-1)
            v = tf_unit_norm(dS[1], axis=-1)

        else:
            u = dS[parents[i]]
            u = tf_unit_norm(u, axis=-1)
            x1 = tf.matmul(RT, dp[i][:, :, None])[:, :, 0]
            x2 = tf.matmul(RT, ap)[:, :, 0]
            x3 = u
            v = tf_prior_get_normal(x1, x2, x3)

        w = tf.cross(u, v)
        w = tf_unit_norm(w)
        # [B, 3, 3]
        basis = tf.concat([u[:, None, :], v[:, None, :], w[:, None, :]], axis=1)
        # [B, 3, 3]
        R = tf_gram_schmidt(basis)
        RT = tf.transpose(R, [0, 2, 1])
        dS[i] = tf.matmul(RT, dS_local[i][:, :, None])[:, :, 0]
    S = prior_estimate_absolute_positions(dS)
    return tf.transpose(S, [1, 0, 2])
