import numpy as np
import tensorflow as tf
from hyperparams import Hyperparams as H
import model as M

import utils


def interpolate(v1, v2, n_points):
    l = np.linspace(0.0, 1.0, n_points + 2)[:, None]
    v_inter = v1 * (1 - l) + v2 * l
    return v_inter


class PoseModel(object):
    def __init__(self, iter_no=None):
        weight_joint, weight_joint_group, weight_full = 1. / (3 * 17), 1. / (3 * 5), 1. / (3 * 1)
        self.weight_vec = np.array([1. / 51] * 17 + [1. / 15] * 5 + [1. / 3])

        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.session = tf.InteractiveSession(config=self.config)
        self.session.run(tf.global_variables_initializer())

        if iter_no is not None:
            self.iter_no = iter_no
            print('Trying to load iter no:', self.iter_no)
            M.load_weights(self.iter_no, self.session, H.best_weights_path, H.best_weights_tag)
        else:
            self.iter_no = H.best_iter_no
            print('Trying to load iter no:', self.iter_no)
            M.load_weights(self.iter_no, self.session, H.best_weights_path, H.best_weights_tag)

    @property
    def iter_number(self):
        return self.iter_no

    def encode(self, x_batch, get_transform_mat=False):
        feed_dict = {M.input_x: x_batch}
        if get_transform_mat:
            return self.session.run([M.z_real, M.transform_mats], feed_dict)
        else:
            return self.session.run(M.z_real, feed_dict)

    def encode_view_norm(self, x_batch):
        return self.session.run(M.z_pred_view_norm, feed_dict={
            M.input_x_view_norm: x_batch
        })

    def decode(self, z_batch):
        feed_dict = {M.input_z: z_batch}
        x = self.session.run(M.x_view_norm_fake, feed_dict)
        return x

    def reconstruct_x(self, x_batch, get_loss=False):
        feed_dict = {M.input_x: x_batch}
        if get_loss:
            return self.session.run([M.x_recon, M.tensor_x_loss], feed_dict)
        else:
            return self.session.run(M.x_recon, feed_dict)

    def reconstruct_z(self, z_batch, get_loss=False):
        feed_dict = {M.input_z: z_batch}
        if get_loss:
            return self.session.run([M.z_recon, M.tensor_z_loss], feed_dict)
        else:
            return self.session.run(M.z_recon, feed_dict)

    def generate_sk_seq_from_z(self, z1, z2, num_steps):
        Z = interpolate(z1, z2, num_steps)
        X = self.decode(Z)
        imgs = [utils.plot_global_skeleton(x) for x in X]
        return imgs

    def generate_sk_seq_from_x(self, x1, x2, num_steps):
        z1, z2 = self.encode(np.array([x1, x2]))
        return self.generate_sk_seq_from_z(z1, z2, num_steps)

    def get_losses_batch(self, x_batch, z_batch=None, losses=(M.loss_encoder, M.x_recon_loss_l1, M.z_recon_loss_l1)):
        if z_batch is None:
            z_batch = np.random.uniform(-1, 1, size=(x_batch.shape[0], M.num_zdim))

        op_vals = self.session.run(losses, feed_dict={
            M.input_x: x_batch,
            M.input_z: z_batch,
            M.weight_vec_ph: self.weight_vec,

        })
        return np.array(op_vals)
