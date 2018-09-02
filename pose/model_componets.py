import json
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layer

from collections import OrderedDict
from src.hyperparams import Hyperparams as H
from .commons import tf_transform as tr
from .commons import prior_sk_data as sk_data

dtype = H.dtype


def debug_dict(d):
    d = {k: str(d[k]) for k in d}
    s = json.dumps(d, indent=4, sort_keys=True)
    print s


########################### Encoder Decoder Discriminator Architecture ##########################
#################################################################################################

num_joints = 17
num_params_per_joint = 3
num_zdim = 32
num_z_angles = 8
num_params_total = num_joints * num_params_per_joint

limb_parents = np.array(sk_data.limb_parents, dtype=np.int32)

limb_lengths = tf.constant(sk_data.limb_ratios * 2, dtype=dtype)


def unit_norm(tensor, axis=-1):
    norm = tf.norm(tensor, axis=axis, keep_dims=True)
    return tensor / (norm + 1e-9)


def path_from_pelvis(joint_index):
    if limb_parents[joint_index] == joint_index:
        return [joint_index]
    return path_from_pelvis(limb_parents[joint_index]) + [joint_index]


def get_parent_relative_joints(root_relative_joints):
    return unit_norm(root_relative_joints - tf.gather(root_relative_joints, limb_parents, axis=1))


def get_root_relative_joints(parent_relative_joints):
    limb_lengths = tf.constant(sk_data.limb_ratios * 2, dtype=dtype)
    parent_relative_joints = parent_relative_joints * limb_lengths[:, tf.newaxis]
    root_relative_joints = tf.concat(
        [
            # [B, 1, 3]
            tf.add_n([
                parent_relative_joints[:, path_joint][:, tf.newaxis, :]
                for path_joint in path_from_pelvis(joint_id)
            ])
            for joint_id in range(num_joints)
        ], axis=1)
    return root_relative_joints


def get_skeleton_transform_matrix(skeleton_batch):
    # skeleton_batch: [B, 17, 3]
    right_hip = sk_data.get_joint_index('right_hip')
    left_hip = sk_data.get_joint_index('left_hip')
    neck = sk_data.get_joint_index('neck')

    r = skeleton_batch[:, right_hip:right_hip + 1]
    l = skeleton_batch[:, left_hip:left_hip + 1]
    n = skeleton_batch[:, neck:neck + 1]

    m = 0.5 * (r + l)
    z_ = unit_norm(n - m, axis=-1)
    y_ = unit_norm(tf.cross(l - r, n - r), axis=-1)
    x_ = tf.cross(y_, z_)

    transform_mats = tf.concat([x_, y_, z_], axis=1)
    print 'dets', tf.linalg.det(transform_mats)
    print 'trasnforms_mats', transform_mats.shape

    return transform_mats


def get_skeleton_z_transform_matrix(skeleton_batch):
    # skeleton_batch: [B, 17, 3]
    batch_size = tf.shape(skeleton_batch)[0]
    right_hip = sk_data.get_joint_index('right_hip')
    left_hip = sk_data.get_joint_index('left_hip')

    r = skeleton_batch[:, right_hip:right_hip + 1]
    l = skeleton_batch[:, left_hip:left_hip + 1]

    z_ = tf.tile(tf.convert_to_tensor([[[0, 0, 1]]], dtype=dtype), [batch_size, 1, 1])
    x_ = unit_norm((r - l) * tf.convert_to_tensor([[[1., 1., 0.]]], dtype=dtype))
    y_ = tf.cross(z_, x_)
    transform_mats = tf.concat([x_, y_, z_], axis=1)
    return transform_mats


def root_relative_to_view_norm(skeleton_batch):
    transform_mats = get_skeleton_z_transform_matrix(skeleton_batch)
    sk_batch_view_norm = tf.matmul(skeleton_batch, transform_mats, transpose_b=True)
    return sk_batch_view_norm, transform_mats


def view_norm_to_root_relative(skeleton_batch, transform_mats):
    return tf.matmul(skeleton_batch, transform_mats)


def root_relative_to_local(skeleton_batch):
    sk_batch_view_norm, transform_mats = root_relative_to_view_norm(skeleton_batch)
    sk_batch_local = tr.tf_global2local(sk_batch_view_norm)
    sk_batch_local = unit_norm(sk_batch_local, axis=-1)
    return sk_batch_view_norm, sk_batch_local, transform_mats


def local_to_root_relative(skeleton_batch, transform_mats):
    skeleton_batch = skeleton_batch * limb_lengths[:, tf.newaxis]
    sk_batch_view_norm = tr.tf_local2global(skeleton_batch)
    sk_batch_root_relative = view_norm_to_root_relative(sk_batch_view_norm, transform_mats)
    return sk_batch_root_relative


def view_norm_to_local(skeleton_batch):
    sk_batch_local = tr.tf_global2local(skeleton_batch)
    sk_batch_local = unit_norm(sk_batch_local, axis=-1)
    return sk_batch_local


def local_to_view_norm(skeleton_batch):
    skeleton_batch = skeleton_batch * limb_lengths[:, tf.newaxis]
    sk_batch_view_norm = tr.tf_local2global(skeleton_batch)
    return sk_batch_view_norm


def get_relu_fn(use_relu=True, leak=0.2):
    def lrelu(x, leak=leak, name="lrelu"):
        return tf.maximum(x, leak * x)

    return tf.nn.relu if use_relu else lrelu


def inverse_graph(graph_dict):
    inverse_graph_dict = {joint_name: [] for joint_name in joint_names}
    for u_node, edges in graph_dict.items():
        for i, v_node in enumerate(edges):
            inverse_graph_dict.setdefault(v_node, [])
            inverse_graph_dict[v_node].append((i, u_node))
    return inverse_graph_dict


def ft_name(name, ft_id=None):
    return name + '_ft' if ft_id is None else "{}_ft_{}".format(name, ft_id)


def ft_less_name(name_ft):
    return name_ft[:-2]


joint_names = [
    'pelvis_0', 'neck_1',
    'r_shldr_2', 'r_elbow_3', 'r_wrist_4',
    'l_shldr_5', 'l_elbow_6', 'l_wrist_7',
    'head_8',
    'r_hip_9', 'r_knee_10', 'r_ankle_11', 'r_foot_12',
    'l_hip_13', 'l_knee_14', 'l_ankle_15', 'l_foot_16',
]

encoder_joints_dict = {
    'r_arm': ['neck_1', 'r_shldr_2', 'r_elbow_3', 'r_wrist_4'],
    'l_arm': ['neck_1', 'l_shldr_5', 'l_elbow_6', 'l_wrist_7'],
    ##
    'r_leg': ['r_hip_9', 'r_knee_10', 'r_ankle_11', 'r_foot_12'],
    'l_leg': ['l_hip_13', 'l_knee_14', 'l_ankle_15', 'l_foot_16'],
    ##
    'trunk': ['pelvis_0', 'neck_1', 'head_8'],
    ##
    'trunk_r_arm': ['trunk_ft', 'r_arm_ft'],
    'trunk_l_arm': ['trunk_ft', 'l_arm_ft'],
    'trunk_r_leg': ['trunk_ft', 'r_leg_ft'],
    'trunk_l_leg': ['trunk_ft', 'l_leg_ft'],
    ##
    'upper_body': ['trunk_r_arm_ft', 'trunk_l_arm_ft'],
    'lower_body': ['trunk_r_leg_ft', 'trunk_l_leg_ft'],
    ##
    'full_body': ['upper_body_ft', 'lower_body_ft'],
}

# This dictionary preserves the index the child of a node in the list of children in the original graph
decoder_joints_dict = inverse_graph(encoder_joints_dict)


def EncoderNet(input_encoder_x, use_relu=False, name="Encoder_net"):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        encoder_net = OrderedDict()

        ############### Indiviual Joints ###############
        for i, joint_name in enumerate(joint_names):
            encoder_net[joint_name] = input_encoder_x[:, i, :]

        ############### Level 1 Joint Group ###############
        ###  joint -> joint_group_1 -> joint_group_1_ft ###
        for joint_group in ['l_arm', 'r_arm', 'r_leg', 'l_leg', 'trunk']:
            joint_group_ft = joint_group + '_ft'
            encoder_net[joint_group] = tf.concat([encoder_net[sub_part] for sub_part in encoder_joints_dict[joint_group]], axis=1)
            encoder_net[joint_group_ft] = tf_layer.fully_connected(encoder_net[joint_group],
                                                                   num_outputs=32,
                                                                   activation_fn=get_relu_fn(use_relu),
                                                                   scope=joint_group_ft)

        ##################### Level 2 Joint Group ####################
        ###  joint_group_1_ft -> joint_group_2 -> joint_group_2_ft ###
        for joint_group in ['trunk_l_arm', 'trunk_r_arm', 'trunk_r_leg', 'trunk_l_leg']:
            joint_group_ft = joint_group + '_ft'
            encoder_net[joint_group] = tf.concat([encoder_net[sub_part] for sub_part in encoder_joints_dict[joint_group]], axis=1)
            encoder_net[joint_group_ft] = tf_layer.fully_connected(encoder_net[joint_group],
                                                                   num_outputs=64,
                                                                   activation_fn=get_relu_fn(use_relu),
                                                                   scope=joint_group_ft)

        ##################### Level 3 Joint Group ####################
        ###  joint_group_2_ft -> joint_group_3 -> joint_group_3_ft ###
        for joint_group in ['upper_body', 'lower_body']:
            joint_group_ft = joint_group + '_ft'
            encoder_net[joint_group] = tf.concat([encoder_net[sub_part] for sub_part in encoder_joints_dict[joint_group]], axis=1)
            encoder_net[joint_group_ft] = tf_layer.fully_connected(encoder_net[joint_group],
                                                                   num_outputs=128,
                                                                   activation_fn=get_relu_fn(use_relu),
                                                                   scope=joint_group_ft)

        ##################### Level 4 Joint Group ####################
        ###  joint_group_3_ft -> joint_group_4 -> joint_group_4_ft ###
        for joint_group in ['full_body']:
            joint_group_ft = joint_group + '_ft'
            encoder_net[joint_group] = tf.concat([encoder_net[sub_part] for sub_part in encoder_joints_dict[joint_group]], axis=1)
            encoder_net[joint_group_ft] = tf_layer.fully_connected(encoder_net[joint_group],
                                                                   num_outputs=512,
                                                                   activation_fn=get_relu_fn(use_relu),
                                                                   scope=joint_group_ft)

        ##################### Final Layer of FCC ####################
        encoder_net['full_body_ft2'] = tf_layer.fully_connected(encoder_net['full_body_ft'],
                                                                num_outputs=512,
                                                                activation_fn=get_relu_fn(use_relu),
                                                                scope='full_body_ft2')

        encoder_net['z_joints'] = tf_layer.fully_connected(encoder_net['full_body_ft'],
                                                           num_outputs=num_zdim,
                                                           activation_fn=tf.tanh,
                                                           scope='z_joints')
    return encoder_net


def DecoderNet(input_decoder_z_joints, reuse=False, use_relu=False, name="Decoder_net"):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        decoder_net = OrderedDict()

        decoder_net['z_joints'] = input_decoder_z_joints

        decoder_net['full_body_ft2'] = tf_layer.fully_connected(decoder_net['z_joints'],
                                                                num_outputs=512,
                                                                activation_fn=get_relu_fn(use_relu),
                                                                scope='full_body_ft2')

        decoder_net['full_body_ft'] = tf_layer.fully_connected(decoder_net['full_body_ft2'],
                                                               num_outputs=512,
                                                               activation_fn=get_relu_fn(use_relu),
                                                               scope='full_body_ft')

        ###
        decoder_net['full_body'] = (
            tf_layer.fully_connected(decoder_net['full_body_ft'],
                                     num_outputs=512,
                                     activation_fn=get_relu_fn(use_relu),
                                     scope='full_body')

        )

        for joint_group in ['upper_body', 'lower_body']:
            n_units = 128
            relu_fn = get_relu_fn(use_relu)
            joint_group_ft = ft_name(joint_group)
            super_group_layers = [decoder_net[super_part][:, i * n_units: (i + 1) * n_units] for i, super_part in
                                  decoder_joints_dict[joint_group_ft]]
            decoder_net[joint_group_ft] = relu_fn(tf.add_n(super_group_layers))

            decoder_net[joint_group] = tf_layer.fully_connected(decoder_net[joint_group_ft],
                                                                num_outputs=n_units,
                                                                activation_fn=get_relu_fn(use_relu),
                                                                scope=joint_group)

        for joint_group in ['trunk_l_arm', 'trunk_r_arm', 'trunk_r_leg', 'trunk_l_leg']:
            n_units = 64
            relu_fn = get_relu_fn(use_relu)
            joint_group_ft = ft_name(joint_group)
            super_group_layers = [decoder_net[super_part][:, i * n_units: (i + 1) * n_units] for i, super_part in
                                  decoder_joints_dict[joint_group_ft]]
            decoder_net[joint_group_ft] = relu_fn(tf.add_n(super_group_layers))

            decoder_net[joint_group] = tf_layer.fully_connected(decoder_net[joint_group_ft],
                                                                num_outputs=n_units,
                                                                activation_fn=get_relu_fn(use_relu),
                                                                scope=joint_group)

        for joint_group in ['l_arm', 'r_arm', 'r_leg', 'l_leg', 'trunk']:
            n_units = 32
            relu_fn = get_relu_fn(use_relu)
            joint_group_ft = ft_name(joint_group)
            super_group_layers = [decoder_net[super_part][:, i * n_units: (i + 1) * n_units] for i, super_part in
                                  decoder_joints_dict[joint_group_ft]]
            decoder_net[joint_group_ft] = relu_fn(tf.add_n(super_group_layers))

            joint_group_ft_units = num_params_per_joint * len(encoder_joints_dict[joint_group])
            decoder_net[joint_group] = tf_layer.fully_connected(decoder_net[joint_group_ft],
                                                                num_outputs=joint_group_ft_units,
                                                                activation_fn=None,
                                                                scope=joint_group)
        for joint in joint_names[:]:
            n_units = num_params_per_joint
            super_group_layers = [decoder_net[super_part][:, i * n_units: (i + 1) * n_units] for i, super_part in
                                  decoder_joints_dict[joint]]
            decoder_net[joint] = tf.add_n(super_group_layers)

        ############### concat Indiviual Joints ###############
        full_body_x = tf.concat([tf.expand_dims(decoder_net[joint_name], axis=1) for joint_name in joint_names], axis=1)
        norms = tf.norm(full_body_x, axis=2, keep_dims=True)
        decoder_net['full_body_x'] = full_body_x / (norms + tr.eps)  # For Handing 0 norm cases
        return decoder_net


def DiscriminatorNet(input_disc_x, reuse=False, use_relu=True, name="DiscNet"):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        disc_net = {'final_fc_names': []}

        ############### Indiviual Joints ###############
        for i, joint_name in enumerate(joint_names):
            disc_net[joint_name] = input_disc_x[:, i, :]

            joint_ft_1 = ft_name(joint_name, 1)
            joint_ft_2 = ft_name(joint_name, 2)
            joint_fc = joint_name + '_fc'
            reuse = False if i == 0 else True
            disc_net[joint_ft_1] = tf_layer.fully_connected(disc_net[joint_name],
                                                            num_outputs=32,
                                                            activation_fn=get_relu_fn(use_relu),
                                                            scope='layer_1_shared',
                                                            reuse=reuse)

            disc_net[joint_ft_2] = tf_layer.fully_connected(disc_net[joint_ft_1],
                                                            num_outputs=32,
                                                            activation_fn=get_relu_fn(use_relu),
                                                            scope='layer_2_shared',
                                                            reuse=reuse)

            disc_net[joint_fc] = tf_layer.fully_connected(disc_net[joint_ft_2],
                                                          num_outputs=1,
                                                          activation_fn=None,
                                                          scope=joint_fc)
            disc_net['final_fc_names'].append(joint_fc)

        ############### Level 1 Joint Group ###############
        ###  joint -> joint_group_1 -> joint_group_1_ft ###
        joint_groups = ['l_arm', 'r_arm', 'r_leg', 'l_leg', 'trunk']

        for joint_group in joint_groups:
            joint_group_ft = joint_group + '_ft'
            joint_group_fc = joint_group + '_fc'

            disc_net[joint_group] = tf.concat([disc_net[ft_name(sub_part, 2)] for sub_part in encoder_joints_dict[joint_group]],
                                              axis=1)

            disc_net[joint_group_ft] = tf_layer.fully_connected(disc_net[joint_group],
                                                                num_outputs=200,
                                                                activation_fn=get_relu_fn(use_relu),
                                                                scope=joint_group_ft)

            disc_net[joint_group_fc] = tf_layer.fully_connected(disc_net[joint_group_ft],
                                                                num_outputs=1,
                                                                activation_fn=None,
                                                                scope=joint_group_fc)
            disc_net['final_fc_names'].append(joint_group_fc)

        disc_net['joint_groups_concat'] = tf.concat([disc_net[ft_name(joint_group)] for joint_group in joint_groups], axis=1)

        disc_net['joint_groups_fcc_1'] = tf_layer.fully_connected(disc_net['joint_groups_concat'],
                                                                  num_outputs=1024,
                                                                  activation_fn=get_relu_fn(use_relu),
                                                                  scope='joint_groups_fcc_1')

        disc_net['joint_groups_fcc_2'] = tf_layer.fully_connected(disc_net['joint_groups_fcc_1'],
                                                                  num_outputs=1024,
                                                                  activation_fn=get_relu_fn(use_relu),
                                                                  scope='joint_groups_fcc_2')

        disc_net['joint_groups_final_fc'] = tf_layer.fully_connected(disc_net['joint_groups_fcc_2'],
                                                                     num_outputs=1,
                                                                     activation_fn=get_relu_fn(use_relu),
                                                                     scope='joint_groups_final_fc')
        disc_net['final_fc_names'].append('joint_groups_final_fc')
        disc_net['fcc_logits'] = tf.concat([disc_net[fc_name] for fc_name in disc_net['final_fc_names']], axis=1)

    return disc_net


'''
encoder_net = Encoder_net(input_encoder_x)
decoder_net = Decoder_net(input_decoder_z)
disc_net = DiscriminatorNet(input_disc_x)

debug_dict(disc_net)
debug_dict(encoder_net)
debug_dict(decoder_net)

print disc_net[disc_net['final_fc_names']]
print len(encoder_net.keys()), len(decoder_net.keys()), len(disc_net.keys())
'''
