import traceback
from .hyperparams import Hyperparams as H
import os
import numpy as np
import tensorflow as tf
import utils
from model_componets import EncoderNet, DecoderNet, DiscriminatorNet, get_parent_relative_joints, get_root_relative_joints
from model_componets import num_joints, num_zdim, num_params_per_joint, num_params_total, num_z_angles
import model_componets as comps

# batch_size_disc, batch_size_gen, learning_rate = 10, 10, 0.01

###################################################################################
############################### Define Placeholders ###############################

# Network Inputs
input_x = tf.placeholder(H.dtype, shape=[None, num_joints, num_params_per_joint], name='input_x')
input_x_view_norm = tf.placeholder(H.dtype, shape=[None, num_joints, num_params_per_joint], name='input_x_view_norm')
input_z = tf.placeholder(H.dtype, shape=[None, num_zdim], name='input_decoder_z_joints')
# input_angles_z = tf.placeholder(H.dtype, shape=[None, 3], name='input_decoder_z_angles')

# Learning Rates
lr_disc_ph = tf.placeholder(H.dtype, shape=[], name='lr_disc_ph')
lr_encoder_ph = tf.placeholder(H.dtype, shape=[], name='lr_encoder_ph')
lr_decoder_ph = tf.placeholder(H.dtype, shape=[], name='lr_decoder_ph')

# Weight Vector for Discriminator
weight_vec_ph = tf.placeholder(H.dtype, shape=[23, ], name='weight_vec_ph')

###################################################################################
########################### Define Model Architechture ############################

### First Cycle ####

x_real = input_x
x_view_norm_real, x_local_real, transform_mats = comps.root_relative_to_local(x_real)
encoder_real = EncoderNet(x_local_real)
z_real = encoder_real['z_joints']

decoder_real = DecoderNet(z_real)

x_local_recon = decoder_real['full_body_x']
x_recon = comps.local_to_root_relative(x_local_recon, transform_mats)
# Check transform mats

determinants = tf.linalg.det(transform_mats)
x_real_dummy = comps.view_norm_to_root_relative(x_view_norm_real, transform_mats)
x_real_vs_dummy = tf.reduce_mean(tf.abs(x_real - x_real_dummy))

### Second Cycle ####
z_rand = input_z

decoder_fake = DecoderNet(z_rand)
x_local_fake = decoder_fake['full_body_x']

encoder_fake = EncoderNet(x_local_fake)
z_recon = encoder_fake['z_joints']

x_view_norm_fake = comps.local_to_view_norm(x_local_fake)

### Disc for x ###
disc_real = DiscriminatorNet(x_view_norm_real)
disc_real_x_logits = disc_real['fcc_logits']

### Disc for x_hat ###
disc_fake = DiscriminatorNet(x_view_norm_fake)
disc_fake_x_logits = disc_fake['fcc_logits']

### Prediction for x_view_norm
z_pred_view_norm = comps.EncoderNet(comps.view_norm_to_local(input_x_view_norm))['z_joints']

###################################################################################
################################ Define losses ####################################

### Adversarial loss ###
tensor_loss_disc_real = tf.reduce_mean(tf.abs(disc_real_x_logits - tf.ones_like(disc_real_x_logits)), axis=0)
tensor_loss_disc_fake = tf.reduce_mean(tf.abs(disc_fake_x_logits + tf.ones_like(disc_fake_x_logits)), axis=0)
tensor_loss_gen_adv = tf.reduce_mean(tf.abs(disc_fake_x_logits - tf.ones_like(disc_fake_x_logits)), axis=0)

tensor_loss_disc_adv = (tensor_loss_disc_real + tensor_loss_disc_fake) / 2.

loss_disc_adv = tf.reduce_sum(weight_vec_ph * tensor_loss_disc_adv)

loss_gen_adv = tf.reduce_sum(weight_vec_ph * tensor_loss_gen_adv)

## Define accuracy ##
disc_acc_real = tf.reduce_mean(tf.cast(disc_real_x_logits >= 0, H.dtype)) * 100.0
disc_acc_fake = tf.reduce_mean(tf.cast(disc_fake_x_logits < 0, H.dtype)) * 100.0

disc_acc = (disc_acc_real + disc_acc_fake) / 2.
gen_acc = 100 - disc_acc_fake

### Cyclic loss ###
# [B, ]
tensor_x_loss = tf.reduce_mean((x_recon - x_real) ** 2, axis=[1, 2])
tensor_z_loss = tf.reduce_mean((z_rand - z_recon) ** 2, axis=1)

x_recon_loss_l1 = tf.reduce_mean(tf.abs(x_recon - x_real))
z_recon_loss_l1 = tf.reduce_mean(tf.abs(z_rand - z_recon))

tensor_c_loss = tensor_x_loss + tensor_z_loss

loss_x_recon = tf.reduce_mean(tensor_x_loss)
loss_z_recon = tf.reduce_mean(tensor_z_loss)

loss_cyclic = loss_x_recon + loss_z_recon

### Total loss ###
loss_disc = loss_disc_adv
loss_encoder = 100 * loss_cyclic
loss_decoder = 100 * loss_cyclic + 5 * loss_gen_adv


# loss_decoder = 0.002 * loss_gen_adv + (1 - 0.002) * loss_cyclic


##############################bat#####################################################
########################## Define operations ######################################

def get_network_params(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)


param_disc = get_network_params(scope='DiscNet')
param_encoder = get_network_params(scope='Encoder_net')
param_decoder = get_network_params(scope='Decoder_net')

disc_train_op = tf.train.AdamOptimizer(learning_rate=lr_disc_ph).minimize(loss_disc, var_list=param_disc)
encoder_train_op = tf.train.AdamOptimizer(learning_rate=lr_encoder_ph).minimize(loss_encoder, var_list=param_encoder)
decoder_train_op = tf.train.AdamOptimizer(learning_rate=lr_decoder_ph).minimize(loss_decoder, var_list=param_decoder)

###################################################################################
############################# Get Parameters #######################################
param_disc = get_network_params(scope='DiscNet')
param_encoder = get_network_params(scope='Encoder_net')
param_decoder = get_network_params(scope='Decoder_net')

params = {
    'encoder': param_encoder,
    'decoder': param_decoder,
    'disc': param_disc
}

#################################################################################################
############################ Summary for the Generator images ###################################

scalars = [
    tf.summary.scalar('loss_disc', loss_disc),
    tf.summary.scalar('disc_acc', disc_acc),
    tf.summary.scalar('gen_acc', gen_acc),
    tf.summary.scalar('loss_encoder', loss_encoder),
    tf.summary.scalar('loss_decoder', loss_decoder),
    tf.summary.scalar('loss_cyclic', loss_cyclic),
    tf.summary.scalar('loss_gen_adv', loss_gen_adv),
    tf.summary.scalar('loss_x_recon', loss_x_recon),
    tf.summary.scalar('loss_z_recon', loss_z_recon),
    tf.summary.scalar('loss_x_recon_l1', x_recon_loss_l1),
    tf.summary.scalar('loss_z_recon_l1', x_recon_loss_l1),
    tf.summary.scalar('loss_real_dummy', x_real_vs_dummy),
    tf.summary.scalar('deter', tf.reduce_mean(determinants)),
]

# z_a_histograms = [
#     tf.summary.histogram('z_a_%d' % i, z_real_angles_a[:, i]) for i in range(num_z_angles)
# ]

z_j_histograms = [
    tf.summary.histogram('z_j_%02d' % i, z_real[:, i]) for i in range(num_zdim)
]

###################   content loss summaries   ####################################################
summary_merge_all = tf.summary.merge(scalars + [tf.summary.histogram('dets', determinants)])
summary_merge_valid = tf.summary.merge(scalars + z_j_histograms)


# summary_merge_encoder = tf.summary.merge([sl3])
# summary_merge_decoder = tf.summary.merge([sl4, sl5, sl6, sl2])
# summary_merge_disc = tf.summary.merge([sl1, sl2])


################################################################################################
############################# Start Session ####################################################

############################# Run the model #####################################
# print('Starting training...')
# x_batch, z_batch = np.ones((10, 17, 3)), np.zeros((10, 64))
# weight_joint, weight_joint_group, weight_full = 1. / (3 * 17), 1. / (3 * 5), 1. / (3 * 1)
# weight_vec = np.concatenate((weight_joint * np.ones((17,)), weight_joint_group * np.ones((5,)), weight_full * np.ones((1,))))
#
# ### Single discriminator training iteration ###
# feed_dict = {input_decoder_z: z_batch, input_encoder_x: x_batch, weight_vec_tf: weight_vec, lr_disc_ph: 0.001}
# outputs_disc_train = [disc_train_op, loss_disc_adv]
# op_network_disc = session_1.run(outputs_disc_train, feed_dict)
#
# ### Single Encoder training iteration ###
# feed_dict = {input_decoder_z: z_batch, input_encoder_x: x_batch, weight_vec_tf: weight_vec, lr_encoder_ph: 0.001}
# outputs_encoder_train = [encoder_train_op, loss_encoder, loss_cyclic]
# op_network_encoder = session_1.run(outputs_encoder_train, feed_dict)
#
# ### Single Decoder training iteration ###
# feed_dict = {input_decoder_z: z_batch, input_encoder_x: x_batch, weight_vec_tf: weight_vec, lr_decoder_ph: 0.001}
# outputs_decoder_train = [decoder_train_op, loss_decoder, loss_cyclic, loss_gen_adv]
# op_network_decoder = session_1.run(outputs_decoder_train, feed_dict)
#
# ### Single Encoder Validation iteration ###
# feed_dict = {input_decoder_z: z_batch, input_encoder_x: x_batch, weight_vec_tf: weight_vec, lr_encoder_ph: 0.001}
# outputs_encoder_val_train = [loss_encoder, loss_cyclic]
# op_network_encoder_val = session_1.run(outputs_encoder_train, feed_dict)

# '''

##################################################################################################


def load_weights(iter_no, session, dir='pretrained_weights', tag='iter'):
    print ('trying to load iter weights...')
    for network in ['encoder', 'decoder']:
        path = os.path.join(dir, '{}_{}-{}'.format(network, tag, iter_no))
        tf.train.Saver(params[network]).restore(session, path)
    return iter_no


def load_best_weights(iter_no, session, dir='pretrained_weights'):
    try:
        print ('trying to load best weights...')
        tf.train.Saver(param_decoder).restore(session, '%s/decoder_best-%d' % (dir, iter_no))
        tf.train.Saver(param_encoder).restore(session, '%s/encoder_best-%d' % (dir, iter_no))
        tf.train.Saver(param_disc).restore(session, '%s/disc_best-%d' % (dir, iter_no))
    except Exception as ex:
        traceback.print_exc()
        print('Could not load best weight... Trying to load Iter...')
        # load_weights(iter_no, session)
    return iter_no
