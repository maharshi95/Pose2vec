import tensorflow as tf

class Hyperparams:
    dtype = tf.float32
    sk_transform_type = 'azimuth'

    best_iter_no = 2958501
    best_weights_tag = 'float32'
    best_weights_path = '' # value set outside the Hyperparams

Hyperparams.best_weights_path = 'pretrained_weights/%s' % Hyperparams.sk_transform_type
