import numpy as np
import tensorflow as tf

import layers
from tftools import calculate_class_weights, trainer
from tensorflow.contrib.tensorboard.plugins import projector


def encoder(inputs, mode, layer_params, scope="encoder", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        nb_layers = len(layer_params)
        L = nb_layers - 1

        z = {}
        z_pre = {}

        h = {}

        outputs = {'y': None, 'z_pre': list(), 'z': list()}

        with tf.variable_scope('input_layer'):
            z_pre[0] = inputs['x']
            h[0] = z[0] = layers.noise(tf.layers.batch_normalization(z_pre[0],
                                                                     training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                                     center=False,
                                                                     scale=False,
                                                                     name='batch_norm'),
                                       noise_pwr=layer_params[0]['noise'] * inputs['noise_mod'],
                                       name="noise")
        for it in range(L):
            i = it + 1  # first layer is input layer

            print(h[i-1])

            # Reshape
            if not np.asarray(layer_params[i]['input_shape'] == layer_params[i - 1]['output_shape']).all():
                with tf.variable_scope('reshape_layer_' + str(i - 1) + str(i)):
                    h[i - 1] = tf.reshape(h[i - 1], layer_params[i]['input_shape'])

            # Transformation
            with tf.variable_scope('enc_layer_' + str(i)):

                if layer_params[i]['type'] == 'conv':

                    k_size = layer_params[i]['kernel']
                    z_pre[i] = tf.layers.conv2d(tf.pad(h[i - 1],
                                                       [[0, 0],
                                                        [k_size - 1, k_size - 1],
                                                        [k_size - 1, k_size - 1],
                                                        [0, 0]]),
                                                filters=layer_params[i]['output_shape'][-1],
                                                kernel_size=[k_size, k_size],
                                                use_bias=False,
                                                name='conv_layer')

                elif layer_params[i]['type'] == 'fc':

                    z_pre[i] = tf.layers.dense(h[i - 1],
                                               units=layer_params[i]['output_shape'][-1],
                                               use_bias=False,
                                               name='dense_layer')

                elif layer_params[i]['type'] == 'maxpool':

                    k_size = layer_params[i]['kernel']
                    z_pre[i] = tf.layers.max_pooling2d(h[i - 1], k_size, k_size, padding='SAME')

                elif layer_params[i]['type'] == 'meanpool':

                    k_size = layer_params[i]['kernel']
                    z_pre[i] = tf.layers.average_pooling2d(h[i - 1], k_size, k_size, padding='SAME')

                else:
                    raise ValueError('Invalid layer type: \n - layer: '+str(i)+" \n - type: "+layer_params[i]['type'])

                # Normalization & Noise
                z[i] = layers.noise(tf.layers.batch_normalization(z_pre[i],
                                                                  training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                                  center=False,
                                                                  scale=False,
                                                                  name="batch_norm"),
                                    noise_pwr=layer_params[i]['noise'],
                                    name="noise")

                # Rescale
                with tf.variable_scope('activation'):
                    activation_shape = layer_params[i]['output_shape'][-1]
                    beta = tf.get_variable('beta',
                                           shape=[activation_shape],
                                           initializer=tf.zeros_initializer(),
                                           trainable=layer_params[i]['center'])
                    gamma = tf.get_variable('gamma',
                                            shape=[activation_shape],
                                            initializer=tf.ones_initializer(),
                                            trainable=layer_params[i]['scale'])

                    # Activation
                    h[i] = layer_params[i]['activation'](gamma * (z[i] + beta))

        outputs['z_pre'] = z_pre
        outputs['z'] = z
        outputs['y'] = h[L]
    return outputs


def decoder(inputs, mode, layer_params, scope="decoder", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):

        z_est = {}
        u_pre = {}
        u = {}
        outputs = {}

        nb_layers = len(layer_params)
        L = nb_layers - 1

        u[L] = inputs['y']
        z_corr = inputs['z']

        for it in range(L):
            i = L - it  # first layer is input layer

            with tf.variable_scope('dec_layer_' + str(i)):

                # Denoise layer
                z_est[i] = layers.denoise(u[i], z_corr[i], "denoise")
                # Transformation
                if layer_params[i]['type'] == 'conv':
                    k_size = layer_params[i]['kernel']
                    u_pre[i - 1] = tf.layers.conv2d_transpose(z_est[i],
                                                              filters=layer_params[i]['input_shape'][-1],
                                                              kernel_size=[k_size, k_size],
                                                              use_bias=False,
                                                              name="deconv_layer")[:, k_size - 1:-k_size + 1, k_size - 1:-k_size + 1, :]

                elif layer_params[i]['type'] == 'maxpool' or layer_params[i]['type'] == 'meanpool':
                    k_size = layer_params[i]['kernel']
                    u_pre[i - 1] = tf.layers.conv2d_transpose(z_est[i],
                                                              filters=layer_params[i]['input_shape'][-1],
                                                              kernel_size=k_size,
                                                              strides=[k_size, k_size],
                                                              use_bias=False,
                                                              kernel_initializer=tf.ones_initializer(),
                                                              trainable=False
                                                              )

                elif layer_params[i]['type'] == 'fc':
                    u_pre[i - 1] = tf.layers.dense(z_est[i],
                                                   units=layer_params[i]['input_shape'][-1],
                                                   use_bias=False,
                                                   name='dense_layer')
                else:
                    raise ValueError('Invalid layer type: \n - layer: '+str(i)+" \n - type: "+layer_params[i]['type'])

                # Normalization
                u[i - 1] = tf.layers.batch_normalization(u_pre[i - 1],
                                                         training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                         center=False,
                                                         scale=False,
                                                         name="batch_norm_layer")
            print(u[i-1])

            if not np.asarray(layer_params[i]['input_shape'] == layer_params[i - 1]['output_shape']).all():
                with tf.variable_scope('reshape_layer_' + str(i) + str(i - 1)):
                    u[i - 1] = tf.reshape(u[i - 1], layer_params[i - 1]['output_shape'])

        with tf.variable_scope('output_layer'):

            z_est[0] = layers.denoise(u[0], z_corr[0], name="denoise")

        outputs['z_est'] = z_est

    return outputs


def ladder_fn(features, labels, mode, params):
    labels = tf.one_hot(labels, depth=10)
    x_mean = params['x_mean']
    x_std = params['x_std']
    layer_params = params['layer_params']

    with tf.variable_scope('ladder'):
        x = features['inputs']
        x_norm = (x - x_mean) / x_std

        inputs = {'x': x_norm}

        with tf.variable_scope('encoder') as enc_scope:
            enc_inputs = {'x': inputs['x'],
                          'noise_mod': tf.zeros([], dtype=tf.float32)}
            enc_clean = encoder(enc_inputs, mode, layer_params=layer_params, scope=enc_scope)

            enc_inputs['noise_mod'] = tf.ones([], dtype=tf.float32)
            enc_noisy = encoder(enc_inputs, mode, layer_params=layer_params, scope=enc_scope, reuse=True)
            embedding = tf.Variable(tf.zeros_like(enc_clean['z'][len(layer_params)-2]), trainable=False, name="emb_code")
            embed_update = embedding.assign(enc_clean['z'][len(layer_params)-2])

            config = projector.ProjectorConfig()
            proj_embedding = config.embeddings.add()
            proj_embedding.tensor_name = embedding.name

        with tf.variable_scope('decoder') as dec_scope:
            dec_inputs = {}
            dec_inputs['z'] = enc_noisy['z']
            dec_inputs['y'] = enc_noisy['y']
            dec = decoder(dec_inputs, mode, layer_params=layer_params, scope=dec_scope)

        nb_layers = len(layer_params)

        # layerwise loss
        for i in range(nb_layers):
            with tf.variable_scope('loss_' + str(i)):
                z_pre_norm = tf.layers.batch_normalization(enc_clean['z'][i],
                                                           training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                           scale=False,
                                                           center=False,
                                                           name="batch_norm")
                z_est_norm = tf.layers.batch_normalization(dec['z_est'][i],
                                                           training=(mode == tf.estimator.ModeKeys.TRAIN),
                                                           trainable=False,
                                                           scale=False,
                                                           center=False,
                                                           reuse=True,
                                                           name="batch_norm")
                tf.losses.mean_squared_error(z_pre_norm, z_est_norm)

        # supervised loss
        with tf.variable_scope('loss_sup'):

            tf.losses.softmax_cross_entropy(labels, enc_noisy['y'])

    predictions = tf.one_hot(tf.argmax(enc_clean['y'], axis=1),
                             depth=layer_params[-1]['output_shape'][-1])

    x_est = dec["z_est"][0] * x_std + x_mean

    tf.summary.image("reconstruction", tf.reshape(x_est, [-1, 28, 28, 1]))

    loss_op = tf.losses.get_total_loss()
    train_op = trainer(loss_op, tf.train.get_global_step(), learning_rate=params['learning_rate'])

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels, predictions),
        "precision": tf.metrics.precision(labels, predictions),
        "mse": tf.metrics.mean_squared_error(x, x_est)
    }

    return tf.estimator.EstimatorSpec(mode,
                                      predictions={'predictions': predictions, 'embedding': enc_clean['z'][len(layer_params)-2]},
                                      loss=loss_op,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)
