from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

from ladder_model import ladder_fn
from tensorflow.examples.tutorials.mnist import input_data

tf.flags.DEFINE_string("output_dir", "", "Optional output dir.")
tf.flags.DEFINE_string("schedule", "train_and_evaluate", "Schedule.")
tf.flags.DEFINE_string("hparams", "", "Hyper parameters.")

FLAGS = tf.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

BATCH_SIZE = 100

layer_params = list()
layer_params.append({'type': 'input',
                     'input_shape': None,
                     'output_shape': np.array([BATCH_SIZE, 784]),
                     'activation': lambda x: x,
                     'gamma': 10,
                     'noise': 0.25})

layer_params.append({'type': 'conv',
                     'input_shape': np.array([BATCH_SIZE, 28, 28, 1]),
                     'output_shape': np.array([BATCH_SIZE, 32, 32, 32]),
                     'kernel': 5,
                     'activation': tf.nn.relu,
                     'gamma': 0.1,
                     'noise': 0.04,
                     'center': True,
                     'scale': False})

layer_params.append({'type': 'maxpool',
                     'input_shape': np.array([BATCH_SIZE, 32, 32, 32]),
                     'output_shape': np.array([BATCH_SIZE, 16, 16, 32]),
                     'activation': lambda x: x,
                     'kernel': 2,
                     'gamma': 0.1,
                     'noise': 0.04,
                     'center': True,
                     'scale': False})

layer_params.append({'type': 'conv',
                     'input_shape': np.array([BATCH_SIZE, 16, 16, 32]),
                     'output_shape': np.array([BATCH_SIZE, 18, 18, 64]),
                     'kernel': 3,
                     'activation': tf.nn.relu,
                     'gamma': 0.1,
                     'noise': 0.04,
                     'center': True,
                     'scale': False})

layer_params.append({'type': 'conv',
                     'input_shape': np.array([BATCH_SIZE, 18, 18, 64]),
                     'output_shape': np.array([BATCH_SIZE, 20, 20, 64]),
                     'kernel': 3,
                     'activation': tf.nn.relu,
                     'gamma': 0.1,
                     'noise': 0.04,
                     'center': True,
                     'scale': False})

layer_params.append({'type': 'maxpool',
                     'input_shape': np.array([BATCH_SIZE, 20, 20, 64]),
                     'output_shape': np.array([BATCH_SIZE, 10, 10, 64]),
                     'activation': lambda x: x,
                     'kernel': 2,
                     'gamma': 0.1,
                     'noise': 0.04,
                     'center': True,
                     'scale': False})

layer_params.append({'type': 'conv',
                     'input_shape': np.array([BATCH_SIZE, 10, 10, 64]),
                     'output_shape': np.array([BATCH_SIZE, 12, 12, 128]),
                     'kernel': 3,
                     'activation': tf.nn.relu,
                     'gamma': 0.1,
                     'noise': 0.04,
                     'center': True,
                     'scale': False})

layer_params.append({'type': 'conv',
                     'input_shape': np.array([BATCH_SIZE, 12, 12, 128]),
                     'output_shape': np.array([BATCH_SIZE, 14, 14, 10]),
                     'kernel': 3,
                     'activation': tf.nn.relu,
                     'gamma': 0.1,
                     'noise': 0.04,
                     'center': True,
                     'scale': False})

layer_params.append({'type': 'meanpool',
                     'input_shape': np.array([BATCH_SIZE, 14, 14, 10]),
                     'output_shape': np.array([BATCH_SIZE, 1, 1, 10]),
                     'kernel': 14,
                     'activation': lambda x: x,
                     'gamma': 0.01,
                     'noise': 0.04,
                     'center': False,
                     'scale': False})

layer_params.append({'type': 'fc',
                     'input_shape': np.array([BATCH_SIZE, 10]),
                     'output_shape': np.array([BATCH_SIZE, 10]),
                     'activation': lambda x: x,
                     'gamma': 0.01,
                     'noise': 0.04,
                     'center': True,
                     'scale': True})

params = {'learning_rate': 0.001,
          'x_mean': 0,
          'x_std': 1,
          'label_stats': np.asarray([0.5]),
          'layer_params': layer_params}

def main(unused_argv):

    mnist = input_data.read_data_sets("MNIST_data/")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"inputs": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"inputs": eval_data},
        y=eval_labels,
        batch_size=100,
        shuffle=False)

    config = tf.estimator.RunConfig()
    config.environment = None
    nn = tf.estimator.Estimator(model_fn=ladder_fn, params=params, model_dir="/tmp/mnist_img", config=config)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=2000)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, throttle_secs=300)

    tf.estimator.train_and_evaluate(nn, train_spec, eval_spec)
    #nn.evaluate(eval_input_fn,)

    # experiment = tf.contrib.learn.Experiment(estimator=nn,
    #                                          train_input_fn=train_input_fn,
    #                                          eval_input_fn=eval_input_fn,
    #                                          train_steps_per_iteration = 10,
    #                                          train_steps=2000
    #                                          )
    #
    # experiment.train_and_evaluate()

tf.app.run()
