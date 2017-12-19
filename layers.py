import numpy as np
import tensorflow as tf
from tensorflow.python.layers import base
from tensorflow.python.framework import tensor_shape
from tftools import variable_summaries


class Noise(base.Layer):
  """Noise layer class.

  This layer implements the operation:
  `outputs = inputs + noise`
  Where `noise` is sampled from a Gaussian distribution with zero mean and `noise_pwr` variance.

  Arguments:
    noise_pwr: Float, AC power of the gaussian noise.
    name: String, the name of the layer.

  Properties:
    noise_pwr: Python integer, variance of the Gaussian distribution.
  """

  def __init__(self,
               noise_pwr=1,
               name=None,
               **kwargs):
    super(Noise, self).__init__(name=name, **kwargs)
    self.noise_pwr = noise_pwr

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if input_shape[-1].value is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    self.input_spec = base.InputSpec(min_ndim=2,
                                     axes={-1: input_shape[-1].value})
    self.built = True

  def call(self, inputs,  **kwargs):
    inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
    noise = tf.random_normal(tf.shape(inputs),mean=0.0, stddev=self.noise_pwr)
    outputs = inputs+noise
    return outputs

  def _compute_output_shape(self, input_shape):
    if input_shape[-1].value is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % input_shape)
    return input_shape

def noise(inputs, noise_pwr, name="noise"):
    """Noise layer.

    This layer implements the operation:
    `outputs = inputs + noise`
    Where `noise` is sampled from a Gaussian distribution with zero mean and `noise_pwr` variance.

    Arguments:
      noise_pwr: Float, AC power of the gaussian noise.
      name: String, the name of the layer.

    """

    layer = Noise(noise_pwr=noise_pwr,
                  name=name,
                  dtype=inputs.dtype.base_dtype,
                  _scope=name)

    return layer.apply(inputs)



def noise_layer(x, noise_pwr):
    with tf.variable_scope("noise_layer"):
        n = tf.random_normal(shape=tf.shape(x), stddev=tf.sqrt(noise_pwr))
        y=x+n
    return y

class Denoise(base.Layer):
    """Denoise layer class.

    This layer implements the operation:
    `outputs = denoise_fct(corrupted,latent)`
    Where `la` is sampled from a Gaussian distribution with zero mean and `noise_pwr` variance.

    Arguments:
      name: String, the name of the layer.

    Properties:
    """

    def __init__(self,
                 name=None,
                 trainable=True,
                 **kwargs):
        super(Denoise, self).__init__(name=name, trainable=trainable, **kwargs)
        self.a1 = tf.zeros_initializer()
        self.a2 = tf.ones_initializer()
        self.a3 = tf.zeros_initializer()
        self.a4 = tf.zeros_initializer()
        self.a5 = tf.zeros_initializer()

        self.a6 = tf.zeros_initializer()
        self.a7 = tf.ones_initializer()
        self.a8 = tf.zeros_initializer()
        self.a9 = tf.zeros_initializer()
        self.a10 = tf.zeros_initializer()

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list):
            raise ValueError('A denoise layer should be called ' 'on a list of inputs.')
        if len(input_shape) != 2:
            raise ValueError('A denois layer should be called '
                             'on a list of 2 inputs. '
                             'Got ' + str(len(input_shape)) + ' inputs.')
        if input_shape[0][1:] != input_shape[1][1:]:
            raise ValueError('Can not denoise tensors with different '
                             'shapes. Got tensors with shapes : ' +
                             str(input_shape))
        if input_shape[0][-1].value is None:
            raise ValueError(
                'The innermost dimension of input_shape must be defined, but saw: %s'
                % input_shape)

        channels = input_shape[0][-1]
        wi = lambda inits, name: self.add_variable(name,
                                                   shape=[channels],
                                                   initializer=tf.constant_initializer(inits),
                                                   trainable=True,
                                                   dtype=self.dtype)
        self.a1 = wi(0., 'a1')
        self.a2 = wi(1., 'a2')
        self.a3 = wi(0., 'a3')
        self.a4 = wi(0., 'a4')
        self.a5 = wi(0., 'a5')

        self.a6 = wi(0., 'a6')
        self.a7 = wi(1., 'a7')
        self.a8 = wi(0., 'a8')
        self.a9 = wi(0., 'a9')
        self.a10 = wi(0., 'a10')

        self.built = True

    def call(self, inputs, **kwargs):

        u = tf.convert_to_tensor(inputs[0], dtype=self.dtype)
        z_corr = tf.convert_to_tensor(inputs[1], dtype=self.dtype)

        mu = self.a1 * tf.sigmoid(self.a2 * u + self.a3) + self.a4 * u + self.a5
        v = self.a6 * tf.sigmoid(self.a7 * u + self.a8) + self.a9 * u + self.a10
        z_est = (z_corr - mu) * v + mu

        return z_est

    def _compute_output_shape(self, input_shape):

        return input_shape[1]

def denoise(latent, corrupted, name="denoise", trainable=True):
    layer = Denoise(name=name,
                    trainable=trainable,
                    dtype=latent.dtype.base_dtype,
                    _scope=name)

    return layer.apply((latent, corrupted))


def denoise_layer(u, z_corr, axis=-1, collections=[tf.GraphKeys.SUMMARIES]):
    with tf.variable_scope("denoise_layer"):
        shape = u.get_shape()
        wi = lambda inits, name: tf.get_variable(name, initializer=inits * tf.ones(shape[axis]))
        a1 = wi(0., 'a1')
        a2 = wi(1., 'a2')
        a3 = wi(0., 'a3')
        a4 = wi(0., 'a4')
        a5 = wi(0., 'a5')

        a6 = wi(0., 'a6')
        a7 = wi(1., 'a7')
        a8 = wi(0., 'a8')
        a9 = wi(0., 'a9')
        a10 = wi(0., 'a10')

        mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
        v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10
        z_est = (z_corr - mu) * v + mu

    return z_est


def siamese_layer(y1, y2):
    with tf.variable_scope("siamese_layer"):
        e2 = tf.square(y1 - y2)
        d2 = tf.reduce_sum(e2, axis=1)
        variable_summaries(d2, 'distance')
    return d2


def flatten(x, shape):
    with tf.variable_scope("flatten"):
        shape_prod = shape[1] * shape[2]
        y = tf.reshape(x, [-1, shape_prod])
    return y


def pool_layer(x, ksize, stride):
    with tf.variable_scope("pool"):

        ksize = strides = [1, 1, ksize, 1]
        y =  tf.nn.max_pool(tf.expand_dims(x, 1), ksize=ksize, strides=strides, padding='SAME')

        return tf.squeeze(y, axis=1)


def unpool_layer(x, ksize, output_shape):
    with tf.variable_scope("unpool"):
        y = tf.layers.conv2d_transpose(x,
                                       filters=1,
                                       kernel_size=ksize,
                                       strides=[ksize, ksize],
                                       use_bias=False,
                                       kernel_initializer=tf.ones_initializer(),
                                       trainable=False
                                       )
        shape = output_shape.copy()
        shape[1] = np.ceil(shape[1]/ksize)*ksize
        x = tf.transpose(tf.tile(tf.expand_dims(x, axis=-1), [1, 1, 1, ksize]),[0,2,1,3])
        y = tf.transpose(tf.reshape(x, [shape[0], shape[2], shape[1]]),[0,2,1])[:,:output_shape[1],:]
    return y


def unflatten(x, shape):
    with tf.variable_scope("unflatten"):
        y = tf.reshape(x, [-1, shape[1], shape[2]])
    return y


def zca_whitening(X):
    with tf.variable_scope("ZCA_whitening"):
        sigma = tf.matmul(tf.transpose(X), X)/tf.cast(tf.shape(X)[0], dtype=tf.float32) #Correlation matrix
        S, U, V = tf.svd(sigma, full_matrices=True) #Singular Value Decomposition
        epsilon = 0.1   #Whitening constant, it prevents division by zeros
        dummy_matrix = tf.diag(tf.constant([1.0])/tf.sqrt(S + tf.constant([epsilon])))
        ZCAMatrix = tf.matmul(tf.matmul(U, dummy_matrix), tf.transpose(U))  #ZCA Whitening matrix
        invZCAMatrix = tf.matrix_inverse(ZCAMatrix)
    return ZCAMatrix, invZCAMatrix
