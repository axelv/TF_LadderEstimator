import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.data as tfdata


def dir_to_run(x):
    try:
        return int(x)
    except:
        return -1


def variable_summaries(var, name="summaries", collections=[tf.GraphKeys.SUMMARIES]):

    """"Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name):
        var = tf.cast(var, tf.float32)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean, collections=collections)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev, collections=collections)
        tf.summary.scalar('max', tf.reduce_max(var), collections=collections)
        tf.summary.scalar('min', tf.reduce_min(var), collections=collections)
        tf.summary.histogram('histogram', var, collections=collections)


def csv_filereader_op(filename_queue, nb_features, nb_labels=1):
    reader =  tf.TextLineReader()
    _, csv_row = reader.read(filename_queue)
    record_defaults = np.ones([nb_features+nb_labels,1]).tolist()

    csv_record = tf.stack(tf.decode_csv(csv_row, record_defaults=record_defaults))
    features = csv_record[:nb_features]
    labels = csv_record[nb_features:]

    return features, labels

def get_path(base_path):
    max_run = -1
    if os.path.isdir(base_path):
        for entry in os.listdir(base_path):
            max_run = np.maximum(max_run, dir_to_run(entry))
        new_run = max_run + 1
    else:
        os.mkdir(base_path)
        new_run = max_run + 1
    return os.path.join(base_path, str(new_run))


def get_data_paths(basepath, prefix="", sufix="", recursive=False):
    files_path = list()
    for entry in os.listdir(basepath):
        if os.path.isdir(os.path.join(basepath, entry)) and recursive is True:
            new_basepath = os.path.join(basepath, entry)
            subfiles_path = get_data_paths(new_basepath, prefix=prefix, sufix=sufix, recursive=True)
            files_path += subfiles_path
        elif entry.startswith(prefix) and entry.endswith(sufix):
            files_path.append(os.path.join(basepath, entry))

    return files_path


def generate_batch(filenames, batch_size, nb_features, nb_labels, nb_epochs=None, nb_read_threads = 10, min_after_dequeue = 1000, name='batch_generator'):
    with tf.name_scope(name):
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=nb_epochs)
        features_list = [csv_filereader_op(filename_queue, nb_features, nb_labels) for _ in range(nb_read_threads)]
        capacity = min_after_dequeue + 3 * batch_size
        features_batch, labels_batch = tf.train.shuffle_batch_join(features_list, batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)
    return features_batch, labels_batch


def random_split(datalist, partition=0.8):
    np_list = np.asarray(datalist)
    n = len(np_list)
    data_order = np.arange(n)
    np.random.shuffle(data_order)
    split = np.floor(n*partition).astype('int64')

    return np_list[data_order[:split]], np_list[data_order[split:]]

def calculate_class_weights(labels, label_stats, name="class_weights"):
    with tf.name_scope(name):
        class_part = label_stats
        labels_flt = tf.cast(labels, dtype=tf.float32)
        weights = labels_flt * 1. / (2. * class_part + 1e-6) + (1. - labels_flt) * 1. / (2. * (1. - class_part) + 1e-6)
        weights = weights * (tf.clip_by_value(labels_flt, clip_value_min=-1., clip_value_max=0.) + 1.)
    return weights

def trainer(cost, global_step, learning_rate=0.01):
    with tf.variable_scope("train"):
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(extra_update_ops):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08,
                                               use_locking=False, name='Adam')
            train_op = optimizer.minimize(cost, global_step=global_step)
    return train_op

def sparse_harmonic_regularizer(weights):

    fft_weights = tf.spectral.fft(weights)
    l1_weights = tf.reduce_sum(tf.abs(fft_weights))
    return l1_weights

def interleave(*args):
    result = tfdata.Dataset.from_tensors(args[0])
    for i in range(1,len(args)):
        result = result.concatenate(tfdata.Dataset.from_tensors(args[i]))

    return result