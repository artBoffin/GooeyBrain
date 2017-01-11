from __future__ import absolute_import, division, print_function

import os

import numpy as np
import tensorflow as tf
import tflearn

from six.moves import range
from skimage import io

GENERATOR_OP_NAME = 'Generator'
DISCRIMINATOR_OP_NAME = 'Discriminator'
TENSORBOARD_DIR = '/tmp/tflearn_logs/'
MODEL_DIR = '/tmp/dcgan/'
CHECKPOINT_PATH = '/tmp/dcgan/ckpt/'

for path in [TENSORBOARD_DIR, MODEL_DIR, CHECKPOINT_PATH]:
    if not os.path.exists(path):
        os.mkdir(path)
class Trainer(tflearn.Trainer):
    def fit(self, feed_dicts, n_epoch=10, val_feed_dicts=None,
            show_metric=False, snapshot_step=None, snapshot_epoch=True,
            shuffle_all=None, dprep_dict=None, daug_dict=None,
            excl_trainops=None, run_id=None, callbacks=[]):
        id_generator = tflearn.utils.id_generator
        to_list = tflearn.utils.to_list
        standarize_dict = tflearn.utils.standarize_dict
        tf_callbacks = tflearn.callbacks
        get_dict_first_element = tflearn.utils.get_dict_first_element
        data_flow = tflearn.data_flow

        if not run_id:
            run_id = id_generator(6)
        print("---------------------------------")
        print("Run id: " + run_id)
        print("Log directory: " + self.tensorboard_dir)

        original_train_ops = list(self.train_ops)
        # Remove excluded train_ops
        for t in self.train_ops:
            if excl_trainops and t in excl_trainops:
                self.train_ops.remove(t)

        # shuffle is an override for simplicty, it will overrides every
        # training op batch shuffling
        if isinstance(shuffle_all, bool):
            for t in self.train_ops:
                t.shuffle = shuffle_all

        with self.graph.as_default():

            try:
                self.summ_writer = tf.train.SummaryWriter(
                    self.tensorboard_dir + run_id, self.session.graph)
            except Exception: # TF 0.7
                self.summ_writer = tf.train.SummaryWriter(
                    self.tensorboard_dir + run_id, self.session.graph_def)

            feed_dicts = to_list(feed_dicts)
            self.feed_dict_all = {}
            for d in feed_dicts:
                standarize_dict(d)
                self.feed_dict_all.update(d)

            termlogger = tf_callbacks.TermLogger()
            modelsaver = tf_callbacks.ModelSaver(self.save,
                                                 self.checkpoint_path,
                                                 self.best_checkpoint_path,
                                                 self.best_val_accuracy,
                                                 snapshot_step,
                                                 snapshot_epoch)

            ####################################################################
            # moved from TrainOp.initialize_fit
            self.n_train_samples = len(
                get_dict_first_element(self.feed_dict_all))

            self.index_array = np.arange(self.n_train_samples)

            self.train_dflow = data_flow.FeedDictFlow(self.feed_dict_all,
                self.coord, continuous=True,
                batch_size=self.train_ops[0].batch_size,
                index_array=self.index_array,
                num_threads=1,
                shuffle=self.train_ops[0].shuffle)

            self.n_batches = len(self.train_dflow.batches)
            self.train_dflow.start()
            ####################################################################

            for train_op in self.train_ops:
                # Prepare all train_ops for fitting
                train_op.initialize_fit(show_metric, self.summ_writer)
                train_op.train_dflow = self.train_dflow

                # Prepare TermLogger for training diplay
                metric_term_name = None
                if train_op.metric is not None:
                    if hasattr(train_op.metric, 'm_name'):
                        metric_term_name = train_op.metric.m_name
                    else:
                        metric_term_name = train_op.metric.name.split(':')[0]
                termlogger.add(self.n_train_samples,
                               metric_name=metric_term_name, name=train_op.name)

            max_batches_len = self.n_batches

            caller = tf_callbacks.ChainCallback(callbacks=[termlogger,
                                                           modelsaver])

            callbacks = to_list(callbacks)

            if callbacks:
                [caller.add(cb) for cb in callbacks]

            caller.on_train_begin(self.training_state)
            train_ops_count = len(self.train_ops)
            snapshot = snapshot_epoch

            try:
                for epoch in range(n_epoch):

                    self.training_state.increaseEpoch()

                    caller.on_epoch_begin(self.training_state)

                    # Global epoch are defined as loop over all data (whatever
                    # which data input), so one epoch loop in a multi-inputs
                    # model is equal to max(data_input) size.
                    for batch_step in range(self.n_batches):

                        self.training_state.increaseStep()
                        self.training_state.resetGlobal()

                        caller.on_batch_begin(self.training_state)

                        ########################################################
                        # moved from TrainOp._train

                        feed_batch_all = self.train_dflow.next()
                        snapshot_epoch = False
                        if epoch != self.train_dflow.data_status.epoch:
                            if bool(self.best_checkpoint_path) | snapshot_epoch:
                                snapshot_epoch = True
                        for t in feed_batch_all:
                            if t.name.startswith('input_z'):
                                feed_batch_all[t] = np.random.uniform(low=-1.0, high=1.0, size=feed_batch_all[t].shape)
                        ########################################################

                        for i, train_op in enumerate(self.train_ops):

                            caller.on_sub_batch_begin(self.training_state)

                            feed_batch = {t:feed_batch_all[t] for t in feed_dicts[i]}

                            snapshot = train_op._train(self.training_state.step,
                                snapshot_epoch, snapshot_step, show_metric,
                                epoch, feed_batch)

                            # Update training state
                            self.training_state.update(train_op, train_ops_count)

                            # Optimizer batch end
                            caller.on_sub_batch_end(self.training_state, i)

                        # All optimizers batch end
                        self.session.run(self.incr_global_step)
                        caller.on_batch_end(self.training_state, snapshot)

                    # Epoch end
                    caller.on_epoch_end(self.training_state)

            finally:
                caller.on_train_end(self.training_state)
                self.train_dflow.interrupt()
                # Set back train_ops
                self.train_ops = original_train_ops

class TrainOp(tflearn.TrainOp):
    def initialize_fit(self, show_metric, summ_writer):
        self.summary_writer = summ_writer

        self.create_testing_summaries(show_metric, self.metric_summ_name, None)

    def _train(self, training_step, snapshot_epoch, snapshot_step, show_metric,
               epoch, feed_batch, snapshot=False):
        summaries = tflearn.helpers.summarizer.summaries

        self.loss_value, self.acc_value = None, None
        train_summ_str = None

        tflearn.is_training(True, session=self.session)
        _, train_summ_str = self.session.run([self.train, self.summ_op],
                                             feed_batch)

        # Retrieve loss value from summary string
        sname = "- Loss/" + self.scope_name
        self.loss_value = summaries.get_value_from_summary_string(
            sname, train_summ_str)

        if show_metric and self.metric is not None:
            # Retrieve accuracy value from summary string
            sname = "- " + self.metric_summ_name + "/" + self.scope_name
            self.acc_value = summaries.get_value_from_summary_string(
                sname, train_summ_str)

        if snapshot_epoch:
            snapshot = True

        # Check if step reached snapshot step
        if snapshot_step:
            if training_step % snapshot_step == 0:
                snapshot = True

        # Write to Tensorboard
        n_step = self.training_steps.eval(session=self.session)
        if n_step > 1:
            if train_summ_str:
                self.summary_writer.add_summary(train_summ_str, n_step)

        return snapshot


class DCGAN:
    def __init__(self, img_shape=None, n_first_filter=None, n_layer=None,
                 dim_z=None, activation='LeakyReLU'):
        self.img_shape = list(img_shape)
        self.n_first_filter = n_first_filter
        self.n_layer = n_layer
        self.dim_z = dim_z
        self.activation = activation
        self.initializer = tflearn.initializations.truncated_normal(mean=0.0,
            stddev=0.02, dtype=tf.float16)
        self.weight_decay_gen = 0.0001
        self.weight_decay_dis = 0.0001
        self.trainer = None
        self.generator = None
        self._build()

    def _build(self, batch_size=64):
        gen = Generator(output_shape=self.img_shape,
            n_first_filter=self.n_first_filter * 2 ** (self.n_layer - 1),
            n_layer=self.n_layer, initializer=self.initializer,
            weight_decay=self.weight_decay_gen, scope='G')
        dis = Discriminator(n_first_filter=self.n_first_filter,
            n_layer=self.n_layer, initializer=self.initializer,
            weight_decay=self.weight_decay_dis, activation=self.activation,
            scope='D')

        input_z = tflearn.input_data(shape=(None, self.dim_z), name='input_z')

        generated = gen(input_z)
        self.generator = tflearn.DNN(generated)

        prediction_generated = dis(generated)

        y_gen = tflearn.input_data(shape=(None, 2), name='y_generator')
        loss_gen = tflearn.categorical_crossentropy(prediction_generated, y_gen)

        inputs = tflearn.input_data(shape=[None] + self.img_shape,
                                    name='input_origin')
        inputs = inputs * 2 - 1

        prediction_origin = dis(inputs, reuse=True)
        prediction_all = tflearn.merge(
            [prediction_origin, prediction_generated], 'concat', axis=0)

        y_dis_origin = tflearn.input_data(shape=(None, 2),
                                          name='y_discriminator_origin')
        y_dis_gen = tflearn.input_data(shape=(None, 2),
                                       name='y_discriminator_generated')
        y_dis = tflearn.merge([y_dis_origin, y_dis_gen], 'concat', axis=0)

        loss_dis = tflearn.categorical_crossentropy(prediction_all, y_dis)

        # print([v.name for v in tflearn.get_all_trainable_variable()])
        trainable_variables = tflearn.get_all_trainable_variable()
        self.generator_variables = [v for v in trainable_variables
                                    if gen.scope + '/' in v.name]
        self.discriminator_variables = [v for v in trainable_variables
                                        if dis.scope + '/' in v.name]

        optimizer_gen = tflearn.Adam(learning_rate=0.001, beta1=0.5).get_tensor()
        optimizer_dis = tflearn.Adam(learning_rate=0.0001, beta1=0.5).get_tensor()
        gen_train_op = tflearn.TrainOp(loss_gen, optimizer_gen, batch_size=batch_size,
                               trainable_vars=self.generator_variables,
                               name=GENERATOR_OP_NAME)
        dis_train_op = tflearn.TrainOp(loss_dis, optimizer_dis, batch_size=batch_size,
                               trainable_vars=self.discriminator_variables,
                               name=DISCRIMINATOR_OP_NAME)
        self.trainer = tflearn.Trainer([gen_train_op, dis_train_op],
                               tensorboard_dir=TENSORBOARD_DIR,
                               checkpoint_path=CHECKPOINT_PATH,
                               max_checkpoints=2, tensorboard_verbose=3,
                               keep_checkpoint_every_n_hours=0.5)

    def _get_tensor_by_name(self, name):
        return tf.get_collection(tf.GraphKeys.INPUTS, scope=name)[0]

    def train(self, x, n_sample):
        zeros = np.zeros((n_sample, 1), dtype=np.uint8)
        ones = np.ones((n_sample, 1), dtype=np.uint8)
        should_true = np.concatenate((zeros, ones), axis=1)
        should_false = np.concatenate((ones, zeros), axis=1)

        input_origin = self._get_tensor_by_name('input_origin')
        input_z = self._get_tensor_by_name('input_z')
        y_generator = self._get_tensor_by_name('y_generator')
        y_discriminator_origin = self._get_tensor_by_name('y_discriminator_origin')
        y_discriminator_generated = self._get_tensor_by_name('y_discriminator_generated')
        zero_matrix = np.zeros((n_sample, self.dim_z))

        self.feed_dict_gen = {input_z:zero_matrix, y_generator:should_true}
        self.feed_dict_dis = {input_origin:x, input_z:zero_matrix,
                              y_discriminator_origin:should_true,
                              y_discriminator_generated:should_false}

        self.trainer.fit(feed_dicts=[self.feed_dict_gen, self.feed_dict_dis],
                         n_epoch=1000, snapshot_step=1000,
                         callbacks=CustomCallback(self),
                         run_id='DCGAN-Training')

    def generate(self, z):
        output = self.generator.predict(z)
        output = np.clip(output, -1, 1)
        output = (output + 1) * 0.5 * 255

        return output.astype(np.uint8)

class Generator(object):
    def __init__(self, output_shape, n_first_filter, n_layer, initializer,
                 weight_decay, scope):
        self.output_channel = output_shape[2]
        self.first_height = output_shape[0] // 2 ** (n_layer - 1)
        self.first_width = output_shape[1] // 2 ** (n_layer - 1)
        self.first_filter = n_first_filter
        self.first_node = self.first_height * self.first_width * self.first_filter
        self.first_shape = [-1, self.first_height, self.first_width, self.first_filter]
        self.n_layer = n_layer
        self.initializer = initializer
        self.weight_decay = weight_decay
        self.scope = scope

    def __call__(self, incoming, reuse=False):
        height = self.first_height
        width = self.first_width
        filter_ = self.first_filter

        net = incoming

        with tf.variable_scope(self.scope):
            net = tflearn.fully_connected(net, self.first_node,
                                          weights_init=self.initializer,
                                          weight_decay=self.weight_decay)
            net = tflearn.reshape(net, self.first_shape)

            for i in range(self.n_layer - 1):
                height *= 2
                width *= 2
                if i < self.n_layer - 2:
                    filter_ //= 2
                else:
                    filter_ = self.output_channel
                net = tflearn.batch_normalization(net)
                net = tflearn.relu(net)
                net = tflearn.conv_2d_transpose(net, filter_, 4,
                                                [height, width], strides=2,
                                                weights_init=self.initializer,
                                                weight_decay=self.weight_decay)

        return tflearn.tanh(net)

class Discriminator(object):
    def __init__(self, n_first_filter, n_layer, initializer, weight_decay,
                 activation, scope):
        self.n_first_filter = n_first_filter
        self.n_layer = n_layer
        self.initializer = initializer
        self.weight_decay = weight_decay
        self.activation = activation
        self.scope = scope

    def __call__(self, incoming, reuse=False):
        net = incoming

        for i in range(self.n_layer):
            net = tflearn.conv_2d(net, self.n_first_filter * 2 ** i, 4,
                strides=2, weights_init=self.initializer,
                weight_decay=self.weight_decay, reuse=reuse,
                scope='{s}/Conv2D_{n}'.format(s=self.scope, n=i))
            net = tflearn.batch_normalization(net, reuse=reuse,
                scope='{s}/BatchNormalization_{n}'.format(s=self.scope, n=i))
            net = tflearn.activation(net, self.activation)

        net = tflearn.fully_connected(net, 2, weights_init=self.initializer,
            weight_decay=self.weight_decay, reuse=reuse,
            scope='{s}/FullyConnected'.format(s=self.scope))

        return tflearn.softmax(net)

class CustomCallback(tflearn.callbacks.Callback):
    def __init__(self, dcgan, epoch=0):
        self.dcgan = dcgan
        self.epoch = epoch
        self.n_side = 10
        self.sample_z = np.random.uniform(low=-1.0, high=1.0,
            size=(self.n_side ** 2, self.dcgan.dim_z))

    def _save(self, file_name):
        gen_imgs = self.dcgan.generate(self.sample_z)
        img_height = self.dcgan.img_shape[0]
        img_width = self.dcgan.img_shape[1]
        img_channel = self.dcgan.img_shape[2]
        image = np.ndarray(shape=(self.n_side * img_height, self.n_side * img_width, img_channel),
                           dtype=np.uint8)
        for y in range(self.n_side):
            for x in range(self.n_side):
                image[y * img_height : (y + 1) * img_height,
                      x * img_width  : (x + 1) * img_width,
                      :] = gen_imgs[x + y * self.n_side]
        io.imsave(file_name + '.png', image)

    def on_batch_end(self, training_state, snapshot=False):
        if snapshot:
            self.dcgan.generator.load('{dir}-{step}'.format(dir=CHECKPOINT_PATH, step=training_state.step))
            file_name = '{dir}step{n}'.format(dir=MODEL_DIR,
                                               n=training_state.step)
            self._save(file_name)



