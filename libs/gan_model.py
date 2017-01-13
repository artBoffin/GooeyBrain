import argparse
import os
import time
import numpy as np
import tensorflow as tf

from libs.utils import *


class Parameter(object):
    def __init__(self, name, value, p_type=None,
                 p_min=0, p_max=0, step=0, description="",
                 size_change=False, list_type=None, is_path=False):
        self.name = name
        self.value = value
        self.type = p_type if p_type else type(value)
        self.min = p_min
        self.max = p_max
        self.step = step
        self.description = description
        self.size_change = size_change
        self.list_type = list_type
        self.is_path = is_path

    def getJson(self):
        json = {
            'name': self.name,
            'type': self.type.__name__,
            'value': self.value,
            'min': self.min,
            'max': self.max,
            'step': self.step,
            'description': self.description,
            'size_change': self.size_change,
            'is_path': self.is_path
        }
        if self.type==list:
            json['list_type'] = self.list_type.__name__
        return json


class GAN(object):
    parameters = [
        Parameter("files_path", "./data", str, description="path to files", is_path=True),
        Parameter("convolutional", True, bool, description="is convolutional network (true for DCGAN)"),
        Parameter("is_grayscale", False, bool, description="is image grayscale? (default rgb)"),
        Parameter("input_h", 64, int, 1, 1000, 1, "height of input image"),
        Parameter("input_w", 64, int, 1, 1000, 1, "width of input image"),
        Parameter("output_h", 64, int, 1, 1000, 1, "height of output image"),
        Parameter("output_w", 64, int, 1, 1000, 1, "width of output image"),
        Parameter("crop_factor", 1.0, float, 0.01, 1, 0.01, "percentage of image to crop (zoom in)"),
        Parameter("learning_rate", 0.0002, p_type=float, p_min=0.00000001, p_max=0.01, step=0.00000001,
                  description="leraning rate"),
        Parameter("beta1", 0.5, p_type=float, p_min=0.0, p_max=1, step=0.0001, description="beta1 for Adam Optimizer"),
        Parameter("batch_size", 64, p_type=int, p_min=2, p_max=500, step=1, description="batch size"),
        Parameter("n_epochs", 25, int, 1, 500, 1, "number of epochs"),
        Parameter("n_examples", 64, int, 1, 200, 1, "number of examples (sample size)"),
        Parameter("z_dim", 100, int, 1, 1000, 1, "size of z input (number of latent inputs for generator)"),
        Parameter("n_features", 32, int, 1, 512, 1, "Number of channels to use in the last hidden layer"),
        Parameter("sample_step", 5, int, 1, 1000, 1, "save sample images every X steps"),
        Parameter("save_step", 50, int, 1, 1000, 1, "save model file every X steps"),
        #Parameter("train_size", float("inf"), float, 1, float("inf"), 1, "limit the files to use for training?")
        Parameter("save_path", "./tmp", str, description="path to sve model files", is_path=True),
        Parameter("run_name", "gan_%s" % time.strftime("%Y%m%d-%H%M%S"), str,
                  description="name of this run for creating relevant folders")
    ]

    def __init__(self, sess, params):
        """
        Args:
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]

        """
        self.train_size = float("inf")

        self.sess = sess
        self.image_size = params['input_h']
        self.output_size = params['output_h']

        self.files_path = params['files_path']
        self.learning_rate = params['learning_rate']
        self.beta1 = params['beta1']
        self.batch_size = params['batch_size']
        self.n_epochs = params['n_epochs']
        self.sample_size = params['n_examples']
        self.z_dim = params['z_dim']
        self.n_features = params['n_features']
        self.is_grayscale = params['is_grayscale']
        self.n_channels = 1 if self.is_grayscale else 3
        self.input_shape = [params['input_h'], params['input_w'], self.n_channels]
        self.output_shape = [params['output_h'], params['output_w'], self.n_channels]
        self.crop_factor = params['crop_factor']
        self.is_crop = self.crop_factor != 1
        self.convolutional = params['convolutional']
        self.save_path = params['save_path']
        self.run_name = params['run_name']
        self.sample_step = params['sample_step']
        self.save_step = params['save_step']

        self.tensorboard_dir = params['tensorboard_dir'] if 'tensorboard_dir' in params else os.path.join(self.save_path, self.run_name, 'logs')
        self.sample_dir = self.model_dir = os.path.join(self.save_path, self.run_name, 'model')
        self.checkpoint_dir = os.path.join(self.save_path, self.run_name, 'ckpt')
        paths = [self.tensorboard_dir, self.model_dir, self.checkpoint_dir]
        print paths

        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

        self._build_model()

    def _build_model(self):
        self.images = tf.placeholder(tf.float32,
                                     [self.batch_size] + self.output_shape,
                                     name='real_images')
        self.images = (self.images / 127.5) - 1.0

        self.z = tf.placeholder(tf.float32, [None, self.z_dim],
                                name='z')

        self.sum_z = tf.summary.histogram("z", self.z)

        # Generator tries to recreate input samples using latent feature vector
        self.G = generator(
            self.z, phase_train=True, output_shape=self.output_shape,
            n_features=self.n_features, activation=tf.nn.relu, output_activation=tf.nn.tanh)

        # Discriminator for real input samples
        self.D_real, self.D_real_logits = discriminator(self.images, phase_train=True,
                                                        n_features=self.n_features,
                                                        convolutional=self.convolutional,reuse=False)

        # Discriminator for generated samples
        self.D_fake, self.D_fake_logits = discriminator(self.G, phase_train=True,
                                                        n_features=self.n_features,
                                                        convolutional=self.convolutional,
                                                        reuse=True)

        self.sampler = generator(self.z, phase_train=True, reuse=False, scope_name='sampler')

        with tf.variable_scope('loss'):
            self.loss_D_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(self.D_real_logits, tf.ones_like(self.D_real)))
            self.loss_D_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(self.D_fake_logits, tf.zeros_like(self.D_fake)))

            self.loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_fake_logits, tf.ones_like(self.D_fake)))

            self.loss_D = self.loss_D_real + self.loss_D_fake

        # Summaries
        self.sum_G = tf.summary.image("G", self.G)
        self.sum_D_real = tf.summary.histogram("D_real", self.D_real)
        self.sum_D_fake = tf.summary.histogram("D_fake", self.D_fake)

        self.sum_loss_D_real = tf.summary.scalar("loss_D_real", self.loss_D_real)
        self.sum_loss_D_fake = tf.summary.scalar("loss_D_fake", self.loss_D_fake)
        self.sum_loss_D = tf.summary.scalar("loss_D", self.loss_D)
        self.sum_loss_G = tf.summary.scalar("loss_G", self.loss_G)

        t_vars = tf.trainable_variables()

        self.vars_d = [v for v in t_vars if v.name.startswith('discriminator')]
        self.vars_g = [v for v in t_vars if v.name.startswith('generator')]

        self.saver = tf.train.Saver()

    def train(self):
        print (" [* ] starting train")
        files = []
        img_types = ['.jpg', '.jpeg', '.png']
        for root, dirnames, filenames in os.walk('./data/1'):
            for filename in filenames:
                if any([filename.endswith(type_str) for type_str in img_types]):
                    files.append(os.path.join(root, filename))
        data = files
        np.random.shuffle(data)
        print ("input data: %d files"%len(data))

        opt_g = tf.train.AdamOptimizer(self.learning_rate, name='Adam_g').minimize(
            self.loss_G, var_list=self.vars_g)
        opt_d = tf.train.AdamOptimizer(self.learning_rate, name='Adam_d').minimize(
            self.loss_D, var_list=self.vars_d)

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        self.G_sum_op = tf.summary.merge([self.sum_z, self.sum_D_fake,
                                          self.sum_G, self.sum_loss_D_fake, self.sum_loss_G])
        self.D_sum_op = tf.summary.merge([self.sum_z,
                                          self.sum_D_real,
                                          self.sum_loss_D_real,
                                          self.sum_loss_D])

        self.writer = tf.summary.FileWriter(self.tensorboard_dir, self.sess.graph)

        zs= np.random.uniform(-1, 1, size=(self.sample_size, self.z_dim))

        sample_files = data[0:self.sample_size]
        sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop, resize_w=self.output_size,
                            is_grayscale=self.is_grayscale) for sample_file in sample_files]
        if (self.is_grayscale):
            sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_images = np.array(sample).astype(np.float32)

        save_images(sample_images, [8, 8],
                    '{}/test_samples.png'.format(self.model_dir))

        counter = 1
        start_time = time.time()

        if self.load():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        step_i, t_i = 0, 0
        # loss_d = 1
        # loss_g = 1
        # n_loss_d, total_loss_d = 1, 1
        # n_loss_g, total_loss_g = 1, 1
        for epoch in xrange(self.n_epochs):
            batch_idxs = min(len(data), self.train_size) // self.batch_size
            np.random.shuffle(data)
            for idx in xrange(0, batch_idxs):
                batch_files = data[idx*self.batch_size:(idx+1)*self.batch_size]
                batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop, resize_w=self.output_size,
                                   is_grayscale=self.is_grayscale) for batch_file in batch_files]
                if (self.is_grayscale):
                    batch_xs = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_xs = np.array(batch).astype(np.float32)
                batch_zs = np.random.uniform(
                    -1.0, 1.0, [self.batch_size, self.z_dim]).astype(np.float32)

                # this_lr_g = min(1e-2, max(1e-6, init_lr_g * (loss_g / loss_d) ** 2))
                # this_lr_d = min(1e-2, max(1e-6, init_lr_d * (loss_d / loss_g) ** 2))

                # Update D network
                loss_d, _, sum_d = self.sess.run([self.loss_D, opt_d, self.D_sum_op],
                                                 feed_dict={self.images: batch_xs,
                                                            self.z: batch_zs})
                # total_loss_d += loss_d
                # n_loss_d += 1
                self.writer.add_summary(sum_d, step_i)

                # Update G network
                loss_g, _, sum_g = self.sess.run([self.loss_G, opt_g, self.G_sum_op],
                                                 feed_dict={self.z: batch_zs})
                # total_loss_g += loss_g
                # n_loss_g += 1
                self.writer.add_summary(sum_g, step_i)

                # update G again - Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                loss_g, _, sum_g = self.sess.run([self.loss_G, opt_g, self.G_sum_op],
                                                 feed_dict={self.z: batch_zs})
                # total_loss_g += loss_g
                # n_loss_g += 1
                self.writer.add_summary(sum_g, step_i)

                step_i += 1
                counter += 1
                print('Epoch: [%2d] [%3d/%3d] time: %4.4f d loss: %08.06f,  ' %
                      (epoch, idx, batch_idxs, time.time() - start_time, loss_d) +
                      'g* = loss: %08.06f' % loss_g)

                if np.mod(counter, self.sample_step) == 0:
                    samples, d_loss, g_loss = self.sess.run(
                        [self.sampler, self.loss_D, self.loss_G],
                        feed_dict={self.z: zs, self.images: sample_images}
                    )
                    save_images(samples, [8, 8],
                                '{}/train_{:02d}_{:04d}.png'.format(self.model_dir, epoch, idx))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                    t_i += 1

                if np.mod(counter, self.save_step) == 0:
                    self.save(counter)

    def save(self, step):
        self.saver.save(self.sess,
                        os.path.join(self.checkpoint_dir, self.run_name),
                        global_step=step)

    def load(self):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(ckpt_name))
            return True
        else:
            print(" [*] Failed to find a checkpoint")
            return False


def encoder(x, phase_train, dimensions=[], filter_sizes=[],
            convolutional=False, activation=tf.nn.relu,
            output_activation=tf.nn.sigmoid, reuse=False):
    """Encoder network codes input `x` to layers defined by dimensions.
    Parameters
    ----------
    x : tf.Tensor
        Input to the encoder network, e.g. tf.Placeholder or tf.Variable
    phase_train : tf.Placeholder
        Placeholder defining whether the network is in train mode or not.
        Used for changing the behavior of batch normalization which updates
        its statistics during train mode.
    dimensions : list, optional
        List of the number of neurons in each layer (convolutional=False) -or-
        List of the number of filters in each layer (convolutional=True), e.g.
        [100, 100, 100, 100] for a 4-layer deep network with 100 in each layer.
    filter_sizes : list, optional
        List of the size of the kernel in each layer, e.g.:
        [3, 3, 3, 3] is a 4-layer deep network w/ 3 x 3 kernels in every layer.
    convolutional : bool, optional
        Whether or not to use convolutional layers.
    activation : fn, optional
        Function for applying an activation, e.g. tf.nn.relu
    output_activation : fn, optional
        Function for applying an activation on the last layer, e.g. tf.nn.relu
    reuse : bool, optional
        For each layer's variable scope, whether to reuse existing variables.
    Returns
    -------
    h : tf.Tensor
        Output tensor of the encoder
    """
    # %%
    # ensure 2-d is converted to square tensor.
    if convolutional:
        x_tensor = to_tensor(x)
    else:
        x_tensor = tf.reshape(
            tensor=x,
            shape=[-1, dimensions[0]])
        dimensions = dimensions[1:]
    current_input = x_tensor

    for layer_i, n_output in enumerate(dimensions):
        with tf.variable_scope(str(layer_i), reuse=reuse):
            if convolutional:
                h , W= conv2d(
                    x=current_input,
                    n_output=n_output,
                    k_h=filter_sizes[layer_i],
                    k_w=filter_sizes[layer_i],
                    padding='SAME',
                    reuse=reuse)
            else:
                h, W = linear(
                    x=current_input,
                    n_output=n_output,
                    reuse=reuse)
            norm = bn(
                x=h,
                phase_train=phase_train,
                name='bn')
            output = activation(norm)

        current_input = output

    flattened = flatten(current_input, name='flatten', reuse=reuse)

    if output_activation is None:
        return flattened, flattened
    else:
        return output_activation(flattened), flattened


def decoder(z,
            phase_train,
            dimensions=[],
            channels=[],
            filter_sizes=[],
            convolutional=False,
            activation=tf.nn.relu,
            output_activation=tf.nn.tanh,
            reuse=None):
    """Decoder network codes input `x` to layers defined by dimensions.
    In contrast with `encoder`, this requires information on the number of
    output channels in each layer for convolution.  Otherwise, it is mostly
    the same.
    Parameters
    ----------
    z : tf.Tensor
        Input to the decoder network, e.g. tf.Placeholder or tf.Variable
    phase_train : tf.Placeholder
        Placeholder defining whether the network is in train mode or not.
        Used for changing the behavior of batch normalization which updates
        its statistics during train mode.
    dimensions : list, optional
        List of the number of neurons in each layer (convolutional=False) -or-
        List of the number of filters in each layer (convolutional=True), e.g.
        [100, 100, 100, 100] for a 4-layer deep network with 100 in each layer.
    channels : list, optional
        For decoding when convolutional=True, require the number of output
        channels in each layer.
    filter_sizes : list, optional
        List of the size of the kernel in each layer, e.g.:
        [3, 3, 3, 3] is a 4-layer deep network w/ 3 x 3 kernels in every layer.
    convolutional : bool, optional
        Whether or not to use convolutional layers.
    activation : fn, optional
        Function for applying an activation, e.g. tf.nn.relu
    output_activation : fn, optional
        Function for applying an activation on the last layer, e.g. tf.nn.relu
    reuse : bool, optional
        For each layer's variable scope, whether to reuse existing variables.
    Returns
    -------
    h : tf.Tensor
        Output tensor of the decoder
    """

    if convolutional:
        with tf.variable_scope('fc', reuse=reuse):
            z1, W = linear(
                x=z,
                n_output=channels[0] * dimensions[0][0] * dimensions[0][1],
                reuse=reuse)
            rsz = tf.reshape(
                z1, [-1, dimensions[0][0], dimensions[0][1], channels[0]])
            current_input = activation(
                features=bn(
                    name='bn',
                    x=rsz,
                    phase_train=phase_train,
                    reuse=reuse))

        dimensions = dimensions[1:]
        channels = channels[1:]
        filter_sizes = filter_sizes[1:]
    else:
        current_input = z

    for layer_i, n_output in enumerate(dimensions):
        with tf.variable_scope(str(layer_i), reuse=reuse):

            if convolutional:
                h, W = deconv2d(
                    x=current_input,
                    n_output_h=n_output[0],
                    n_output_w=n_output[1],
                    n_output_ch=channels[layer_i],
                    k_h=filter_sizes[layer_i],
                    k_w=filter_sizes[layer_i],
                    padding='SAME',
                    reuse=reuse)
            else:
                h, W = linear(
                    x=current_input,
                    n_output=n_output,
                    reuse=reuse)

            if layer_i < len(dimensions) - 1:
                norm = bn(
                    x=h,
                    phase_train=phase_train,
                    name='bn', reuse=reuse)
                output = activation(norm)
            else:
                output = h
        current_input = output

    if output_activation is None:
        return current_input
    else:
        return output_activation(current_input)


def generator(z, phase_train=True, output_shape=[64,64,3], convolutional=True,
              n_features=32, reuse=None, activation=tf.nn.relu6, output_activation=tf.nn.tanh,
              scope_name='generator'):
    """Simple interface to build a decoder network given the input parameters.

    Parameters
    ----------
    z : tf.Tensor
        Input to the generator, i.e. tf.Placeholder of tf.Variable
    phase_train : tf.Placeholder of type bool
        Whether or not the network should be trained (used for Batch Norm).
    output_h : int
        Final generated height
    output_w : int
        Final generated width
    convolutional : bool, optional
        Whether or not to build a convolutional generative network.
    n_features : int, optional
        Number of channels to use in the last hidden layer.
    rgb : bool, optional
        Whether or not the final generated image is RGB or not.
    reuse : None, optional
        Whether or not to reuse the variables if they are already created.

    Returns
    -------
    x_tilde : tf.Tensor
        Output of the generator network.
    """
    output_h = output_shape[0]
    output_w = output_shape[1]
    n_channels = output_shape[2]
    with tf.variable_scope(scope_name, reuse=reuse):
        return decoder(z=z,
                       phase_train=phase_train,
                       convolutional=convolutional,
                       filter_sizes=[5, 5, 5, 5, 5],
                       channels=[n_features * 8, n_features * 4,
                                 n_features * 2, n_features, n_channels],
                       dimensions=[
                           [output_h // 16, output_w // 16],
                           [output_h // 8, output_w // 8],
                           [output_h // 4, output_w // 4],
                           [output_h // 2, output_w // 2],
                           [output_h, output_w]]
                       if convolutional else [384, 512, n_features],
                       activation=activation,
                       output_activation=output_activation,
                       reuse=reuse)


def discriminator(x, phase_train=True, convolutional=True, n_features=32, reuse=False,
                  activation=lrelu, output_activation=tf.nn.sigmoid):
    with tf.variable_scope('discriminator', reuse=reuse):
        return encoder(x=x,
                       phase_train=phase_train,
                       convolutional=convolutional,
                       filter_sizes=[5, 5, 5, 5],
                       dimensions=[n_features, n_features * 2,
                                   n_features * 4, n_features * 8]
                       if convolutional
                       else [n_features, 128, 256],
                       activation=activation,
                       output_activation=output_activation,
                       reuse=reuse)