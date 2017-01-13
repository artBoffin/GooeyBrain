import argparse
import os
import time
import numpy as np
import tensorflow as tf

from libs.dataset_utils import create_input_pipeline
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
        Parameter("learning_rate", 0.0002, p_type=float, p_min=0.00000001, p_max=0.01, step=0.00000001,
                  description="leraning rate"),
        Parameter("beta1", 0.5, p_type=float, p_min=0.0, p_max=1, step=0.0001, description="beta1 for Adam Optimizer"),
        Parameter("batch_size", 64, p_type=int, p_min=2, p_max=500, step=1, description="batch size"),
        Parameter("n_epochs", 25, int, 1, 500, 1, "number of epochs"),
        Parameter("n_examples", 10, int, 1, 200, 1, "number of examples (sample size)"),
        Parameter("z_dim", 100, int, 1, 1000, 1, "size of z input (number of latent inputs for generator)"),
        Parameter("input_h", 64, int, 1, 1000, 1, "height of input image"),
        Parameter("input_w", 64, int, 1, 1000, 1, "width of input image"),
        Parameter("output_h", 64, int, 1, 1000, 1, "height of output image"),
        Parameter("output_w", 64, int, 1, 1000, 1, "width of output image"),
        Parameter("rgb", True, bool, description="is image rgb or greyscale"),
        Parameter("crop_factor", 1.0, float, 0.01, 1, 0.01, "percentage of image to crop (zoom in)"),
        Parameter("convolutional", True, bool, description="is convolutional network (true for DCGAN)"),
        Parameter("n_features", 32, int, 1, 512, 1, "Number of channels to use in the last hidden layer"),
        Parameter("save_path", "./tmp", str, description="path to sve model files", is_path=True),
        Parameter("run_name", "gan_%s" % time.strftime("%Y%m%d-%H%M%S"), str,
                  description="name of this run for creating relevant folders"),
        Parameter("sample_step", 5, int, 1, 1000, 1, "save sample images every X steps"),
        Parameter("save_step", 50, int, 1, 1000, 1, "save model file every X steps")
    ]

    def __init__(self, params):

        self.files_path = params['files_path']
        self.learning_rate = params['learning_rate']
        self.beta1 = params['beta1']
        self.batch_size = params['batch_size']
        self.n_epochs = params['n_epochs']
        self.sample_size = params['n_examples']
        self.z_dim = params['z_dim']
        self.n_features = params['n_features']
        self.rgb = params['rgb']
        self.n_channels = 3 if self.rgb else 1
        self.input_shape = [params['input_h'], params['input_w'], self.n_channels]
        self.output_shape = [params['output_h'], params['output_w'], self.n_channels]
        self.crop_factor = params['crop_factor']
        self.convolutional = params['convolutional']
        self.save_path = params['save_path']
        self.run_name = params['run_name']
        self.sample_step = params['sample_step']
        self.save_step = params['save_step']

        self.tensorboard_dir = os.path.join(self.save_path, self.run_name, 'logs')
        self.model_dir = os.path.join(self.save_path, self.run_name, 'model')
        self.checkpoint_dir = os.path.join(self.save_path, self.run_name, 'ckpt')
        paths = [self.tensorboard_dir, self.model_dir, self.checkpoint_dir]
        print paths

        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

        self._build_model()

    def _build_model(self):
        # place holders for images, samples, z
        # self.images = tf.placeholder(tf.float32, [self.batch_size] + self.output_shape,
        #                             name='real_images')
        # self.sample_images= tf.placeholder(tf.float32, [self.sample_size] + self.output_shape,
        #                                 name='sample_images')
        # self.z = tf.placeholder(tf.float32, [None, self.z_dim],
        #                         name='z')
        #
        # self.z_sum = tf.summary.histogram("z", self.z)

        # Real input samples
        # n_features is either the image dimension or flattened number of features
        x = tf.placeholder(tf.float32, [self.batch_size] + self.output_shape, 'x')
        x = (x / 127.5) - 1.0
        # sum_x = tf.summary.image("x", x)

        # Generator tries to recreate input samples using latent feature vector
        z = tf.placeholder(tf.float32, [None, self.z_dim], 'z')
        sum_z = tf.summary.histogram("z", z)

        phase_train = tf.placeholder(tf.bool, name='phase_train')

        self.G = generator(
            z, phase_train, output_shape=self.output_shape,
            n_features=self.n_features, activation=tf.nn.relu, output_activation=tf.nn.tanh)

        # Discriminator for real input samples
        D_real, D_real_logits = discriminator(x, phase_train, n_features=self.n_features,
                                      convolutional=self.convolutional,reuse=False)

        # Discriminator for generated samples
        D_fake, D_fake_logits = discriminator(self.G, phase_train, n_features=self.n_features,
                                      convolutional=self.convolutional, reuse=True)

        self.sum_G = tf.summary.image("G", self.G)

        with tf.variable_scope('loss'):
            # Loss functions
            # self.loss_D_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            #     D_real_logits, tf.ones_like(D_real), name='loss_D_real'))
            # self.loss_D_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            #     D_fake_logits, tf.zeros_like(D_fake), name='loss_D_fake'))
            # self.loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            #     D_fake_logits, tf.ones_like(D_fake), name='loss_G'))
            # self.loss_D = self.loss_D_real + self.loss_D_fake

            # Loss functions
            self.loss_D_real = binary_cross_entropy(
                D_real, tf.ones_like(D_real), name='loss_D_real')
            self.loss_D_fake = binary_cross_entropy(
                D_fake, tf.zeros_like(D_fake), name='loss_D_fake')
            self.loss_D = tf.reduce_mean((self.loss_D_real + self.loss_D_fake) / 2)
            self.loss_G = tf.reduce_mean(binary_cross_entropy(
                D_fake, tf.ones_like(D_fake), name='loss_G'))



            # Summaries
            # sum_loss_D_real = tf.summary.scalar("loss_D_real", self.loss_D_real)
            # sum_loss_D_fake = tf.summary.scalar("loss_D_fake", self.loss_D_fake)
            # sum_loss_D = tf.summary.scalar("loss_D", self.loss_D)
            # sum_loss_G = tf.summary.scalar("loss_G", self.loss_G)

            sum_loss_D_real = tf.summary.histogram("loss_D_real", self.loss_D_real)
            sum_loss_D_fake = tf.summary.histogram("loss_D_fake", self.loss_D_fake)
            sum_loss_D = tf.summary.scalar("loss_D", self.loss_D)
            sum_loss_G = tf.summary.scalar("loss_G", self.loss_G)

            self.sum_D_real = histogram_summary("D_real", D_real)
            self.sum_D_fake = histogram_summary("D_fake", D_fake)

        self.vars_d = [v for v in tf.trainable_variables()
                  if v.name.startswith('discriminator')]

        self.vars_g = [v for v in tf.trainable_variables()
                  if v.name.startswith('generator')]

        self.vars = {
            'loss_D': self.loss_D,
            'loss_G': self.loss_G,
            'x': x,
            'G': self.G,
            'z': z,
            'train': phase_train,
            'sums': {
                'G': self.sum_G,
                'D_real': self.sum_D_real,
                'D_fake': self.sum_D_fake,
                'loss_G': sum_loss_G,
                'loss_D': sum_loss_D,
                'loss_D_real': sum_loss_D_real,
                'loss_D_fake': sum_loss_D_fake,
                'z': sum_z
                # 'x': sum_x
            },
            'vars_d': self.vars_d,
            'vars_g': self.vars_g
        }

        self.saver = tf.train.Saver()
        # We create a session to use the graph
        self.sess = tf.Session()

    def train(self):
        # get list of files names
        files = []
        img_types = ['.jpg', '.jpeg', '.png']
        for root, dirnames, filenames in os.walk(self.files_path):
            for filename in filenames:
                if any([filename.endswith(type_str) for type_str in img_types]):
                    files.append(os.path.join(root, filename))

        batch = create_input_pipeline(
            files=files,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            crop_shape=self.output_shape,
            crop_factor=self.crop_factor,
            shape=self.input_shape)

        gan=self.vars

        init_lr_g = self.learning_rate
        init_lr_d = self.learning_rate

        lr_g = tf.placeholder(tf.float32, shape=[], name='learning_rate_g')
        lr_d = tf.placeholder(tf.float32, shape=[], name='learning_rate_d')

        try:
            from tf.contrib.layers import apply_regularization
            d_reg = apply_regularization(
                tf.contrib.layers.l2_regularizer(1e-6), self.vars_d)
            g_reg = apply_regularization(
                tf.contrib.layers.l2_regularizer(1e-6), self.vars_g)
        except:
            d_reg, g_reg = 0, 0

        opt_d = tf.train.AdamOptimizer(self.learning_rate, name='Adam_d', beta1=self.beta1).minimize(
            gan['loss_D'] + d_reg, var_list=self.vars_d)
        opt_g = tf.train.AdamOptimizer(self.learning_rate, name='Adam_g', beta1=self.beta1).minimize(
            gan['loss_G'] + g_reg, var_list=self.vars_g)

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

        epoch_num =  [v for v in tf.local_variables() if v.name == 'input_producer/limit_epochs/epochs:0'][0]

        sums = gan['sums']
        # self.sum_G = tf.summary.merge([
        #     sums['z'],
        #     sums['D_fake'],
        #     sums['G'],
        #     sums['loss_D_fake'],
        #     sums['loss_G']])
        # self.sum_D_real = tf.summary.merge([
        #     sums['z'],
        #     sums['D_real'],
        #     sums['loss_D_real'],
        #     sums['loss_D']])
        G_sum_op = tf.summary.merge([
            sums['G'], sums['loss_G'], sums['z'],
            sums['loss_D_fake'], sums['D_fake']])
        D_sum_op = tf.summary.merge([
            sums['loss_D'], sums['loss_D_real'], sums['loss_D_fake'],
            sums['z'], sums['D_real'], sums['D_fake']])
        self.writer = tf.summary.FileWriter(self.tensorboard_dir, self.sess.graph)

        zs = np.random.uniform(
            -1.0, 1.0, [self.sample_size, self.z_dim])
        # zs = make_latent_manifold(zs, n_samples)

        start_time = time.time()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        # g = tf.get_default_graph()
        # print [(op.name) for op in g.get_operations()]

        if load(self):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        step_i, t_i = 0, 0
        loss_d = 1
        loss_g = 1
        n_loss_d, total_loss_d = 1, 1
        n_loss_g, total_loss_g = 1, 1
        try:
            while not coord.should_stop():
                batch_xs = self.sess.run(batch)
                batch_zs = np.random.uniform(
                    -1.0, 1.0, [self.batch_size, self.z_dim]).astype(np.float32)

                this_lr_g = min(1e-2, max(1e-6, init_lr_g * (loss_g / loss_d) ** 2))
                this_lr_d = min(1e-2, max(1e-6, init_lr_d * (loss_d / loss_g) ** 2))

                if step_i % 3 == 1:
                    loss_d, _, sum_d = self.sess.run([gan['loss_D'], opt_d, D_sum_op],
                                                feed_dict={gan['x']: batch_xs,
                                                           gan['z']: batch_zs,
                                                           gan['train']: True,
                                                           lr_d: this_lr_d})
                    total_loss_d += loss_d
                    n_loss_d += 1
                    self.writer.add_summary(sum_d, step_i)

                else:
                    loss_g, _, sum_g = self.sess.run([gan['loss_G'], opt_g, G_sum_op],
                                                feed_dict={gan['z']: batch_zs,
                                                           gan['train']: True,
                                                           lr_g: this_lr_g})
                    total_loss_g += loss_g
                    n_loss_g += 1
                    self.writer.add_summary(sum_g, step_i)

                step_i += 1
                curr_epoch = epoch_num.eval(session=self.sess)
                print('Epoch: [%2d] %04d d  = lr: %0.08f, loss: %08.06f, \t' %
                      (curr_epoch, step_i, this_lr_d, loss_d) +
                      'g* = lr: %0.08f, loss: %08.06f' % (this_lr_g, loss_g))

                # # Update D network
                # _, summary_str = self.sess.run([opt_d, self.sum_D_real],
                #                                 feed_dict={gan['x']: batch_xs,
                #                                            gan['z']: batch_z,
                #                                            gan['train']: True})
                # self.writer.add_summary(summary_str, step_i)

                # # Update G network
                # _, summary_str = self.sess.run([opt_g, self.sum_G],
                #     feed_dict={gan['z']: batch_z, gan['train']: True})
                # self.writer.add_summary(summary_str, step_i)
                #
                # # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                # _, summary_str = self.sess.run([opt_g, self.sum_G],
                #     feed_dict={gan['z']: batch_z, gan['train']: True})
                # self.writer.add_summary(summary_str, step_i)
                #
                # errD_fake = self.loss_D_fake.eval({gan['z']: batch_z, gan['train']: True}, session=self.sess)
                # errD_real = self.loss_D_real.eval({gan['x']: batch_xs, gan['train']: True}, session=self.sess)
                # errG = self.loss_G.eval({gan['z']: batch_z, gan['train']: True}, session=self.sess)

                if step_i % self.sample_step == 0:
                    samples = self.sess.run(gan['G'], feed_dict={
                        gan['z']: zs,
                        gan['train']: False})
                    montage(np.clip((samples + 1) * 127.5, 0, 255).astype(np.uint8),
                            os.path.join(self.model_dir, 'sample_%08d.png'%t_i))

                    print("save sample %d"%t_i)
                    # print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                    t_i += 1

                if step_i % self.save_step == 0:
                    # print('generator loss:', total_loss_g / n_loss_g)
                    # print('discriminator loss:', total_loss_d / n_loss_d)
                    # Save the variables to disk.
                    save(self, step_i)
                    print ("model saved to disk")

        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # One of the threads has issued an exception.  So let's tell all the
            # threads to shutdown.
            coord.request_stop()

        # Wait until all threads have finished.
        coord.join(threads)

        # Clean up the session.
        self.sess.close()


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
                h, W = conv2d(
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
            norm = batch_norm(
                x=h,
                phase_train=phase_train,
                name='bn',
                reuse=reuse)
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
                features=batch_norm(
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
                norm = batch_norm(
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


def generator(z, phase_train, output_shape=[64,64,3], convolutional=True,
              n_features=32, reuse=None, activation=tf.nn.relu6, output_activation=tf.nn.tanh):
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
    with tf.variable_scope('generator', reuse=reuse):
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


def discriminator(x, phase_train, convolutional=True, n_features=32, reuse=False,
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


def parse_arguments(parameters):
    parser = argparse.ArgumentParser(description='Calling GAN model')
    for p in parameters:
        name = '--%s' % p.name
        nargs = '?'
        if type == list:
            nargs = '+'
            if not p.size_change:
                nargs = str(len(p.value))
        parser.add_argument(name, type=p.type, nargs=nargs, deafult=p.value,
                            help=p.description)
    return parser


def save(model, step):
    model_name = model.run_name
    checkpoint_dir = model.checkpoint_dir

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    model.saver.save(model.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)


def load(model):
    print(" [*] Reading checkpoints...")
    checkpoint_dir = model.checkpoint_dir
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        model.saver.restore(model.sess, os.path.join(checkpoint_dir, ckpt_name))
        print(" [*] Success to read {}".format(ckpt_name))
        return True
    else:
        print(" [*] Failed to find a checkpoint")
        return False