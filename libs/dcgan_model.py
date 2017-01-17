from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from libs.dcgan_ops import *
from libs.dcgan_utils import *
from libs.parameter import Parameter


class DCGAN(object):
    parameters = [
        Parameter("files_path", "", str, description="path to files", is_path=True),
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
        Parameter("gf_dim", 64, int, 1, 512, 1, "Dimension of gen filters in first conv layer. [64]"),
        Parameter("df_dim", 64, int, 1, 512, 1, "Dimension of disc filters in first conv layer. [64]"),
        Parameter("gfc_dim", 1024, int, 1, 2048, 1, "Dimension of gen filters in first conv layer. [1024]"),
        Parameter("dfc_dim", 1024, int, 1, 2048, 1, "Dimension of gen filters in first conv layer. [1024]"),
        Parameter("sample_step", 5, int, 1, 1000, 1, "save sample images every X steps"),
        Parameter("save_step", 50, int, 1, 1000, 1, "save model file every X steps"),
        #Parameter("train_size", float("inf"), float, 1, float("inf"), 1, "limit the files to use for training?")
        Parameter("save_path", "./tmp", str, description="path to sve model files", is_path=True),
        Parameter("run_name", "gan_%s" % time.strftime("%Y%m%d-%H%M%S"), str,
                  description="name of this run for creating relevant folders")
    ]

    def __init__(self, sess, params):
        self.sess = sess

        self.y_dim = None

        self.gf_dim = params['gf_dim']
        self.df_dim = params['df_dim']

        self.gfc_dim = params['gfc_dim']
        self.dfc_dim = params['dfc_dim']

        self.train_size = float("inf")

        self.image_size = params['input_h']
        self.output_size = params['output_h']

        self.files_path = params['files_path']
        self.learning_rate = float(params['learning_rate'])
        self.beta1 = float(params['beta1'])
        self.batch_size = int(params['batch_size'])
        self.n_epochs = int(params['n_epochs'])
        self.sample_num = int(params['n_examples'])
        self.z_dim = int(params['z_dim'])
        self.is_grayscale = params['is_grayscale']
        self.c_dim = 1 if self.is_grayscale else 3
        self.input_shape = [int(params['input_h']), int(params['input_w']), self.c_dim]
        self.output_shape = [int(params['output_h']), int(params['output_w']), self.c_dim]

        self.input_height = int(params['input_h'])
        self.input_width = int(params['input_w'])
        self.output_height = int(params['output_h'])
        self.output_width = int(params['output_w'])

        self.crop_factor = float(params['crop_factor'])
        self.is_crop = self.crop_factor != 1
        self.save_path = params['save_path']
        self.run_name = params['run_name']
        self.sample_step = int(params['sample_step'])
        self.save_step = int(params['save_step'])
        self.dataset = self.run_name

        self.tensorboard_dir = params['tensorboard_dir'] if 'tensorboard_dir' in params else os.path.join(self.save_path, self.run_name, 'logs')
        self.sample_dir = os.path.join(self.save_path, self.run_name, 'model')
        self.checkpoint_dir = os.path.join(self.save_path, self.run_name, 'ckpt')
        paths = [self.tensorboard_dir, self.sample_dir, self.checkpoint_dir]
        print paths

        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        if not self.y_dim:
            self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        if not self.y_dim:
            self.g_bn3 = batch_norm(name='g_bn3')
        self.build_model()

    def build_model(self):
        if self.y_dim:
            self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

        image_dims = [self.output_height, self.output_width, self.c_dim]

        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + image_dims, name='real_images')
        self.sample_inputs = tf.placeholder(
            tf.float32, [self.sample_num] + image_dims, name='sample_inputs')

        inputs = self.inputs
        sample_inputs = self.sample_inputs

        self.z = tf.placeholder(
            tf.float32, [None, self.z_dim], name='z')
        self.z_sum = histogram_summary("z", self.z)

        if self.y_dim:
            self.G = self.generator(self.z, self.y)
            self.D, self.D_logits = \
                self.discriminator(inputs, self.y, reuse=False)

            self.sampler = self.sampler(self.z, self.y)
            self.D_, self.D_logits_ = \
                self.discriminator(self.G, self.y, reuse=True)
        else:
            self.G = self.generator(self.z)
            self.D, self.D_logits = self.discriminator(inputs)

            self.sampler = self.sampler(self.z)
            self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        self.d_sum = histogram_summary("d", self.D)
        self.d__sum = histogram_summary("d_", self.D_)
        self.G_sum = image_summary("G", self.G)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                self.D_logits_, tf.ones_like(self.D_)))

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self):
        """Train DCGAN"""
        files = []
        img_types = ['.jpg', '.jpeg', '.png']
        for root, dirnames, filenames in os.walk(self.files_path):
            for filename in filenames:
                if any([filename.endswith(type_str) for type_str in img_types]):
                    files.append(os.path.join(root, filename))
        data = files
        print ("found %d files"%len(data))
        np.random.shuffle(data)
        batch_idxs = min(len(data), self.train_size) // self.batch_size

        d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.g_sum = merge_summary([self.z_sum, self.d__sum,
                                    self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary(
            [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = SummaryWriter(self.tensorboard_dir, self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

        sample_files = data[0:self.sample_num]
        sample = [
            get_image(sample_file,
                      input_height=self.input_height,
                      input_width=self.input_width,
                      resize_height=self.output_height,
                      resize_width=self.output_width,
                      is_crop=self.is_crop,
                      is_grayscale=self.is_grayscale) for sample_file in sample_files]
        if (self.is_grayscale):
            sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            sample_inputs = np.array(sample).astype(np.float32)
        save_images(sample_inputs, [8, 8],
                    '{}/test_samples.png'.format(self.sample_dir))

        counter = 1
        start_time = time.time()

        if self.load():
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(self.n_epochs):
            np.random.shuffle(data)
            for idx in xrange(0, batch_idxs):
                batch_files = data[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch = [
                    get_image(batch_file,
                              input_height=self.input_height,
                              input_width=self.input_width,
                              resize_height=self.output_height,
                              resize_width=self.output_width,
                              is_crop=self.is_crop,
                              is_grayscale=self.is_grayscale) for batch_file in batch_files]
                if (self.is_grayscale):
                    batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]) \
                    .astype(np.float32)

                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={self.inputs: batch_images, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                errD_real = self.d_loss_real.eval({self.inputs: batch_images})
                errG = self.g_loss.eval({self.z: batch_z})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, idx, batch_idxs,
                         time.time() - start_time, errD_fake + errD_real, errG))

                if np.mod(counter, self.sample_step) == 1:
                    try:
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={
                                self.z: sample_z,
                                self.inputs: sample_inputs,
                            },
                        )
                        save_images(samples, [8, 8],
                                    './{}/train_{:02d}_{:04d}.png'.format(self.sample_dir, epoch, idx))
                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))
                    except:
                        print("one pic error!...")

                if np.mod(counter, self.save_step) == 2:
                    self.save (counter)
            self.save(counter)

    def discriminator(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4

    def generator(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_h4, s_h8, s_h16 = \
                int(s_h / 2), int(s_h / 4), int(s_h / 8), int(s_h / 16)
            s_w2, s_w4, s_w8, s_w16 = \
                int(s_w / 2), int(s_w / 4), int(s_w / 8), int(s_w / 16)

            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(
                z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(
                self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(
                h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            h2, self.h2_w, self.h2_b = deconv2d(
                h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(
                h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d(
                h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

            return tf.nn.tanh(h4)

    def sampler(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            s_h, s_w = self.output_height, self.output_width
            s_h2, s_h4, s_h8, s_h16 = \
                int(s_h / 2), int(s_h / 4), int(s_h / 8), int(s_h / 16)
            s_w2, s_w4, s_w8, s_w16 = \
                int(s_w / 2), int(s_w / 4), int(s_w / 8), int(s_w / 16)

            # project `z` and reshape
            h0 = tf.reshape(
                linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin'),
                [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0, train=False))

            h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1, train=False))

            h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2, train=False))

            h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3, train=False))

            h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

            return tf.nn.tanh(h4)

    def save(self, step):
        self.saver.save(self.sess,
                        os.path.join(self.checkpoint_dir, "%s.model"%self.run_name),
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

    def generate(self, num_smaples, save_single=False):
        for i in range(max(int(num_smaples/self.batch_size), 1)):
            batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]) \
                .astype(np.float32)
            if save_single:
                samples = self.sess.run(self.G, feed_dict={self.z: batch_z})
                save_images_single(samples,
                                   os.path.join(self.sample_dir,
                                                'test_%s_%d.png' % (strftime("%Y-%m-%d %H:%M:%S", gmtime()), i)))
            else:
                save_images(samples, [8, 8],
                        os.path.join(self.sample_dir,'test_%s_%d.png' % (strftime("%Y-%m-%d %H:%M:%S", gmtime()), i)))
