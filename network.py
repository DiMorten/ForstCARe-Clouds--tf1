import tensorflow as tf
import tensorflow.compat.v1 as tf1
slim = tf.contrib.slim
from resnet import resnet_v2, resnet_utils

import sys
from ops import *
import numpy as np



# # # # # # # # # # # # # DEEPLAB # # # # # # # # # # # # #
@slim.add_arg_scope
def atrous_spatial_pyramid_pooling(net, scope, rate=None, depth=256, reuse=None):
    """
    ASPP consists of (a) one 1×1 convolution and three 3×3 convolutions with rates = (6, 12, 18) when output stride = 16
    (all with 256 filters and batch normalization), and (b) the image-level features as described in https://arxiv.org/abs/1706.05587
    :param net: tensor of shape [BATCH_SIZE, WIDTH, HEIGHT, DEPTH]
    :param scope: scope name of the aspp layer
    :return: network layer with aspp applyed to it.
    """
    
    with tf1.variable_scope(scope, reuse=reuse):
        feature_map_size = tf.shape(net)

        # apply global average pooling
        image_level_features = tf.reduce_mean(net, [1, 2], name='image_level_global_pool', keepdims=True)

        image_level_features = slim.conv2d(image_level_features, depth, [1, 1], scope="image_level_conv_1x1",
                                           activation_fn=None)
        image_level_features = tf1.image.resize_bilinear(image_level_features, (feature_map_size[1], feature_map_size[2]))

        at_pool1x1 = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_0", activation_fn=None)

        at_pool3x3_1 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_1", rate=rate[0], activation_fn=None)

        at_pool3x3_2 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_2", rate=rate[1], activation_fn=None)

        at_pool3x3_3 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_3", rate=rate[2], activation_fn=None)

        net = tf.concat((image_level_features, at_pool1x1, at_pool3x3_1, at_pool3x3_2, at_pool3x3_3), axis=3,
                        name="concat")
        net = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_output", activation_fn=None)

        return net

@slim.add_arg_scope
def Decoder(dec, scope, args, skip_connections, fine_size, reuse=None, rate_dropout=0.5, is_train=True):
    
    with tf1.variable_scope(scope, reuse = reuse):

        s = fine_size
        s2, s4 = s//2, s//4

        depth = dec.get_shape()[3].value
        if 'dropout' in args.sampling_type:
                dec = tf.layers.dropout(dec, rate_dropout, training=is_train)
                # dec = tf.nn.dropout(dec, keep_prob)
        
        if args.image_size_tr == 256 and args.output_stride == 16:
            dec = deconv2d(tf.nn.leaky_relu(dec), 
                                [tf.shape(dec)[0], s4, s4, depth//(2**2)], name='g_d3')
            dec = tf.contrib.layers.batch_norm(dec, scope = 'g_d3_bn', is_training=is_train)
            dec = tf.concat([dec, skip_connections[0]], 3)            # (batch, 64, 64, 256)
        

        dec = deconv2d(tf.nn.leaky_relu(dec), 
                            [tf.shape(dec)[0], s2, s2, depth//(2**3)], name='g_d2')
        dec = tf.contrib.layers.batch_norm(dec, scope = 'g_d2_bn', is_training=is_train)
        dec = tf.concat([dec, skip_connections[1]], 3)            # (batch, 128, 128, 128)
        
        dec = deconv2d(tf.nn.leaky_relu(dec), 
                            [tf.shape(dec)[0], s, s, args.output_nc], name='g_d1')
                                                                
                                                                #(batch, 256, 256, 7)
    return dec

def deeplab_v3(inputs, args, fine_size, is_training, reuse, rate_dropout = 0.5):

    # inputs has shape - Original: [batch, 768, 768, ...]

    with slim.arg_scope(resnet_utils.resnet_arg_scope(args.l2_regularizer, is_training,
                                                      args.batch_norm_decay,
                                                      args.batch_norm_epsilon)):
        
        resnet = getattr(resnet_v2, args.resnet_model)
        _, end_points = resnet(inputs,
                               args.number_of_classes,
                               is_training=is_training,
                               global_pool=False,
                               spatial_squeeze=False,
                               output_stride=args.output_stride,
                               reuse=reuse)

        with tf1.variable_scope("DeepLab_v3", reuse=reuse):
            
            # get outputs for skip connections            
            skip_connections = [end_points['generator/' + args.resnet_model + '/block2/unit_3/bottleneck_v2'],
                                end_points['generator/' + args.resnet_model + '/block1/unit_2/bottleneck_v2']]
            # rates
            rate = [6, 12, 18]
            # get block 4 feature outputs
            net = end_points['generator/' + args.resnet_model + '/block4']

            if 'dropout' in args.sampling_type:
                net = tf.layers.dropout(net, rate_dropout, training=is_training)
                # net = tf.nn.dropout(net, keep_prob)

            net = atrous_spatial_pyramid_pooling(net, "ASPP_layer", rate = rate, depth=512, reuse=reuse)
            
            net = Decoder(net, "Decoder", args, skip_connections, fine_size, reuse=reuse, rate_dropout=rate_dropout, is_train=is_training)
            
            
            # for key, value in end_points.items():
            #     print (key)
            #     print (end_points[key].get_shape())
            # sys.exit()

            # net = slim.conv2d(net, args.number_of_classes, [1, 1], activation_fn=None,
            #                   normalizer_fn=None, scope='logits')
            # net = slim.conv2d(net, args.number_of_classes, [1, 1], activation_fn=None,
            #                   scope='logits')
            # # resize the output logits to match the labels dimensions
            # size = tf.shape(inputs)[1:3]
            # net = tf1.image.resize_nearest_neighbor(net, size)
            # net = tf1.image.resize_bicubic(net, size)
            # net = tf1.image.resize_bilinear(net, size)

            return net

def deeplab(self, img_A, img_B=None, reuse = False, is_train=True):

    with tf1.variable_scope("generator"):
        # inputs = slim.conv2d(img_A, img_A.get_shape()[-1]*8, [5, 5], stride = 1, \
        #                     activation_fn = tf.nn.leaky_relu, scope = "g_er_conv_2", reuse = reuse)
        inputs = img_A
        if img_B is not None:
            inputs = tf.concat([inputs, img_B], 3)
        net = deeplab_v3(inputs, self.args, tf.shape(img_A)[1], is_training = is_train, reuse = reuse, rate_dropout = self.rate_dropout)

        if self.norm_type == 'std' or self.norm_type == 'wise_frame_mean':
            return net
        else:
            return tf.nn.tanh(net)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # U-Net # # # # # # # # # # # # #
def unet(self, img_A, img_B=None, reuse = False, is_train=True):

    with tf1.variable_scope("generator") as scope:

        if reuse:
            scope.reuse_variables()
        
        # s = self.image_size_tr
        # s2, s4, s8, s16, s32, s64, s128 = int(s/2), int(s/4), int(s/8), int(s/16), int(s/32), int(s/64), int(s/128)

        s = tf.shape(img_A)[1]
        s2, s4, s8, s16, s32, s64, s128 = s//2, s//4, s//8, s//16, s//32, s//64, s//128
        
        # image = tf.tanh(conv2d(img_A, img_A.get_shape()[-1], d_h=1, d_w=1, name='g_er_conv'))
        image = img_A
        if img_B is not None:
            image = tf.concat([image, img_B], 3)
        
        # image is (256 x 256 x input_c_dim)
        e1 = conv2d(image, self.gf_dim, name='g_e1_conv') # 64x2x5x5+64 = 3,264
        # e1 is (128 x 128 x self.gf_dim)
        e2 = self.g_bn_e2(conv2d(lrelu(e1), self.gf_dim*2, name='g_e2_conv'), is_training=is_train) # (2x64)x64x5x5 + (2x64) = 204,928
        # e2 is (64 x 64 x self.gf_dim*2)
        e3 = self.g_bn_e3(conv2d(lrelu(e2), self.gf_dim*4, name='g_e3_conv'), is_training=is_train) # (4x64)x(2x64)x5x5 + (4x64) = 819,456
        # e3 is (32 x 32 x self.gf_dim*4)
        e4 = self.g_bn_e4(conv2d(lrelu(e3), self.gf_dim*8, name='g_e4_conv'), is_training=is_train) # (8x64)x(4x64)x5x5 + (8x64) = 3,277,312
        # e4 is (16 x 16 x self.gf_dim*8)
        e5 = self.g_bn_e5(conv2d(lrelu(e4), self.gf_dim*8, name='g_e5_conv'), is_training=is_train) # (8x64)x(8x64)x5x5 + (8x64) = 6,554,112
        # e5 is (8 x 8 x self.gf_dim*8)
        e6 = self.g_bn_e6(conv2d(lrelu(e5), self.gf_dim*8, name='g_e6_conv'), is_training=is_train) # (8x64)x(8x64)x5x5 + (8x64) = 6,554,112
        # e6 is (4 x 4 x self.gf_dim*8)
        e7 = self.g_bn_e7(conv2d(lrelu(e6), self.gf_dim*8, name='g_e7_conv'), is_training=is_train) # (8x64)x(8x64)x5x5 + (8x64) = 6,554,112
        # e7 is (2 x 2 x self.gf_dim*8)
        enc = e7
        
        if self.image_size_tr >= 256:
            enc = self.g_bn_e8(conv2d(lrelu(e7), self.gf_dim*8, name='g_e8_conv'), is_training=is_train) # (8x64)x(8x64)x5x5 + (8x64) = 6,554,112
            # e8 is (1 x 1 x self.gf_dim*8)

            enc, self.d1_w, self.d1_b = deconv2d(tf.nn.relu(enc),
                [tf.shape(enc)[0], s128, s128, self.gf_dim*8], name='g_d1', with_w=True) # (8x64)x(8x64)x5x5 + (8x64) = 6,554,112
            if 'dropout' in self.sampling_type:
                enc = tf.layers.dropout(self.g_bn_d1(enc, is_training=is_train), self.rate_dropout, training=is_train)
                # enc = tf.nn.dropout(self.g_bn_d1(enc, is_training=is_train), self.keep_prob)
            else:
                enc = self.g_bn_d1(enc, is_training=is_train)
            enc = tf.concat([enc, e7], 3)
            # d1 is (2 x 2 x self.gf_dim*8*2)

        self.d2, self.d2_w, self.d2_b = deconv2d(tf.nn.relu(enc),
            [tf.shape(enc)[0], s64, s64, self.gf_dim*8], name='g_d2', with_w=True) # (2*8x64)x(8x64)x5x5 + (2*8x64) = 13,108,224
        if 'dropout' in self.sampling_type:
            d2 = tf.layers.dropout(self.g_bn_d2(self.d2, is_training=is_train), self.rate_dropout, training=is_train)
            # d2 = tf.nn.dropout(self.g_bn_d2(self.d2, is_training=is_train), self.keep_prob)
        else:
            d2 = self.g_bn_d2(self.d2, is_training=is_train)
        d2 = tf.concat([d2, e6], 3)
        # d2 is (4 x 4 x self.gf_dim*8*2)

        self.d3, self.d3_w, self.d3_b = deconv2d(tf.nn.relu(d2),
            [tf.shape(d2)[0], s32, s32, self.gf_dim*8], name='g_d3', with_w=True) # (2*8x64)x(8x64)x5x5 + (2*8x64) = 13,108,224
        if 'dropout' in self.sampling_type:
            d3 = tf.layers.dropout(self.g_bn_d3(self.d3, is_training=is_train), self.rate_dropout, training=is_train)
            # d3 = tf.nn.dropout(self.g_bn_d3(self.d3, is_training=is_train), self.keep_prob)
        else:
            d3 = self.g_bn_d3(self.d3, is_training=is_train)
        d3 = tf.concat([d3, e5], 3)
        # d3 is (8 x 8 x self.gf_dim*8*2)

        self.d4, self.d4_w, self.d4_b = deconv2d(tf.nn.relu(d3),
            [tf.shape(d3)[0], s16, s16, self.gf_dim*8], name='g_d4', with_w=True) # (2*8x64)x(8x64)x5x5 + (2*8x64) = 13,108,224
        d4 = self.g_bn_d4(self.d4, is_training=is_train)
        d4 = tf.concat([d4, e4], 3)
        # d4 is (16 x 16 x self.gf_dim*8*2)

        self.d5, self.d5_w, self.d5_b = deconv2d(tf.nn.relu(d4),
            [tf.shape(d4)[0], s8, s8, self.gf_dim*4], name='g_d5', with_w=True) # (2*4x64)x(8x64)x5x5 + (2*4x64) = 6,554,112
        d5 = self.g_bn_d5(self.d5, is_training=is_train)
        d5 = tf.concat([d5, e3], 3)
        # d5 is (32 x 32 x self.gf_dim*4*2)

        self.d6, self.d6_w, self.d6_b = deconv2d(tf.nn.relu(d5),
            [tf.shape(d5)[0], s4, s4, self.gf_dim*2], name='g_d6', with_w=True) # (2*2x64)x(4x64)x5x5 + (2*2x64) = 1,638,656
        d6 = self.g_bn_d6(self.d6, is_training=is_train)
        d6 = tf.concat([d6, e2], 3)
        # d6 is (64 x 64 x self.gf_dim*2*2)

        self.d7, self.d7_w, self.d7_b = deconv2d(tf.nn.relu(d6),
            [tf.shape(d6)[0], s2, s2, self.gf_dim], name='g_d7', with_w=True) # (2*1x64)x(2x64)x5x5 + (2*1x64) = 409,728
        d7 = self.g_bn_d7(self.d7, is_training=is_train)
        d7 = tf.concat([d7, e1], 3)
        # d7 is (128 x 128 x self.gf_dim*1*2)

        self.d8, self.d8_w, self.d8_b = deconv2d(tf.nn.relu(d7),
            [tf.shape(d7)[0], s, s, self.output_c_dim], name='g_d8', with_w=True) # (1*1x64)x(1x7)x5x5 + (1*1x7) = 11,207
        # d8 is (256 x 256 x output_c_dim)

        if self.norm_type == 'std' or self.norm_type == 'wise_frame_mean':
            return self.d8
        else:
            return tf.nn.tanh(self.d8)
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def atrous_discriminator(self, img_A, img_B, y=None, reuse=False, is_train=True):

    def atrous_convs(net, scope, rate=None, depth=256, reuse=None):
        """
        ASPP layer 1×1 convolution and three 3×3 atrous convolutions
        """
        
        with tf1.variable_scope(scope, reuse=reuse):
            at_pool1x1 = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_0", activation_fn=None)

            at_pool3x3_1 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_1", rate=rate[0], activation_fn=None)

            at_pool3x3_2 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_2", rate=rate[1], activation_fn=None)

            at_pool3x3_3 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_3", rate=rate[2], activation_fn=None)

            net = tf.concat((at_pool1x1, at_pool3x3_1, at_pool3x3_2, at_pool3x3_3), axis=3,
                            name="concat")
            net = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_output", activation_fn=None)

            return net
    with tf1.variable_scope("discriminator") as scope:

        # image is 256 x 256 x (input_c_dim + output_c_dim)
        if reuse:
            scope.reuse_variables()
        
        # img_A = tf.tanh(conv2d(img_A, img_A.get_shape()[-1], d_h=3, d_w=3, name='d_hr_conv'))
        image = tf.concat([img_A, img_B], 3)

        rate = [2, 3, 4]
        atrous0 = atrous_convs(image, "d_atrous_0", rate = rate, depth=self.df_dim, reuse=reuse)
        h0 = lrelu(conv2d(atrous0, self.df_dim, name='d_h0_conv'))
        # h0 is (64 x 64 x self.df_dim)

        atrous1 = atrous_convs(h0, "d_atrous_1", rate = rate, depth=self.df_dim, reuse=reuse)
        h1 = lrelu(self.d_bn1(conv2d(atrous1, self.df_dim*2, name='d_h1_conv'), is_training=is_train))
        # h1 is (32 x 32 x self.df_dim*2)

        atrous2 = atrous_convs(h1, "d_atrous_2", rate = rate, depth=self.df_dim*2, reuse=reuse)
        h2 = lrelu(self.d_bn2(conv2d(atrous2, self.df_dim*4, name='d_h2_conv'), is_training=is_train))
        # h2 is (16 x 16 x self.df_dim*4)

        # h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv'), is_training=is_train))
        # # h3 is (8 x 8 x self.df_dim*8)
        
        # shape = h3.get_shape()
        # h4 = linear(tf.reshape(h3, [-1, shape[1]*shape[2]*shape[3]]), 1, 'd_h3_lin')

        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv'), is_training=is_train))
        h4 = conv2d(h3, 1, d_h=1, d_w=1, name='d_h4_conv')

        return tf.nn.sigmoid(h4), h4

def pix2pix_discriminator(self, img_A, img_B, y=None, reuse=False, is_train=True):

    with tf1.variable_scope("discriminator") as scope:

        # image is 256 x 256 x (input_c_dim + output_c_dim)
        if reuse:
            scope.reuse_variables()
        
        # img_A = tf.tanh(conv2d(img_A, img_A.get_shape()[-1], d_h=3, d_w=3, name='d_hr_conv'))
        image = tf.concat([img_A, img_B], 3)

        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
        # h0 is (64 x 64 x self.df_dim)

        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv'), is_training=is_train))
        # h1 is (32 x 32 x self.df_dim*2)

        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv'), is_training=is_train))
        # h2 is (16 x 16 x self.df_dim*4)

        # h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv'), is_training=is_train))
        # # h3 is (8 x 8 x self.df_dim*8)
        
        # shape = h3.get_shape()
        # h4 = linear(tf.reshape(h3, [-1, shape[1]*shape[2]*shape[3]]), 1, 'd_h3_lin')

        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv'), is_training=is_train))
        h4 = conv2d(h3, 1, d_h=1, d_w=1, name='d_h4_conv')

        return tf.nn.sigmoid(h4), h4




# -------------------------------------
# from spectral_normalization import SpectralNormalization
from SpectralNormalizationKeras import DenseSN, ConvSN2D
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, Activation
from tensorflow.keras.initializers import RandomNormal

def encoder_block(input_data, n_filters, k_size=3, strides=2, activation='None', padding='same', 
                  batchnorm=False, spectralnorm=False, name='None', is_training=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # conv2d = Conv2D(n_filters, k_size, strides=strides, padding=padding, kernel_initializer=init, name=name+'_conv2D')
    # if spectralnorm:
    #     conv2d = SpectralNormalization(conv2d)
    # x = conv2d(input_data)
    if spectralnorm:
        x = ConvSN2D(n_filters, k_size, strides=strides, padding=padding, kernel_initializer=init, name=name+'_convSN2D')(input_data)
    else:
        x = Conv2D(n_filters, k_size, strides=strides, padding=padding, kernel_initializer=init, name=name+'_conv2D')(input_data)

    if batchnorm:
        x = BatchNormalization(momentum=0.8, name=name+'_bn')(x, training=is_training)
    if activation is 'LReLU':
        x = LeakyReLU(alpha=0.2, name=name+'_act_LReLU')(x)        
    elif activation is 'ReLU':
        x = Activation('relu', name=name+'_act_relu')(x)
    return x

def atrous_discriminator_spectralNorm(self, img_A, img_B, y=None, reuse=False, is_train=True):

    init = RandomNormal(stddev=0.02)
    def atrous_convs(net, scope, rate=None, depth=256, reuse=None):
        """
        ASPP layer 1×1 convolution and three 3×3 atrous convolutions
        """
        
        with tf1.variable_scope(scope, reuse=reuse):
            at_pool1x1 = Conv2D(depth, 1, padding='same', kernel_initializer=init, name='conv_1x1_0')(net)
            # at_pool1x1 = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_0", activation_fn=None)

            at_pool3x3_1 = Conv2D(depth, 3, dilation_rate=rate[0], padding='same', kernel_initializer=init, name='conv_3x3_1')(net)
            # at_pool3x3_1 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_1", rate=rate[0], activation_fn=None)

            at_pool3x3_2 = Conv2D(depth, 3, dilation_rate=rate[1], padding='same', kernel_initializer=init, name='conv_3x3_2')(net)
            # at_pool3x3_2 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_2", rate=rate[1], activation_fn=None)

            at_pool3x3_3 = Conv2D(depth, 3, dilation_rate=rate[2], padding='same', kernel_initializer=init, name='conv_3x3_3')(net)
            # at_pool3x3_3 = slim.conv2d(net, depth, [3, 3], scope="conv_3x3_3", rate=rate[2], activation_fn=None)

            net = tf.concat((at_pool1x1, at_pool3x3_1, at_pool3x3_2, at_pool3x3_3), axis=3, name="concat")
            net = Conv2D(depth, 1, padding='same', kernel_initializer=init, name='conv_1x1_output')(net)
            # net = slim.conv2d(net, depth, [1, 1], scope="conv_1x1_output", activation_fn=None)

            return net

    with tf1.variable_scope("discriminator") as scope:

        # image is 256 x 256 x (input_c_dim + output_c_dim)
        if reuse:
            scope.reuse_variables()
        
        # img_A = tf.tanh(conv2d(img_A, img_A.get_shape()[-1], d_h=3, d_w=3, name='d_hr_conv'))
        image = tf.concat([img_A, img_B], 3)

        rate = [2, 3, 4]
        atrous0 = atrous_convs(image, "d_atrous_0", rate = rate, depth=self.df_dim, reuse=reuse)
        h0 = encoder_block(atrous0, self.df_dim, k_size=4, activation='LReLU', spectralnorm = True, name='d_h0_conv', is_training=is_train)
        # h0 is (64 x 64 x self.df_dim)

        atrous1 = atrous_convs(h0, "d_atrous_1", rate = rate, depth=self.df_dim, reuse=reuse)
        h1 = encoder_block(atrous1, self.df_dim*2, k_size=4, activation='LReLU', spectralnorm = True, name='d_h1_conv', is_training=is_train)
        # h1 is (32 x 32 x self.df_dim*2)

        atrous2 = atrous_convs(h1, "d_atrous_2", rate = rate, depth=self.df_dim*2, reuse=reuse)
        h2 = encoder_block(atrous2, self.df_dim*4, k_size=4, activation='LReLU', spectralnorm = True, name='d_h2_conv', is_training=is_train)
        # h2 is (16 x 16 x self.df_dim*4)

        h3 = encoder_block(h2, self.df_dim*8, k_size=4, activation='LReLU', spectralnorm = True, name='d_h3_conv', is_training=is_train)

        h4 = encoder_block(h3, 1, k_size=4, strides=1, name='d_h4_conv')

        return tf.nn.sigmoid(h4), h4


def pix2pix_discriminator_spectralNorm(self, img_A, img_B, y=None, reuse=False, is_train=True):

    with tf1.variable_scope("discriminator") as scope:

        # image is 256 x 256 x (input_c_dim + output_c_dim)
        if reuse:
            scope.reuse_variables()
        
        image = tf.concat([img_A, img_B], 3)

        h0 = encoder_block(image, self.df_dim, k_size=4, activation='LReLU', spectralnorm = True, name='d_h0_conv', is_training=is_train)

        h1 = encoder_block(h0, self.df_dim*2, k_size=4, activation='LReLU', spectralnorm = True, name='d_h1_conv', is_training=is_train)

        h2 = encoder_block(h1, self.df_dim*4, k_size=4, activation='LReLU', spectralnorm = True, name='d_h2_conv', is_training=is_train)

        h3 = encoder_block(h2, self.df_dim*8, k_size=4, activation='LReLU', spectralnorm = True, name='d_h3_conv', is_training=is_train)

        h4 = encoder_block(h3, 1, k_size=4, strides=1, name='d_h4_conv')

        return tf.nn.sigmoid(h4), h4
