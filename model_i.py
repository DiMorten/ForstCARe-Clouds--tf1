from __future__ import division
import os
import time
import glob
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import numpy as np
from six.moves import xrange
from sklearn import preprocessing as pre
import joblib
import scipy.io as io
import matplotlib.pyplot as plt
from skimage.util.shape import view_as_windows
import scipy.io as sio
from tqdm import trange
import json

from ops import *
from utils import *

#####___No@___#####
import network
import sys
slim = tf.contrib.slim
from sen12ms_cr_dataLoader import *

import multiprocessing
from functools import partial
n_cores = multiprocessing.cpu_count()

#####_________#####


class cGAN(object):

    def __init__(self, sess, args, image_size_tr=256, image_size = 256, load_size=286,
                 batch_size=1, sample_size=1, output_size=256,
                 gf_dim=64, df_dim=64, L1_lambda=100,
                 input_c_dim=11, output_c_dim=7, dataset_name='facades',
                 checkpoint_dir=None, sample_dir=None):

        """
        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [256]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            input_c_dim: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
            output_c_dim: (optional) Dimension of output image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_grayscale = (input_c_dim == 1)
        self.batch_size = batch_size
        self.image_size_tr = image_size_tr
        self.image_size = image_size
        self.sample_size = sample_size
        # self.load_size = load_size
        # self.fine_size = image_size

        self.data_path = args.datasets_dir + args.dataset_name


        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim

        self.L1_lambda = L1_lambda
        
        self.args = args
        self.sampling_type = args.sampling_type
        self.norm_type = args.norm_type
        self.rate_dropout = 0.5

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bnr = batch_norm(name='d_bnr')
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.d_bn4 = batch_norm(name='d_bn4')
        self.d_bn5 = batch_norm(name='d_bn5')

        self.g_bn_er = batch_norm(name='g_bn_er')
        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')
        self.g_bn_e5 = batch_norm(name='g_bn_e5')
        self.g_bn_e6 = batch_norm(name='g_bn_e6')
        self.g_bn_e7 = batch_norm(name='g_bn_e7')
        self.g_bn_e8 = batch_norm(name='g_bn_e8')

        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')
        self.g_bn_d4 = batch_norm(name='g_bn_d4')
        self.g_bn_d5 = batch_norm(name='g_bn_d5')
        self.g_bn_d6 = batch_norm(name='g_bn_d6')
        self.g_bn_d7 = batch_norm(name='g_bn_d7')

        self.checkpoint_dir = checkpoint_dir
        self.visible_bands = [2, 1, 0]
        self.build_model()
        # self.norm_routine = Clip_Norm_sen12mscr(feature_range=[-1, 1])

        # Load dataset
        # if args.dataset_name == 'SEN2MS-CR':
        #     self.train_patches, self.val_patches, \
        #         self.test_patches, self.data_files = Load_SEN2MSCR(self.data_path,
        #                                                             train_ROIs = [[Seasons.SUMMER, 86], [Seasons.SUMMER, 120],
        #                                                                           [Seasons.FALL, 39]  , [Seasons.SPRING, 17]],
        #                                                             val_ROIs   = [[Seasons.SPRING, 40]],
        #                                                             test_ROIs  = [[Seasons.FALL, 42], [Seasons.WINTER, 49]],
        #                                                             custom_sets = True)

        if args.dataset_name == 'Para_10m':
            self.lims = np.array([0, 17730, 0, 9200])
            self.sar_path = args.datasets_dir + args.dataset_name + '/Sentinel1_'
            self.opt_path = args.datasets_dir + args.dataset_name + '/Sentinel2_'
            self.opt_cloudy_path = args.datasets_dir + args.dataset_name + '/Sentinel2_Clouds_'
            self.labels_path = args.datasets_dir + args.dataset_name + '/Reference'

            self.sar_name_t0 = ['2018/COPERNICUS_S1_20180719_20180726_VV', 
                                '2018/COPERNICUS_S1_20180719_20180726_VH']
            self.opt_name_t0 = ['2018/COPERNICUS_S2_20180721_20180726_B1_B2_B3',
                                '2018/COPERNICUS_S2_20180721_20180726_B4_B5_B6',
                                '2018/COPERNICUS_S2_20180721_20180726_B7_B8_B8A',
                                '2018/COPERNICUS_S2_20180721_20180726_B9_B10_B11',
                                '2018/COPERNICUS_S2_20180721_20180726_B12']
            self.opt_cloudy_name_t0 = ['2018/COPERNICUS_S2_20180611_B1_B2_B3',
                                       '2018/COPERNICUS_S2_20180611_B4_B5_B6',
                                       '2018/COPERNICUS_S2_20180611_B7_B8_B8A',
                                       '2018/COPERNICUS_S2_20180611_B9_B10_B11',
                                       '2018/COPERNICUS_S2_20180611_B12']
            self.opt_cloudmask_name_t0 = '2018/cloudmask_s2_2018'
            self.opt_cloudy_cloudmask_name_t0 = '2018/cloudmask_s2_cloudy_2018'

            self.sar_name_t1 = ['2019/COPERNICUS_S1_20190721_20190726_VV', 
                                '2019/COPERNICUS_S1_20190721_20190726_VH']
            self.opt_name_t1 = ['2019/COPERNICUS_S2_20190721_20190726_B1_B2_B3',
                                '2019/COPERNICUS_S2_20190721_20190726_B4_B5_B6',
                                '2019/COPERNICUS_S2_20190721_20190726_B7_B8_B8A',
                                '2019/COPERNICUS_S2_20190721_20190726_B9_B10_B11',
                                '2019/COPERNICUS_S2_20190721_20190726_B12']
            self.opt_cloudy_name_t1 = ['2019/COPERNICUS_S2_20190706_B1_B2_B3',
                                       '2019/COPERNICUS_S2_20190706_B4_B5_B6',
                                       '2019/COPERNICUS_S2_20190706_B7_B8_B8A',
                                       '2019/COPERNICUS_S2_20190706_B9_B10_B11',
                                       '2019/COPERNICUS_S2_20190706_B12']
            self.opt_cloudmask_name_t1 = '2019/cloudmask_s2_2019'
            self.opt_cloudy_cloudmask_name_t1 = '2019/cloudmask_s2_cloudy_2019'

            self.labels_name = '/mask_label_17730x9203'

            self.mask_tr_vl_ts_name = '/tile_mask_0tr_1vl_2ts'

        elif args.dataset_name == 'MG_10m':
            self.lims = np.array([0, 20795-4000, 0+3000, 13420])
            self.sar_path = args.datasets_dir + args.dataset_name + '/S1/'
            self.opt_path = args.datasets_dir + args.dataset_name + '/S2/'
            self.opt_cloudy_path = args.datasets_dir + args.dataset_name + '/S2_cloudy/'
            self.labels_path = args.datasets_dir + args.dataset_name

            self.sar_name_t0 = ['2019/S1_R1_MT_2019_08_02_2019_08_09_VV', 
                                '2019/S1_R1_MT_2019_08_02_2019_08_09_VH']
            self.opt_name_t0 = ['2019/S2_R1_MT_2019_08_02_2019_08_05_B1_B2',
                                '2019/S2_R1_MT_2019_08_02_2019_08_05_B3_B4',
                                '2019/S2_R1_MT_2019_08_02_2019_08_05_B5_B6',
                                '2019/S2_R1_MT_2019_08_02_2019_08_05_B7_B8',
                                '2019/S2_R1_MT_2019_08_02_2019_08_05_B8A_B9',
                                '2019/S2_R1_MT_2019_08_02_2019_08_05_B10_B11',
                                '2019/S2_R1_MT_2019_08_02_2019_08_05_B12']
            self.opt_cloudy_name_t0 = ['2019/S2CL_R1_MT_2019_09_26_2019_09_29_B1_B2',
                                       '2019/S2CL_R1_MT_2019_09_26_2019_09_29_B3_B4',
                                       '2019/S2CL_R1_MT_2019_09_26_2019_09_29_B5_B6',
                                       '2019/S2CL_R1_MT_2019_09_26_2019_09_29_B7_B8',
                                       '2019/S2CL_R1_MT_2019_09_26_2019_09_29_B8A_B9',
                                       '2019/S2CL_R1_MT_2019_09_26_2019_09_29_B10_B11',
                                       '2019/S2CL_R1_MT_2019_09_26_2019_09_29_B12']
            self.opt_cloudmask_name_t0 = '2019/cloudmask_s2_2019_MG'
            self.opt_cloudy_cloudmask_name_t0 = '2019/cloudmask_s2_cloudy_2019_MG'

            self.sar_name_t1 = ['2020/S1_R1_MT_2020_08_03_2020_08_08_VV', 
                                '2020/S1_R1_MT_2020_08_03_2020_08_08_VH']
            self.opt_name_t1 = ['2020/S2_R1_MT_2020_08_03_2020_08_15_B1_B2',
                                '2020/S2_R1_MT_2020_08_03_2020_08_15_B3_B4',
                                '2020/S2_R1_MT_2020_08_03_2020_08_15_B5_B6',
                                '2020/S2_R1_MT_2020_08_03_2020_08_15_B7_B8',
                                '2020/S2_R1_MT_2020_08_03_2020_08_15_B8A_B9',
                                '2020/S2_R1_MT_2020_08_03_2020_08_15_B10_B11',
                                '2020/S2_R1_MT_2020_08_03_2020_08_15_B12']
            self.opt_cloudy_name_t1 = ['2020/S2CL_R1_MT_2020_09_15_2020_09_18_B1_B2',
                                       '2020/S2CL_R1_MT_2020_09_15_2020_09_18_B3_B4',
                                       '2020/S2CL_R1_MT_2020_09_15_2020_09_18_B5_B6',
                                       '2020/S2CL_R1_MT_2020_09_15_2020_09_18_B7_B8',
                                       '2020/S2CL_R1_MT_2020_09_15_2020_09_18_B8A_B9',
                                       '2020/S2CL_R1_MT_2020_09_15_2020_09_18_B10_B11',
                                       '2020/S2CL_R1_MT_2020_09_15_2020_09_18_B12']
            self.opt_cloudmask_name_t1 = '2020/cloudmask_s2_2020_MG'
            self.opt_cloudy_cloudmask_name_t1 = '2020/cloudmask_s2_cloudy_2020_MG'

            self.labels_name = '/ref_2019_2020_20798x13420'

            self.mask_tr_vl_ts_name = '/MT_tr_0_val_1_ts_2_16795x10420_new'
        elif args.dataset_name == 'Santarem':
            self.lims = np.array([0, 9676, 0, 10540])
            self.sar_path = args.datasets_dir + args.dataset_name + '/S1/'
            self.opt_path = args.datasets_dir + args.dataset_name + '/S2/'
            self.opt_cloudy_path = args.datasets_dir + args.dataset_name + '/S2_cloudy/'
            self.labels_path = args.datasets_dir + args.dataset_name

            self.sar_name_t0 = ['2020/S1_NS_2020_08_08_08_13_VV_VH']
            self.opt_name_t0 = ['2020/S2_R5_ST_2020_08_09_B1_B7',
                                '2020/S2_R5_ST_2020_08_09_B8_B12']
            self.opt_cloudy_name_t0 = ['2020/S2_CL_R5_ST_2020_08_24_B1_B7',
                                       '2020/S2_CL_R5_ST_2020_08_24_B8_B12']
            self.opt_cloudmask_name_t0 = '2019/cloudmask_s2_2019_MG'
            self.opt_cloudy_cloudmask_name_t0 = '2019/cloudmask_s2_cloudy_2019_MG'

            self.sar_name_t1 = ['2021/S1_NS_2021_07_22_07_27_VV_VH']
            self.opt_name_t1 = ['2021/S2_R5_ST_2021_07_25_B1_B7',
                                '2021/S2_R5_ST_2021_07_25_B8_B12']
            self.opt_cloudy_name_t1 = ['2021/S2_CL_R5_ST_2021_07_30_B1_B7',
                                       '2021/S2_CL_R5_ST_2021_07_30_B8_B12']
            self.opt_cloudmask_name_t1 = '2020/cloudmask_s2_2020_MG'
            self.opt_cloudy_cloudmask_name_t1 = '2020/cloudmask_s2_cloudy_2020_MG'


            self.labels_name = '/mask_label_17730x9203'

            self.mask_tr_vl_ts_name = '/tr_0_val_1_ts_2_9676x10540'            

    def build_model(self):

        # Picking up the generator and discriminator
        generator = getattr(network, self.args.generator)
        # discriminator = getattr(network, self.args.discriminator + '_discriminator_spectralNorm')
        discriminator = getattr(network, self.args.discriminator + '_discriminator')

        # self.output_size = self.image_size//2
        # ============== PLACEHOLDERS ===============
        self.SAR = tf1.placeholder(tf.float32,
                                    [None, None, None, self.input_c_dim],
                                    name='sar')
        self.OPT = tf1.placeholder(tf.float32,
                                    [None, None, None, self.output_c_dim],
                                    name='opt')
        self.OPT_cloudy = tf1.placeholder(tf.float32,
                                    [None, None, None, self.output_c_dim],
                                    name='opt_cloudy')
        self.learning_rate = tf1.placeholder(tf.float32, [], name="learning_rate")


        # =============== NETWORKS =================
        self.fake_opt_t0 = generator(self, self.SAR, self.OPT_cloudy, reuse=False, is_train=True)
        self.fake_opt_t0_sample = generator(self, self.SAR, self.OPT_cloudy, reuse=True, is_train=True)

        self.OPT_pair = tf.concat([self.OPT_cloudy, self.OPT], 3)
        self.OPT_pair_fake = tf.concat([self.OPT_cloudy, self.fake_opt_t0], 3)

        self.D, self.D_logits = discriminator(self, self.SAR, self.OPT_pair, reuse=False)
        self.D_, self.D_logits_ = discriminator(self, self.SAR, self.OPT_pair_fake, reuse=True)
        
        self.d_sum = tf1.summary.histogram("d", self.D)
        self.d__sum = tf1.summary.histogram("d_", self.D_)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_))) \
                        + self.L1_lambda * tf.reduce_mean(tf.abs(self.OPT - self.fake_opt_t0))

        self.d_loss_real_sum = tf1.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf1.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf1.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf1.summary.scalar("d_loss", self.d_loss)
        
        t_vars = tf1.trainable_variables()
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        
        # =============== OPTIMIZERS =================
        self.g_optim = tf1.train.AdamOptimizer(self.learning_rate, beta1=self.args.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        lr = self.learning_rate/10 if self.args.discriminator == "atrous" else self.learning_rate
        # lr = self.learning_rate
        self.d_optim = tf1.train.AdamOptimizer(lr, beta1=self.args.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)

        # # ========== This updates moving_mean and moving_variance 
        # # ========== in batch normalization layers when training
        # update_ops = tf1.get_collection(tf1.GraphKeys.UPDATE_OPS)
        # self.g_ops = [ops for ops in update_ops if 'generator' in ops.name]
        # self.d_ops = [ops for ops in update_ops if 'discriminator' in ops.name]
        # self.g_optim = tf.group([self.g_optim, self.g_ops])
        # self.d_optim = tf.group([self.d_optim, self.d_ops])

        self.model = "%s_bs%s_%s_ps%s" % \
                      (self.args.discriminator, self.batch_size, self.norm_type, self.image_size_tr)

        self.saver = tf1.train.Saver(max_to_keep=3)

        print('_____Generator_____')
        self.count_params(self.g_vars)
        print('_____Discriminator_____')
        self.count_params(self.d_vars)
        print('_____Full Model_____')
        self.count_params(t_vars)


    def train(self, args):
        """Train cGAN"""

        # Model
        model_dir = os.path.join(self.checkpoint_dir, self.model, args.dataset_name)
        sample_dir = os.path.join(model_dir, 'samples')
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)

        #================== CREATE DATASET ==================
        if args.date == 'both':
            train_patches, val_patches, test_patches, self.data_dic, \
                self.sar_norm, self.opt_norm = create_dataset_both_images(self)
        elif args.date == 'd0':
            self.sar_name = self.sar_name_t0
            self.opt_name = self.opt_name_t0
            self.opt_cloudy_name = self.opt_cloudy_name_t0
            train_patches, val_patches, test_patches, self.data_dic, \
                self.sar_norm, self.opt_norm = create_dataset_coordinates(self, prefix = 0)
        elif args.date == 'd1':
            self.sar_name = self.sar_name_t1
            self.opt_name = self.opt_name_t1
            self.opt_cloudy_name = self.opt_cloudy_name_t1
            train_patches, val_patches, test_patches, self.data_dic, \
                self.sar_norm, self.opt_norm = create_dataset_coordinates(self, prefix = 1)
        
        # print("mask_shape:", Split_Image(self, random_tiles="fixed").shape)
        # print(self.data_dic['sar_t0'].shape, self.data_dic['sar_t0'].min(), self.data_dic['sar_t0'].max())
        # print(self.data_dic['opt_t0'].shape, self.data_dic['opt_t0'].min(), self.data_dic['opt_t0'].max())
        # print(self.data_dic['opt_cloudy_t0'].shape, self.data_dic['opt_cloudy_t0'].min(), self.data_dic['opt_cloudy_t0'].max())
        # print(self.data_dic['sar_t1'].shape, self.data_dic['sar_t1'].min(), self.data_dic['sar_t1'].max())
        # print(self.data_dic['opt_t1'].shape, self.data_dic['opt_t1'].min(), self.data_dic['opt_t1'].max())
        # print(self.data_dic['opt_cloudy_t1'].shape, self.data_dic['opt_cloudy_t1'].min(), self.data_dic['opt_cloudy_t1'].max())
        # plot_hist(self.data_dic['sar_t0'], 2**16-1, None, "sar_t0_nonnorm", sample_dir)
        # plot_hist(self.data_dic['opt_t0'], 2**16-1, None, "opt_t0_nonnorm", sample_dir)
        # plot_hist(self.data_dic['opt_cloudy_t0'], 2**16-1, None, "opt_cloudy_t0_nonnorm", sample_dir)
        # plot_hist(self.data_dic['sar_t1'], 2**16-1, None, "sar_t1_nonnorm", sample_dir)
        # plot_hist(self.data_dic['opt_t1'], 2**16-1, None, "opt_t1_nonnorm", sample_dir)
        # plot_hist(self.data_dic['opt_cloudy_t1'], 2**16-1, None, "opt_cloudy_t1_nonnorm", sample_dir)

        # Normalize
        if args.date == 'both' or args.date == 'd0':
            self.data_dic["sar_t0"] = self.sar_norm.Normalize(self.data_dic["sar_t0"])
            self.data_dic["opt_t0"] = self.opt_norm.Normalize(self.data_dic["opt_t0"])
            self.data_dic["opt_cloudy_t0"] = self.opt_norm.Normalize(self.data_dic["opt_cloudy_t0"])
        if args.date == 'both' or args.date == 'd1':
            self.data_dic["sar_t1"] = self.sar_norm.Normalize(self.data_dic["sar_t1"])
            self.data_dic["opt_t1"] = self.opt_norm.Normalize(self.data_dic["opt_t1"])
            self.data_dic["opt_cloudy_t1"] = self.opt_norm.Normalize(self.data_dic["opt_cloudy_t1"])

        # plot_hist(self.data_dic['sar_t0'], 2**16-1, None, "sar_t0", sample_dir)
        # plot_hist(self.data_dic['opt_t0'], 2**16-1, None, "opt_t0", sample_dir)
        # plot_hist(self.data_dic['opt_cloudy_t0'], 2**16-1, None, "opt_cloudy_t0", sample_dir)
        # plot_hist(self.data_dic['sar_t1'], 2**16-1, None, "sar_t1", sample_dir)
        # plot_hist(self.data_dic['opt_t1'], 2**16-1, None, "opt_t1", sample_dir)
        # plot_hist(self.data_dic['opt_cloudy_t1'], 2**16-1, None, "opt_cloudy_t1", sample_dir)

        # save normalizers
        joblib.dump(self.sar_norm, self.args.datasets_dir + self.args.dataset_name  + '/' + 'sar_norm.pkl')
        joblib.dump(self.opt_norm, self.args.datasets_dir + self.args.dataset_name  + '/' + 'opt_norm.pkl')
        with open(sample_dir + '/' + 'normalization_values.txt', 'w') as f:
            f.write("SAR min-max values\n")
            q = self.sar_norm.__dict__
            for i in q.keys():
                f.write("{}: {}\n".format(i, str(q[i])))
            
            f.write("\n\n\n")
            f.write("OPT min-max values\n")
            q = self.opt_norm.__dict__
            for i in q.keys():
                f.write("{}: {}\n".format(i, str(q[i])))


        # Initialize graph
        init_op = tf1.global_variables_initializer()
        self.sess.run(init_op)

        counter = self.load(model_dir)
        if counter:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        idx = 1500
        generate_samples(self, output_path=sample_dir, idx=idx, 
                         patch_list = val_patches, epoch=counter, real_flag = True)
        

        loss_trace_G, loss_trace_D = [], []
        start_time = time.time()
        for e in xrange(counter+1, args.epoch+1):

            # Learning rate
            p = max(0.0, np.floor((e - (self.args.init_e - self.args.epoch_drop)) / self.args.epoch_drop)) 
            lr = self.args.lr * (self.args.lr_decay ** p)

            errG, errD = self.Routine_batches(train_patches, lr, e, start_time)
                
            loss_trace_G.append(errG)
            loss_trace_D.append(errD)
            np.save(model_dir + '/loss_trace_G', loss_trace_G)
            np.save(model_dir + '/loss_trace_D', loss_trace_D)

            generate_samples(self, output_path=sample_dir, idx=idx, 
                            patch_list = val_patches, epoch=e)

            # save sample
            self.save(args.checkpoint_dir, e)

    def Routine_batches(self, patch_list, lr, e, start_time):

        np.random.shuffle(patch_list)

        errD, errG = 0, 0

        batches = trange(len(patch_list) // self.batch_size)
        for batch in batches:

            # Taking the Batch
            s1, s2, s2_cloudy = [], [], []
            for im in xrange(batch*self.batch_size, (batch+1)*self.batch_size):
                batch_image = Take_patches(patch_list, idx=im,
                                           data_dic=self.data_dic,
                                           fine_size=self.image_size_tr,
                                           random_crop_transformation=True)
                s1.append(batch_image[0])
                s2.append(batch_image[1])
                s2_cloudy.append(batch_image[2])

            s1 = np.asarray(s1)
            s2 = np.asarray(s2)
            s2_cloudy = np.asarray(s2_cloudy)

            ###
                # sar = self.sar_norm.Denormalize(s1[0,:,:,:])
                # opt = self.opt_norm.Denormalize(s2[0,:,:,:])            # Save Sentinel 1
                # opt_cloudy = self.opt_norm.Denormalize(s2_cloudy[0,:,:,:])            # Save Sentinel 1

                # k = batch
                # image = (sar - self.sar_norm.min_val) / (self.sar_norm.max_val - self.sar_norm.min_val)
                # file_name = "s1_" + str(k)
                # save_image(image, file_name, sensor = "s1")

                # # Save Sentinel 2
                # image = opt[:, :, self.visible_bands] / self.opt_norm.max_val.max()
                # file_name = "s2_" + str(k)
                # save_image(image, file_name, sensor = "s2")

                # image = opt_cloudy[:, :, self.visible_bands] / self.opt_norm.max_val.max()
                # file_name = "s2_cludy" + str(k)
                # save_image(image, file_name, sensor = "s2")
                # exit(0)

            # Update D network
            _, summary_str = self.sess.run([self.d_optim, self.d_sum],
                                        feed_dict={self.SAR: s1, self.OPT: s2, self.OPT_cloudy: s2_cloudy, self.learning_rate: lr})

            # Update G network
            # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
            for _ in range(2):
                _ = self.sess.run([self.g_optim],
                                feed_dict={self.SAR: s1, self.OPT: s2, self.OPT_cloudy: s2_cloudy, self.learning_rate: lr})
            
            if np.mod(batch + 1, 1000) == 0:
                errD = self.d_loss.eval({ self.SAR: s1, self.OPT: s2, self.OPT_cloudy: s2_cloudy })
                errG = self.g_loss.eval({ self.SAR: s1, self.OPT: s2, self.OPT_cloudy: s2_cloudy })
                print("Epoch: [%2d] [%4d/%4d] lr: %.6f time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (e, (batch+1)*self.batch_size, len(patch_list), lr,
                        time.time() - start_time, errD, errG))

        return errG, errD


    def Translate_complete_image(self, args, date):


        print( 'Generating Image for ' + args.dataset_name + ' dataset')

        init_op = tf1.global_variables_initializer()
        self.sess.run(init_op)

        output_path = os.path.join(args.test_dir, self.model, args.dataset_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        model_dir = os.path.join(self.checkpoint_dir, self.model, args.dataset_name)
        mod = self.load(model_dir)
        if mod:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            return

        # Loading normalizers used during training
        self.sar_norm = joblib.load(self.args.datasets_dir + self.args.dataset_name  + '/' + 'sar_norm.pkl')
        self.opt_norm = joblib.load(self.args.datasets_dir + self.args.dataset_name  + '/' + 'opt_norm.pkl')

        if date is "t0":
            opt_cloudy_cloudmask_name = self.opt_cloudy_cloudmask_name_t0
            self.sar_name = self.sar_name_t0
            self.opt_name = self.opt_name_t0
            self.opt_cloudy_name = self.opt_cloudy_name_t0
            prefix = 0
        elif date is "t1":
            opt_cloudy_cloudmask_name = self.opt_cloudy_cloudmask_name_t1
            self.sar_name = self.sar_name_t1
            self.opt_name = self.opt_name_t1
            self.opt_cloudy_name = self.opt_cloudy_name_t1
            prefix = 1

        # Loading masks
        mask_tr_vl_ts = Split_Image(self, random_tiles='fixed')
        opt_cloudy_cloudmask = np.load(self.opt_cloudy_path + opt_cloudy_cloudmask_name + '.npy')
        opt_cloudy_cloudmask = opt_cloudy_cloudmask[:mask_tr_vl_ts.shape[0], :mask_tr_vl_ts.shape[1]]
        test_mask, mask_cloud_free, \
            mask_cloud, mask_shadow = [np.zeros_like(mask_tr_vl_ts) for i in range(4)]
                
        # mask_shadow    [opt_cloudy_cloudmask==-1] = 1
        mask_cloud     [opt_cloudy_cloudmask==1 ] = 1
        # mask_cloud_shadow = mask_cloud + mask_shadow
        mask_cloud_free = 1 - mask_cloud

        test_mask[mask_tr_vl_ts==2] = 1
        mask_cloud *= test_mask
        mask_cloud_free *= test_mask

        img = Image.fromarray(np.uint8((mask_cloud)*255))
        img.save(output_path + '/test_mask_cloud_' + date + '.tiff')

        img = Image.fromarray(np.uint8((mask_cloud_free)*255))
        img.save(output_path + '/test_mask_cloud_free_' + date + '.tiff')


        # ====================== LOAD DATA =====================
        # Loading images
        _, _, _, self.data_dic, _, _, = create_dataset_coordinates(self, prefix = prefix, padding=False,
                                                                   flag_image = [1, 0, 1], cut=False)        
        sar = self.sar_norm.Normalize(self.data_dic["sar_" + date])
        opt_cloudy = self.opt_norm.Normalize(self.data_dic["opt_cloudy_" + date])
        del self.data_dic

        start_time = time.time()
        print("Start Inference {}".format(date))
        # opt_fake = self.sess.run(self.fake_opt_t0_sample,
        #                         feed_dict={self.SAR: sar[np.newaxis, ...],
        #                                 self.OPT_cloudy: opt_cloudy[np.newaxis, ...]})
        opt_fake = Image_reconstruction([self.SAR, self.OPT_cloudy], self.fake_opt_t0_sample, 
                                        self.output_c_dim, patch_size=3840, 
                                        overlap_percent=0.02).Inference(np.concatenate((sar, opt_cloudy), axis=2))
        print("Inference complete --> {} segs".format(time.time()-start_time))
        del sar
        # 4096, 3840
        opt_cloudy = self.opt_norm.Denormalize(opt_cloudy)
        print("Saving opt_cloudy image")
        GeoReference_Raster_from_Source_data(self.opt_path + self.opt_name[prefix] + '.tif', 
                                             opt_cloudy.transpose(2, 0, 1),
                                             output_path + '/S2_cloudy_' + date + '_10bands.tif')
        del opt_cloudy

        opt_fake = self.opt_norm.Denormalize(opt_fake)
        print("Saving opt_fake image")
        GeoReference_Raster_from_Source_data(self.opt_path + self.opt_name[prefix] + '.tif', 
                                             opt_fake.transpose(2, 0, 1),
                                             output_path + '/S2_' + date + '_10bands' + '_Fake_.tif')
        np.save(output_path + '/S2_' + date + '_10bands' + '_Fake_', opt_fake)


        # Loading Free-cloud image
        _, _, _, self.data_dic, _, _, = create_dataset_coordinates(self, prefix = prefix, padding=False,
                                                                   flag_image = [0, 1, 0], cut=False)
        opt = self.opt_norm.clip_image(self.data_dic["opt_" + date])
        del self.data_dic
        print("Saving opt_cloudy image")
        GeoReference_Raster_from_Source_data(self.opt_path + self.opt_name[prefix] + '.tif', 
                                             opt.transpose(2, 0, 1),
                                             output_path + '/S2_' + date + '_10bands.tif')

        ########### METRICS ##################
        opt =           opt[self.lims[0]:self.lims[1], self.lims[2]:self.lims[3],:]
        opt_fake = opt_fake[self.lims[0]:self.lims[1], self.lims[2]:self.lims[3],:]

        with open(output_path + '/' + 'Similarity_Metrics.txt', 'a') as f:
            # test area (cloudy)
            mae, mse, rmse, psnr, sam, ssim = METRICS(opt, opt_fake, mask_cloud)
            Write_metrics_on_file(f, "Metrics " + date + "-- Test area(cloudy)", mae, mse, rmse, psnr, sam, ssim)
            # test area (cloud-free)
            mae, mse, rmse, psnr, sam, ssim = METRICS(opt, opt_fake, mask_cloud_free)
            Write_metrics_on_file(f, "Metrics " + date + "-- Test area(cloud-free)", mae, mse, rmse, psnr, sam, ssim)
            # test area
            mae, mse, rmse, psnr, sam, ssim = METRICS(opt, opt_fake, test_mask, ssim_flag=True, dataset=args.dataset_name)
            Write_metrics_on_file(f, "Metrics " + date + "-- Test area", mae, mse, rmse, psnr, sam, ssim)

        del opt, opt_fake

    def Meraner_metrics(self, args, date):

        path = args.test_dir + '/Meraner_approach/' + args.dataset_name + '/'
        
        # Loading normalizers used during training
        self.opt_norm = joblib.load(self.args.datasets_dir + self.args.dataset_name  + '/' + 'opt_norm.pkl')
        
        if date is "t0":
            opt_cloudy_cloudmask_name = self.opt_cloudy_cloudmask_name_t0
            self.sar_name = self.sar_name_t0
            self.opt_name = self.opt_name_t0
            self.opt_cloudy_name = self.opt_cloudy_name_t0
            prefix = 0
            
            # Pará
            # file_ = 'predictions_pretrained_2018.tif'
            # output_file = 'predictions_pretrained.txt'

            # file_ = 'predictions_scratch_2018.tif'
            # output_file = 'predictions_scratch.txt'

            # file_ = 'predictions_remove60m_2018.tif'
            # output_file = 'predictions_remove60m.txt'

            # file_ = 'predictions_scratch_2018_60epoch.tif'
            # output_file = 'predictions_scratch_60epoch.txt'

            # Mato Grosso
            file_ = 'predictions_scratch_MG_2019.tif'
            output_file = 'predictions_scratch_MG.txt'

        elif date is "t1":
            opt_cloudy_cloudmask_name = self.opt_cloudy_cloudmask_name_t1
            self.sar_name = self.sar_name_t1
            self.opt_name = self.opt_name_t1
            self.opt_cloudy_name = self.opt_cloudy_name_t1
            prefix = 1
            
            # Pará
            # file_ = 'predictions_pretrained_2019.tif'
            # output_file = 'predictions_pretrained.txt'

            # file_ = 'predictions_scratch_2019.tif'
            # output_file = 'predictions_scratch.txt'

            # file_ = 'predictions_remove60m_2019.tif'
            # output_file = 'predictions_remove60m.txt'

            # file_ = 'predictions_scratch_2019_60epoch.tif'
            # output_file = 'predictions_scratch_60epoch.txt'

            # Mato Grosso
            file_ = 'predictions_scratch_MG_2020.tif'
            output_file = 'predictions_scratch_MG.txt'

        # Loading masks
        mask_tr_vl_ts = Split_Image(self, random_tiles='fixed')
        opt_cloudy_cloudmask = np.load(self.opt_cloudy_path + opt_cloudy_cloudmask_name + '.npy')
        opt_cloudy_cloudmask = opt_cloudy_cloudmask[:mask_tr_vl_ts.shape[0], :mask_tr_vl_ts.shape[1]]
        test_mask, mask_cloud_free, \
            mask_cloud, mask_shadow = [np.zeros_like(mask_tr_vl_ts) for i in range(4)]
                
        # mask_shadow    [opt_cloudy_cloudmask==-1] = 1
        mask_cloud     [opt_cloudy_cloudmask==1 ] = 1
        # mask_cloud_shadow = mask_cloud + mask_shadow
        mask_cloud_free = 1 - mask_cloud

        test_mask[mask_tr_vl_ts==2] = 1
        mask_cloud *= test_mask
        mask_cloud_free *= test_mask

        # img = Image.fromarray(np.uint8((mask_cloud)*255))
        # img.save(path + '/test_mask_cloud_' + date + '.tiff')

        # img = Image.fromarray(np.uint8((mask_cloud_free)*255))
        # img.save(path + '/test_mask_cloud_free_' + date + '.tiff')

        # ====================== LOAD DATA =====================
        opt_fake = load_tiff_image(path + file_).astype('float32')
        if opt_fake.shape[0] == 13:
            opt_fake = opt_fake[[1, 2, 3, 4, 5, 6, 7, 8, 11, 12], :, :]
        opt_fake = opt_fake.transpose([1, 2, 0])
        opt_fake[np.isnan(opt_fake)] = np.nanmean(opt_fake)
        opt_fake = self.opt_norm.clip_image(opt_fake)

        _, _, _, self.data_dic, _, _, = create_dataset_coordinates(self, prefix = prefix, padding=False,
                                                                   flag_image = [0, 1, 0], cut=False)
        opt = self.opt_norm.clip_image(self.data_dic["opt_" + date])
        del self.data_dic

        ########### METRICS ##################
        opt =           opt[self.lims[0]:self.lims[1], self.lims[2]:self.lims[3],:]
        opt_fake = opt_fake[self.lims[0]:self.lims[1], self.lims[2]:self.lims[3],:]

        with open(path + output_file, 'a') as f:
            # test area (cloudy)
            mae, mse, rmse, psnr, sam, ssim = METRICS(opt, opt_fake, mask_cloud)
            Write_metrics_on_file(f, "Metrics " + date + "-- Test area(cloudy)", mae, mse, rmse, psnr, sam, ssim)
            # test area (cloud-free)
            mae, mse, rmse, psnr, sam, ssim = METRICS(opt, opt_fake, mask_cloud_free)
            Write_metrics_on_file(f, "Metrics " + date + "-- Test area(cloud-free)", mae, mse, rmse, psnr, sam, ssim)
            # test area
            mae, mse, rmse, psnr, sam, ssim = METRICS(opt, opt_fake, test_mask, ssim_flag=True, dataset=args.dataset_name)
            Write_metrics_on_file(f, "Metrics " + date + "-- Test area", mae, mse, rmse, psnr, sam, ssim)

        del opt, opt_fake

    def GEE_metrics(self, args, date):

        # Loading normalizers used during training
        self.opt_norm = joblib.load(self.args.datasets_dir + self.args.dataset_name  + '/' + 'opt_norm.pkl')
        output_file = '/Similarity_Metrics.txt'
        
        if date is "t0":
            self.sar_name = self.sar_name_t0
            self.opt_name = self.opt_name_t0
            self.opt_cloudy_name = self.opt_cloudy_name_t0
            prefix = 0
            
            # Pará
            # path = args.test_dir + '/GEE/' + args.dataset_name
            # file_ = '/img_2018.tif'

            # Mato Grosso OK
            # path = args.test_dir + '/GEE/' + args.dataset_name
            # file_ = '/2019_09_15_2019_09_30.tif'

        elif date is "t1":
            self.sar_name = self.sar_name_t1
            self.opt_name = self.opt_name_t1
            self.opt_cloudy_name = self.opt_cloudy_name_t1
            prefix = 1
            
            # Pará
            # path = args.test_dir + '/GEE/' + args.dataset_name
            # file_ = '/img_2019.tif'

            # path = args.test_dir + '/GEE_wet_season/' + args.dataset_name
            # file_ = '/2019_01_01_2019_02_01.tif'
            # output_file = '/one month.txt'

            # path = args.test_dir + '/GEE_wet_season/' + args.dataset_name
            # file_ = '/2019_01_01_2019_04_01.tif'
            # output_file = '/three months.txt'

            # Mato Grosso OK
            # path = args.test_dir + '/GEE/' + args.dataset_name
            # file_ = '/2020_09_10_2020_09_30.tif'

            # path = args.test_dir + '/GEE_wet_season/' + args.dataset_name
            # file_ = '/MG_1month_2020.tif'
            # output_file = '/one month.txt'

            path = args.test_dir + '/GEE_wet_season/' + args.dataset_name
            file_ = '/MG_3months_2020.tif'
            output_file = '/three months.txt'

        # Loading mask
        mask_tr_vl_ts = Split_Image(self, random_tiles='fixed')
        test_mask  = np.zeros_like(mask_tr_vl_ts)                
        test_mask[mask_tr_vl_ts==2] = 1

        # ====================== LOAD DATA =====================
        opt_fake = load_tiff_image(path + file_).astype('float32')
        opt_fake = opt_fake[[1, 2, 3, 4, 5, 6, 7, 8, 11, 12], :, :]
        opt_fake = opt_fake.transpose([1, 2, 0])
        opt_fake[np.isnan(opt_fake)] = np.nanmean(opt_fake)
        opt_fake = self.opt_norm.clip_image(opt_fake)

        _, _, _, self.data_dic, _, _, = create_dataset_coordinates(self, prefix = prefix, padding=False, 
                                                                   flag_image = [0, 1, 0], cut=False)
        opt = self.opt_norm.clip_image(self.data_dic["opt_" + date])
        del self.data_dic

        ########### METRICS ##################
        opt =           opt[self.lims[0]:self.lims[1], self.lims[2]:self.lims[3],:]
        opt_fake = opt_fake[self.lims[0]:self.lims[1], self.lims[2]:self.lims[3],:]

        with open(path + output_file, 'a') as f:
            # Complete image
            mae, mse, rmse, psnr, sam, ssim = METRICS(opt, opt_fake, ssim_flag=True)
            Write_metrics_on_file(f, "Metrics " + date + "-- Complete Image", mae, mse, rmse, psnr, sam, ssim)

            # test area
            mae, mse, rmse, psnr, sam, ssim = METRICS(opt, opt_fake, test_mask, ssim_flag=True, dataset=args.dataset_name)
            Write_metrics_on_file(f, "Metrics " + date + "-- Test area", mae, mse, rmse, psnr, sam, ssim)





    def Translate_samples(self, args):

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        output_path = os.path.join(args.test_dir, self.dataset_name)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        model_dir = os.path.join(self.checkpoint_dir, self.model)
        mod = self.load(model_dir)
        if mod:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            return

        print( 'Translating samples for ' + args.dataset_name)

        # LOAD TEST DATA #############
        print("Loading test data... [*]")
        test_samples = np.concatenate((self.test_patches, self.val_patches))
        patches = self.Multiproc(range(test_samples.shape[0]), test_samples)
        print("Test data loaded!!")
        #############################

        s1, s2, s2_fake = [], [], []
        for im in range(len(patches)):
            s1.append(patches[im][0])
            s2.append(patches[im][1])
            
            fake_patch = self.sess.run(self.fake_opt_t0_sample,
                            feed_dict={self.SAR: patches[im][0][np.newaxis, ...]})
            s2_fake.append(fake_patch)

        s1 = np.asarray(s1)
        s2 = np.asarray(s2)
        s2_fake = np.asarray(s2_fake)

        ########### METRICS ##################
        mae  = MAE (s2, s2_fake)
        rmse = RMSE(s2, s2_fake)
        psnr = PSNR(s2, s2_fake)
        sam  = SAM (s2, s2_fake)

        for i in range(s2.shape[0]):
            s1[i,:,:,:] = self.norm_routine.Denormalize(s1[i,:,:,:], "s1")
            s2[i,:,:,:] = self.norm_routine.Denormalize(s2[i,:,:,:], "s2")
            s2_fake[i,:,:,:] = self.norm_routine.Denormalize(s2_fake[i,:,:,:], "s2")

            # Save Sentinel 1
            image = (s1[i,:,:,:] - self.norm_routine.clips_s1[0]) / (self.norm_routine.clips_s1[1] - self.norm_routine.clips_s1[0])
            file_name = "/{}_{}_p{}_{}".format(self.data_files[test_samples[i]][0], 
                                               self.data_files[test_samples[i]][1], 
                                               self.data_files[test_samples[i]][2], "s1")
            save_image(image, output_path + file_name, sensor = "s1")

            # Save Sentinel 2
            image = s2[i, :, :, self.visible_bands] / self.norm_routine.clips_s2[1]
            file_name = "/{}_{}_p{}_{}".format(self.data_files[test_samples[i]][0], 
                                               self.data_files[test_samples[i]][1], 
                                               self.data_files[test_samples[i]][2], "s2")
            save_image(image, output_path + file_name, sensor = "s2")

            # Save Sentinel 2 Fake
            image = s2_fake[i, :, :, self.visible_bands] / self.norm_routine.clips_s2[1]
            file_name = "/{}_{}_p{}_{}".format(self.data_files[test_samples[i]][0], 
                                               self.data_files[test_samples[i]][1], 
                                               self.data_files[test_samples[i]][2], "s2_fake")
            save_image(image, output_path + file_name, sensor = "s2")

        psnr = PSNR(s2, s2_fake)
        ssim = SSIM(s2, s2_fake)

        print("MAE  = %.4f"%(mae))
        print("RMSE = %.4f"%(rmse))
        print("PSNR = %.4f"%(psnr))
        print("SAM  = %.4f"%(sam))
        print("SSIM = %.4f"%(ssim))


    def save(self, checkpoint_dir, step):
        model_name = "pix2pix.model"

        checkpoint_dir = os.path.join(checkpoint_dir, self.model, self.args.dataset_name)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
        print("Saving checkpoint!")

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        print(checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            aux = 'model_example'
            for i in range(len(ckpt_name)):
                if ckpt_name[-i-1] == '-':
                    aux = ckpt_name[-i:]
                    break
            return int(aux)
        else:
            return int(0)


#    def generate_image(self, args):

    #     init_op = tf1.global_variables_initializer()
    #     self.sess.run(init_op)

    #     model_dir = os.path.join(self.checkpoint_dir, args.dataset_name, self.model)
    #     output_path = os.path.join(model_dir, 'samples')

    #     mod = self.load(model_dir)
    #     if mod:
    #         print(" [*] Load SUCCESS")
    #     else:
    #         print(" [!] Load failed...")
    #         return

    #     print( 'Generating Image for_' + args.dataset_name)

    #     # Percent of overlap between consecutive patches.
    #     # The overlap will be multiple of 2 and 3, this guarantes to
    #     # use the same variables to construct the optical image.
    #     P = 0.75
    #     overlap = 3 * round(self.image_size * P)
    #     overlap -= overlap % 6
    #     stride = 3 * self.image_size - overlap
    #     stride_opt   = stride // 3
    #     overlap_opt  = overlap // 3

    #     # Opening SAR Image
    #     print('sar_name: %s' %(self.sar_name))
    #     sar_path_t0 = self.sar_path + self.sar_name + '/' + self.sar_name + '.npy'
    #     sar_t0 = np.load(sar_path_t0).astype('float32')
                
    #     # Normalization
    #     scaler = joblib.load('../datasets/' + args.dataset_name + '/Norm_params' + \
    #                          '/sar_' + self.sar_name   + '_' + self.norm_type + '.pkl')
    #     num_rows, num_cols, _ = sar_t0.shape
    #     sar_t0 = sar_t0.reshape((num_rows * num_cols, -1))
    #     sar_t0 = scaler.transform(sar_t0)
    #     sar_t0 = sar_t0.reshape((num_rows, num_cols, -1))

    #     # Add Padding to the image to match with the patch size and the overlap
    #     step_row = (stride - num_rows % stride) % stride
    #     step_col = (stride - num_cols % stride) % stride

    #     pad_tuple = ( (overlap//2, overlap//2 + step_row), ((overlap//2, overlap//2 + step_col)), (0,0) )
    #     sar_pad_t0 = np.pad(sar_t0, pad_tuple, mode = 'symmetric')

    #     # Number of patches: k1xk2
    #     k1, k2 = (num_rows+step_row)//stride, (num_cols+step_col)//stride
    #     print('Number of patches: %d x %d' %(k1, k2))

    #     # Inference
    #     fake_pad_opt_t0 = np.zeros((k1*stride_opt, k2*stride_opt, self.output_c_dim))
    #     # for k in range(1):
    #     sample_name = self.model  # + "_" +  str(k)
    #     print(sample_name)
    #     start = time.time()
    #     for i in range(k1):
    #         for j in range(k2):
                
    #             sar_t0 = sar_pad_t0[i*stride:(i*stride + 3*self.image_size),
    #                                 j*stride:(j*stride + 3*self.image_size), :]
    #             sar_t0 = sar_t0.reshape(1, 3*self.image_size, 3*self.image_size, -1)

    #             fake_patch = self.sess.run(self.fake_opt_t0_sample,
    #                                        feed_dict={self.SAR: sar_t0})

    #             fake_pad_opt_t0[i*stride_opt : i*stride_opt+stride_opt, 
    #                             j*stride_opt : j*stride_opt+stride_opt, :] = fake_patch[0, overlap_opt//2 : overlap_opt//2 + stride_opt, 
    #                                                                                         overlap_opt//2 : overlap_opt//2 + stride_opt, :]
            
    #         print('row %d: %.2f min' %(i+1, (time.time() - start)/60))
        
    #     # Taken off the padding
    #     rows = k1*stride_opt-step_row//3
    #     cols = k2*stride_opt-step_col//3
    #     fake_opt_t0 = fake_pad_opt_t0[:rows, :cols]
        
    #     print('fake_opt_t0 size: ');  print(fake_opt_t0.shape)
    #     print('Inference time: %.2f min' %((time.time() - start)/60))

    #     # Denomarlize
    #     if self.norm_type == 'wise_frame_mean':
    #         scaler = joblib.load('../datasets/' + args.dataset_name + '/Norm_params' + \
    #                              '/opt_' + self.opt_name + '_' + 'std' + '.pkl')
    #     else:
    #         scaler = joblib.load('../datasets/' + args.dataset_name + '/Norm_params' + \
    #                              '/opt_' + self.opt_name + '_' + self.norm_type + '.pkl')
    #     fake_opt_t0 = Denormalization(fake_opt_t0, scaler)

    #     # np.save(output_path + '/' + sample_name, fake_opt_t0)
    #     np.savez_compressed(output_path + '/' + sample_name, fake_opt_t0)
    #     sio.savemat(output_path + '/' + sample_name,  {sample_name: fake_opt_t0})

#
    def count_params(self, t_vars):
        """
        print number of trainable variables
        """
        n = np.sum([np.prod(v.get_shape().as_list()) for v in t_vars])
        print("Model size: %dK params" %(n/1000))

        # init_op = tf1.global_variables_initializer()
        # self.sess.run(init_op)
        # w = self.sess.run(t_vars)
        # for val, var in zip(w, t_vars):
        #     # if 'biases' not in var.name:
        #     print(var.name)
        #     print(val.shape)
        # #         # break
        # exit(0)
