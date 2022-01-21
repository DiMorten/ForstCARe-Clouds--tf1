"""
No@
"""
from __future__ import division
import os
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
import tensorflow as tf
from time import gmtime, strftime
from PIL import Image
import glob
from skimage.transform import resize
from skimage import exposure
from skimage.metrics import structural_similarity
from sklearn import preprocessing as pre
from sklearn.preprocessing._data import _handle_zeros_in_scale
import matplotlib.pyplot as plt
import collections  
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import joblib
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)
import sys
from skimage.util.shape import view_as_windows
import itertools
import multiprocessing
from time import sleep
from functools import partial
from osgeo import gdal
#from sen12ms_cr_dataLoader import *
import rasterio
from icecream import ic
import pdb
pp = pprint.PrettyPrinter()
import cv2
get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def load_tiff_image(path):
    # Read tiff Image
    print (path) 
    gdal_header = gdal.Open(path)
    img = gdal_header.ReadAsArray()
    return img

def GeoReference_Raster_from_Source_data(source_file, numpy_image, target_file):

    with rasterio.open(source_file) as src:
        ras_meta = src.profile

    ras_meta.update(count=10)

    with rasterio.open(target_file, 'w', **ras_meta) as dst:
        dst.write(numpy_image)


# def load_tiff_image(path):
    # print(path)
    # img = np.array(Image.open(path))
    # return img

def load_landsat(path):
    images = sorted(glob.glob(path + '/*.tif'))
    band = load_tiff_image(images[0])
    rows, cols = band.shape
    img = np.zeros((rows, cols, 7), dtype='float32')
    num_band = 0
    for im in images:
        band = load_tiff_image(im)
        img[:, :, num_band] = band
        num_band += 1

    return img

def Normalization(img, mask, norm_type, scaler_name="scaler"):

    num_rows, num_cols, bands = img.shape
    img = img.reshape(num_rows * num_cols, bands)

    if norm_type == 'min_max':
        scaler = pre.MinMaxScaler((-1, 1)).fit(img[mask.ravel() == 1])
        print('min_max normalization!!!')
    elif norm_type == 'std':
        scaler = pre.StandardScaler().fit(img[mask.ravel() == 1])
        print('std normalization!!!')
    elif norm_type == 'wise_frame_mean':
        scaler = pre.StandardScaler(with_std=False).fit(img[mask.ravel() == 1])
        print('wise_frame_mean normalization!!!')
    else:
        print('without normalization!!!')
        img = img.reshape(num_rows, num_cols, bands)
        return img
    
    # save scaler
    joblib.dump(scaler, scaler_name  + '_' + norm_type + '.pkl')
    img = np.float32(scaler.transform(img))
    img = img.reshape(num_rows, num_cols, bands)

    return img

def Denormalization(img, scaler):
    rows, cols, _ = img.shape
    img = img.reshape((rows * cols, -1))
    img = scaler.inverse_transform(img)
    img = img.reshape((rows, cols, -1))
    return img
    

def filter_outliers(img, bins=2**16-1, bth=0.001, uth=0.999, mask=[0]):
    img[np.isnan(img)] = np.mean(img) # Filter NaN values.
    if len(mask)==1:
        mask = np.zeros((img.shape[:2]), dtype='int64')
    min_value, max_value = [], []
    for band in range(img.shape[-1]):
        hist = np.histogram(img[:mask.shape[0], :mask.shape[1]][mask!=2, band].ravel(), bins=bins) # select not testing pixels
        cum_hist = np.cumsum(hist[0])/hist[0].sum()
        min_value.append(hist[1][len(cum_hist[cum_hist<bth])])
        max_value.append(hist[1][len(cum_hist[cum_hist<uth])])
        
    return [np.array(min_value), np.array(max_value)]


class Min_Max_Norm_Denorm():

    def __init__(self, img, mask, feature_range=[-1, 1]):

        self.feature_range = feature_range
        self.clips = filter_outliers(img.copy(), bins=2**16-1, bth=0.0005, uth=0.9995, mask=mask)
        self.min_val = np.nanmin(img, axis=(0,1))
        self.max_val = np.nanmax(img, axis=(0,1))

        self.min_val = np.clip(self.min_val, self.clips[0], None)
        self.max_val = np.clip(self.max_val, None, self.clips[1])
    
    def clip_image(self, img):
        return np.clip(img.copy(), self.clips[0], self.clips[1])

    def Normalize(self, img):
        data_range = self.max_val - self.min_val
        scale = (self.feature_range[1] - self.feature_range[0]) / _handle_zeros_in_scale(data_range)
        min_ = self.feature_range[0] - self.min_val * scale
        
        img = self.clip_image(img.copy())
        img *= scale
        img += min_
        return img

    def Denormalize(self, img):
        data_range = self.max_val - self.min_val
        scale = (self.feature_range[1] - self.feature_range[0]) / _handle_zeros_in_scale(data_range)
        min_ = self.feature_range[0] - self.min_val * scale

        img = img.copy() - min_
        img /= scale
        return img



def Split_Tiles(tiles_list, xsz, ysz, stride=256, patch_size=256):
   
    coor = []
    for i in tiles_list:
        b = np.random.choice([-1, 1])
        if b == 1:
            x = np.arange(0, xsz - patch_size + 1, b*stride)
        else:
            x = np.arange(xsz - patch_size, -1, b*stride)
       
        b = np.random.choice([-1, 1])
        if b == 1:
            y = np.arange(0, ysz - patch_size + 1, b*stride)
        else:
            y = np.arange(ysz - patch_size, -1, b*stride)
       
        coor += list(itertools.product(x + i[0], y + i[1]))

    for i in range(len(coor)):
        coor[i] = (coor[i][0], coor[i][1], i)

    return coor

def Split_Image(obj, rows=1000, cols=1000, no_tiles_h=5, no_tiles_w=5, random_tiles = "fixed"):

    xsz = rows // no_tiles_h
    ysz = cols // no_tiles_w

    if random_tiles == 'random':

        # Tiles coordinates
        h = np.arange(0, rows, xsz)
        w = np.arange(0, cols, ysz)
        if (rows % no_tiles_h): h = h[:-1]
        if (cols % no_tiles_w): w = w[:-1]
        tiles = list(itertools.product(h, w))

        np.random.seed(3); np.random.shuffle(tiles)

        # Take test tiles
        idx = len(tiles) * 50 // 100; idx += (idx == 0)
        test_tiles = tiles[:idx]
        train_tiles = tiles[idx:]
        # Take validation tiles
        idx = len(train_tiles) * 10 // 100; idx += (idx == 0)
        val_tiles = train_tiles[:idx]
        train_tiles = train_tiles[idx:]

        mask = np.zeros((rows, cols))
        for i in val_tiles:
            finx = rows if (rows-(i[0] + xsz) < xsz) else (i[0] + xsz)
            finy = cols if (cols-(i[1] + ysz) < ysz) else (i[1] + ysz)
            mask[i[0]:finx, i[1]:finy] = 1
        for i in test_tiles:
            finx = rows if (rows-(i[0] + xsz) < xsz) else (i[0] + xsz)
            finy = cols if (cols-(i[1] + ysz) < ysz) else (i[1] + ysz)
            mask[i[0]:finx, i[1]:finy] = 2
        
        # save_mask = Image.fromarray(np.uint8(mask*255/2))
        # save_mask.save('../datasets/' + obj.args.dataset_name + '/mask_train_val_test.tif')
        # np.save('../datasets/' + obj.args.dataset_name + '/tiles', tiles)

    elif random_tiles == 'k-fold':

        k = obj.args.k
        mask = Image.open('../datasets/' + obj.args.dataset_name + '/mask_train_val_test_fold_' + str(k) + '.tif')
        mask = np.array(mask) / 255

        # tiles = np.load('../datasets/' + obj.args.dataset_name + '/tiles.npy')

        # # Split in folds
        # size_fold = len(tiles) // obj.args.n_folds
        # test_tiles = tiles[k*size_fold:(k+1)*size_fold]
        # train_tiles = np.concatenate((tiles[:k*size_fold], tiles[(k+1)*size_fold:]))
        # # Take validation tiles
        # np.random.shuffle(train_tiles)
        # idx = len(train_tiles) * 10 // 100; idx += (idx == 0)
        # val_tiles = train_tiles[:idx]
        # train_tiles = train_tiles[idx:]

        # mask = np.zeros((rows, cols))
        # for i in val_tiles:
        #     finx = rows if (rows-(i[0] + xsz) < xsz) else (i[0] + xsz)
        #     finy = cols if (cols-(i[1] + ysz) < ysz) else (i[1] + ysz)
        #     mask[i[0]:finx, i[1]:finy] = 1
        # for i in test_tiles:
        #     finx = rows if (rows-(i[0] + xsz) < xsz) else (i[0] + xsz)
        #     finy = cols if (cols-(i[1] + ysz) < ysz) else (i[1] + ysz)
        #     mask[i[0]:finx, i[1]:finy] = 2
        
        # save_mask = Image.fromarray(np.uint8(mask*255))
        # save_mask.save('../datasets/' + obj.args.dataset_name + '/mask_train_val_test_fold_' + str(k) + '.tif')

    elif random_tiles == 'fixed':
        # Distribute the tiles from a mask
        #ic(obj.args.datasets_dir + obj.args.dataset_name + obj.mask_tr_vl_ts_name + '.npy')
        if obj.args.dataset_name == 'Santarem_I1' or obj.args.dataset_name == 'Santarem_I2' or obj.args.dataset_name == 'Santarem_I3' or obj.args.dataset_name == 'Santarem_I4' or obj.args.dataset_name == 'Santarem_I5':
            model_dataset_name = 'Santarem'
        else:
            model_dataset_name = obj.args.dataset_name        
        mask  = np.load(obj.args.datasets_dir + model_dataset_name + obj.mask_tr_vl_ts_name + '.npy')
        #ic(mask.shape)
        #pdb.set_trace()
        img = Image.fromarray(np.uint8((mask/2)*255))
        img.save(obj.args.datasets_dir + model_dataset_name + obj.mask_tr_vl_ts_name + '.tiff')
    
    return mask

def Split_in_Patches(rows, cols, patch_size, mask, 
#                     lbl, augmentation_list, cloud_mask, 
                     augmentation_list, cloud_mask, 
                     prefix=0, percent=0):

    """
    Everything  in this function is made operating with
    the upper left corner of the patch
    """
    # nan_mask
    nan_mask = np.load('D:/Jorge/dataset/NRW/nan_mask.npy')

    # Percent of overlap between consecutive patches.
    overlap = round(patch_size * percent)
    overlap -= overlap % 2
    stride = patch_size - overlap
    # Add Padding to the image to match with the patch size
    step_row = (stride - rows % stride) % stride
    step_col = (stride - cols % stride) % stride
    pad_tuple_msk = ( (overlap//2, overlap//2 + step_row), ((overlap//2, overlap//2 + step_col)) )
#    lbl = np.pad(lbl, pad_tuple_msk, mode = 'symmetric')
    mask_pad = np.pad(mask, pad_tuple_msk, mode = 'symmetric')
    cloud_mask = np.pad(cloud_mask, pad_tuple_msk, mode = 'symmetric')

    k1, k2 = (rows+step_row)//stride, (cols+step_col)//stride
    print('Total number of patches: %d x %d' %(k1, k2))

    train_mask = np.zeros_like(mask_pad)
    val_mask = np.zeros_like(mask_pad)
    test_mask = np.zeros_like(mask_pad)
    train_mask[mask_pad==0] = 1
    test_mask [mask_pad==2] = 1
    val_mask = (1-train_mask) * (1-test_mask)

    train_patches, val_patches, test_patches = [], [], []
    only_bck_patches = 0
    cloudy_patches = 0
#    lbl[lbl!=1] = 0
    for i in range(k1):
        for j in range(k2):
            # Train
            if train_mask[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].all():
                if cloud_mask[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].any():
                    cloudy_patches += 1
                    continue
                if nan_mask[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].any():
                    continue 
                for k in augmentation_list:
                    train_patches.append((prefix, i*stride, j*stride, k))
#                if not lbl[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].any():
                    # train_patches.append((prefix, i*stride, j*stride, 0))
#                    only_bck_patches += 1
            # Test                !!!!!Not necessary with high overlap!!!!!!!!
            elif test_mask[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].all():
                test_patches.append((prefix, i*stride, j*stride, 0))
            # Val                 !!!!!Not necessary with high overlap!!!!!!!!
            elif val_mask[i*stride:i*stride + patch_size, j*stride:j*stride + patch_size].all():
                val_patches.append((prefix, i*stride, j*stride, 0))
    print('Training Patches with background only: %d' %(only_bck_patches))
    print('Patches with clouds in the cloud-free image: %d' %(cloudy_patches))
    
    return train_patches, val_patches, test_patches, step_row, step_col, overlap

def create_dataset_coordinates(obj, prefix = 0, padding=True,
                               flag_image = [1, 1, 1], cut=True):
    
    '''
        Generate patches for trn, val and tst
    '''

    patch_size = obj.image_size_tr

    # number of tiles per axis
    no_tiles_h, no_tiles_w = 5, 5
    rows, cols = obj.lims[1] - obj.lims[0], obj.lims[3] - obj.lims[2]
    mask_tr_vl_ts = Split_Image(obj, rows, cols, no_tiles_h, 
                           no_tiles_w, random_tiles=obj.args.mask)

    # Loading Labels
    # lbl = np.load(obj.labels_path + obj.labels_name + '.npy')
    # lbl = lbl[obj.lims[0]: obj.lims[1], obj.lims[2]: obj.lims[3]]
    # lbl[lbl==2.0] = 3.0; lbl[lbl==1.0] = 2.0; lbl[lbl==3.0] = 1.0
    # img = Image.fromarray(np.uint8((lbl/2)*255))
    # img.save(obj.labels_path + obj.labels_name + '.tiff')

    # Loading cloud mask
    cloud_mask = np.zeros((rows, cols))    
    
    # Generate Patches for trn, val and tst
    if obj.args.data_augmentation:
        augmentation_list = [-1]                    # A random transformation each epoch
        # augmentation_list = [0, 1, 2, 3, 4, 5]    # All transformation each epoch
    else:
        augmentation_list = [0]                         # Without transformations
    train_patches, val_patches, test_patches, \
    step_row, step_col, overlap = Split_in_Patches(rows, cols, patch_size, 
#                                                   mask_tr_vl_ts, lbl, augmentation_list,
                                                   mask_tr_vl_ts, augmentation_list,
                                                   cloud_mask, prefix = prefix,
                                                   percent=obj.args.patch_overlap)
    pad_tuple = ( (overlap//2, overlap//2+step_row), (overlap//2, overlap//2+step_col), (0,0) )
#    del lbl, cloud_mask
    del cloud_mask
    
    print('--------------------')
    print('Training Patches: %d' %(len(train_patches)))
    print('Validation Patches: %d' %(len(val_patches)))
    print('Testing Patches: %d' %(len(test_patches)))
    
    data_dic = {}

    # Sentinel 1
    if flag_image[0]:
        sar_vv = load_tiff_image(obj.sar_path + obj.sar_name[0] + '.tif').astype('float32')
        sar_vh = load_tiff_image(obj.sar_path + obj.sar_name[1] + '.tif').astype('float32')
        sar = np.concatenate((np.expand_dims(sar_vv, 2), np.expand_dims(sar_vh, 2)), axis=2)
        del sar_vh, sar_vv
        '''
        if obj.dataset_name != 'Santarem':
            sar_vv = load_tiff_image(obj.sar_path + obj.sar_name[0] + '.tif').astype('float32')
            sar_vh = load_tiff_image(obj.sar_path + obj.sar_name[1] + '.tif').astype('float32')
            sar = np.concatenate((np.expand_dims(sar_vv, 2), np.expand_dims(sar_vh, 2)), axis=2)
            del sar_vh, sar_vv
        else:
            sar = load_tiff_image(obj.sar_path + obj.sar_name[0] + '.tif').astype('float32')    
            sar = np.transpose(sar, (1, 2, 0))  
        ''' 
        if cut:
            sar = sar[obj.lims[0]:obj.lims[1], obj.lims[2]:obj.lims[3],:]
        sar[np.isnan(sar)] = np.nanmean(sar)
        sar = db2linear(sar)
        sar_norm = Min_Max_Norm_Denorm(sar, mask_tr_vl_ts)
    else:
        sar, sar_norm = [], []
    
    if padding:
        # Add Padding to the images to match with the patch size
        data_dic["sar_t" + str(prefix)] = np.pad(sar, pad_tuple, mode = 'symmetric')
    else:
        data_dic["sar_t" + str(prefix)] = sar
    del sar

    # Sentinel 2
    if flag_image[1]:
        for i in range(len(obj.opt_name)): 
            ic(obj.opt_path + obj.opt_name[i] + '.tif')       
            isNrwDataset = True
            if isNrwDataset == True:
                img = load_tiff_image(obj.opt_path + obj.opt_name[i]).astype('float32')
                bands_res = ['60m', '10m', '10m', '10m', '20m', '20m', '20m', '10m', 
                    '20m', '60m', '10m', '20m', '20m']
                dim = (10980, 10980)
                # ic(bands_res[i])
                if bands_res[i] == '20m' or bands_res[i] == '60m':
                    img = cv2.resize(img, dim, interpolation = cv2.INTER_NEAREST)
            else:
                img = load_tiff_image(obj.opt_path + obj.opt_name[i] + '.tif').astype('float32')

            if len(img.shape) == 2: img = img[np.newaxis, ...]
            if i:
                opt = np.concatenate((opt, img), axis=0)
            else:
                opt = img
        del img
        # np.save('s2' + '.npy', opt)

        opt = opt[[1, 2, 3, 4, 5, 6, 7, 8, 11, 12], :, :]
        opt = opt.transpose([1, 2, 0])
        if cut:
            opt = opt[obj.lims[0]:obj.lims[1], obj.lims[2]:obj.lims[3],:]
        opt[np.isnan(opt)] = np.nanmean(opt)
        opt_norm = Min_Max_Norm_Denorm(opt, mask_tr_vl_ts)
    else:
        opt, opt_norm = [], []

    if padding:
        # Add Padding to the images to match with the patch size
        data_dic["opt_t" + str(prefix)] = np.pad(opt, pad_tuple, mode = 'symmetric')
    else:
        data_dic["opt_t" + str(prefix)] = opt
    del opt

    # Sentinel 2 cloudy
    if flag_image[2]:
        for i in range(len(obj.opt_cloudy_name)):
            isNrwDataset = True
            if isNrwDataset == True:
                img = load_tiff_image(obj.opt_cloudy_path + obj.opt_cloudy_name[i]).astype('float32')
                bands_res = ['60m', '10m', '10m', '10m', '20m', '20m', '20m', '10m', 
                    '20m', '60m', '10m', '20m', '20m']
                dim = (10980, 10980)
                # ic(bands_res[i])
                if bands_res[i] == '20m' or bands_res[i] == '60m':
                    img = cv2.resize(img, dim, interpolation = cv2.INTER_NEAREST)
            else:
                img = load_tiff_image(obj.opt_cloudy_path + obj.opt_cloudy_name[i] + '.tif').astype('float32')

            if len(img.shape) == 2: img = img[np.newaxis, ...]
            if i:
                opt_cloudy = np.concatenate((opt_cloudy, img), axis=0)
            else:
                opt_cloudy = img
        del img
        opt_cloudy = opt_cloudy[[1, 2, 3, 4, 5, 6, 7, 8, 11, 12], :, :]
        opt_cloudy = opt_cloudy.transpose([1, 2, 0])
        if cut:
            opt_cloudy = opt_cloudy[obj.lims[0]:obj.lims[1], obj.lims[2]:obj.lims[3],:]
        opt_cloudy[np.isnan(opt_cloudy)] = np.nanmean(opt_cloudy)
    else:
        opt_cloudy = []
    
    if padding:
        # Add Padding to the images to match with the patch size
        data_dic["opt_cloudy_t" + str(prefix)] = np.pad(opt_cloudy, pad_tuple, mode = 'symmetric')
    else:
        data_dic["opt_cloudy_t" + str(prefix)] = opt_cloudy
    del opt_cloudy

    print('Dataset created!!')
    return train_patches, val_patches, test_patches, data_dic, sar_norm, opt_norm

def create_dataset_both_images(obj):
    
    obj.sar_name = obj.sar_name_t0
    obj.opt_name = obj.opt_name_t0
    obj.opt_cloudy_name = obj.opt_cloudy_name_t0
    train_patches_0, val_patches_0, test_patches_0, \
        data_dic_0, sar_norm_0, opt_norm_0 = create_dataset_coordinates(obj, prefix = 0)

    obj.sar_name = obj.sar_name_t1
    obj.opt_name = obj.opt_name_t1
    obj.opt_cloudy_name = obj.opt_cloudy_name_t1
    obj.args.patch_overlap += 0.03 # In case that t0 and t1 images are co-registered,
                                   # this allows to extract patches that are not co-registered between dates.
    train_patches_1, val_patches_1, test_patches_1, \
        data_dic_1, sar_norm_1, opt_norm_1 = create_dataset_coordinates(obj, prefix = 1)
    obj.args.patch_overlap -= 0.03

    train_patches = train_patches_0 + train_patches_1
    val_patches   = val_patches_0   + val_patches_1
    test_patches  = test_patches_0  + test_patches_1
    data_dic    = {**data_dic_0, **data_dic_1}

    sar_norm_0.min_val = np.minimum(sar_norm_0.min_val, sar_norm_1.min_val)
    sar_norm_0.max_val = np.maximum(sar_norm_0.max_val, sar_norm_1.max_val)
    sar_norm_0.clips[0] = np.minimum(sar_norm_0.clips[0], sar_norm_1.clips[0])
    sar_norm_0.clips[1] = np.maximum(sar_norm_0.clips[1], sar_norm_1.clips[1])

    opt_norm_0.min_val = np.minimum(opt_norm_0.min_val, opt_norm_1.min_val)
    opt_norm_0.max_val = np.maximum(opt_norm_0.max_val, opt_norm_1.max_val)
    opt_norm_0.clips[0] = np.minimum(opt_norm_0.clips[0], opt_norm_1.clips[0])
    opt_norm_0.clips[1] = np.maximum(opt_norm_0.clips[1], opt_norm_1.clips[1])

    return train_patches, val_patches, test_patches, data_dic, sar_norm_0, opt_norm_0



def db2linear(x):
    return 10.0**(x/10.0)

class Clip_Norm_sen12mscr():

    def __init__(self, feature_range=[-1, 1]):

        self.feature_range = feature_range

        # input data preprocessing parameters
        # self.clips_s1 = [np.array([-25.0, -32.5], dtype='float32'), 0.0]
        # self.clips_s1 = [db2linear(np.array([-25.0, -32.5], dtype='float32')), db2linear(0.0)]
        self.clips_s1 = [-np.inf, db2linear(np.array([0.0, -5.0], dtype='float32'))]
        self.clips_s2 = [0.0, 5000.0] 

        ############ VALUES CALCULATED FOR SUMMER SEASON ########
        # self.min_s1 = np.array([-56.6, -54.94], dtype='float32')
        # self.max_s1 = np.array([5.1, 0.0], dtype='float32')
        self.min_s1 = db2linear(np.array([-56.6, -54.94], dtype='float32'))
        self.max_s1 = db2linear(np.array([5.1, 0.0], dtype='float32'))

        # # FOR ALL BANDS [B01, B02, B03, B04, B05, B06, B07, B08, B08A, B09, B10, B11, B12]
        # self.min_s2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32')
        # self.max_s2 = np.array([16868., 22761., 22579., 25325., 26501., 28002., 28003., 28000., 28003., 14575., 5617., 28000., 28000.], dtype='float32')

        # FOR BANDS OF 10M AND 20M [B02, B03, B04, B05, B06, B07, B08, B08A, B11, B12]
        self.min_s2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32')
        self.max_s2 = np.array([22761., 22579., 25325., 26501., 28002., 28003., 28000., 28003., 28000., 28000.], dtype='float32')

        # # FOR BANDS OF 10M AND 20M [B02, B03, B04, B08]
        # self.min_s2 = np.array([0.0, 0.0, 0.0, 0.0], dtype='float32')
        # self.max_s2 = np.array([22761., 22579., 25325., 28000.], dtype='float32')
        ########################

        self.min_s1 = np.clip(self.min_s1, self.clips_s1[0], None)
        self.max_s1 = np.clip(self.max_s1, None, self.clips_s1[1])
        self.min_s2 = np.clip(self.min_s2, self.clips_s2[0], None)
        self.max_s2 = np.clip(self.max_s2, None, self.clips_s2[1])


    def Normalize(self, img, sensor):
        if sensor == "s1":
            return self.MinMaxNormalization(self.min_s1, self.max_s1, self.clips_s1, self.feature_range, img)
        elif sensor == "s2" or sensor == "s2_cloudy":
            return self.MinMaxNormalization(self.min_s2, self.max_s2, self.clips_s2, self.feature_range, img)

    def Denormalize(self, img, sensor):

        if sensor == "s1":
            return self.MinMaxDenormalization(self.min_s1, self.max_s1, self.clips_s1, self.feature_range, img)
        elif sensor == "s2" or sensor == "s2_cloudy":
            return self.MinMaxDenormalization(self.min_s2, self.max_s2, self.clips_s2, self.feature_range, img)

    def MinMaxNormalization(self, min_val, max_val, clips, feature_range, img):
        data_range = max_val - min_val
        scale = (feature_range[1] - feature_range[0]) / _handle_zeros_in_scale(data_range)
        min_ = feature_range[0] - min_val * scale
        
        img = np.clip(img, clips[0], clips[1])
        img *= scale
        img += min_

        return img

    def MinMaxDenormalization(self, min_val, max_val, clips, feature_range, img):
        data_range = max_val - min_val
        scale = (feature_range[1] - feature_range[0]) / _handle_zeros_in_scale(data_range)
        min_ = feature_range[0] - min_val * scale

        img -= min_
        img /= scale

        return img

def Transform(arr, b):

    sufix = ''

    if b == 1:
        arr = np.rot90(arr, k = 1)
        sufix = '_rot90'
    elif b == 2:
        arr = np.rot90(arr, k = 2)
        sufix = '_rot180'
    elif b == 3:
        arr = np.rot90(arr, k = 3)
        sufix = '_rot270'
    elif b == 4:
        arr = np.flipud(arr)
        sufix = '_flipud'
    elif b == 5:
        arr = np.fliplr(arr)
        sufix = '_fliplr'
    elif b == 6:
        if len(arr.shape) == 3:
            arr = np.transpose(arr, (1, 0, 2))
        elif len(arr.shape) == 2:
            arr = np.transpose(arr, (1, 0))
        sufix = '_transpose'
    elif b == 7:
        if len(arr.shape) == 3:
            arr = np.rot90(arr, k = 2)
            arr = np.transpose(arr, (1, 0, 2))
        elif len(arr.shape) == 2:
            arr = np.rot90(arr, k = 2)
            arr = np.transpose(arr, (1, 0))
        sufix = '_transverse'

    return arr, sufix

def Data_augmentation(s1, s2, s2_cloudy,
                      id_transform,
                      fine_size=256,
                      random_crop_transformation=False):

    if id_transform == -1:
        id_transform = np.random.randint(6)
        
    s1, _ = Transform(s1, id_transform)
    s2, _ = Transform(s2, id_transform)
    s2_cloudy, _ = Transform(s2_cloudy, id_transform)

    if random_crop_transformation and np.random.rand() > .5:
        dif_size = round(fine_size * 10/100)
        h1 = np.random.randint(dif_size + 1)
        w1 = np.random.randint(dif_size + 1)

        s1 = np.float32(resize(s1, ((dif_size+fine_size), (dif_size+fine_size)), preserve_range=True))
        s2 = np.float32(resize(s2, ((dif_size+fine_size), (dif_size+fine_size)), preserve_range=True))
        s2_cloudy = np.float32(resize(s2_cloudy, ((dif_size+fine_size), (dif_size+fine_size)), preserve_range=True))

        s1 = s1[h1:(h1+fine_size), w1:(w1+fine_size)]
        s2 = s2[h1:h1+fine_size, w1:w1+fine_size]
        s2_cloudy = s2_cloudy[h1:h1+fine_size, w1:w1+fine_size]

    return s1, s2, s2_cloudy

# def Data_augmentation_sar_landsat(sar_t0, opt_t0,
    #                   id_transform,
    #                   fine_size=256,
    #                   random_crop_transformation=False,
    #                   labels=False):

    # if id_transform == -1:
    #     id_transform = np.random.randint(6)
        
    # sar_t0, _ = Transform(sar_t0, id_transform)
    # opt_t0, _ = Transform(opt_t0, id_transform)
    # if labels is not False:
    #     labels, _ = Transform(labels, id_transform)

    # if random_crop_transformation and np.random.rand() > .5:
    #     dif_size = round(fine_size * 10/100)
    #     h1 = np.random.randint(dif_size + 1)
    #     w1 = np.random.randint(dif_size + 1)

    #     sar_t0 = np.float32(resize(sar_t0, (3*(dif_size+fine_size), 3*(dif_size+fine_size)), preserve_range=True))
    #     opt_t0 = np.float32(resize(opt_t0, ((dif_size+fine_size), (dif_size+fine_size)), preserve_range=True))

    #     sar_t0 = sar_t0[3*h1:3*(h1+fine_size), 3*w1:3*(w1+fine_size)]
    #     opt_t0 = opt_t0[h1:h1+fine_size, w1:w1+fine_size]
    #     if labels is not False:
    #         labels = np.float32(resize(labels, ((dif_size+fine_size), (dif_size+fine_size)), preserve_range=True, order=0))
    #         labels = labels[h1:h1+fine_size, w1:w1+fine_size]
    
    # return sar_t0, opt_t0, labels

def Take_patches(patch_list, idx, data_dic,
                 fine_size=256,
                 random_crop_transformation=False):
#    print(idx, len(patch_list))
    sar = data_dic['sar_t' + str(patch_list[idx][0])] \
                  [patch_list[idx][1]:patch_list[idx][1]+fine_size,
                   patch_list[idx][2]:patch_list[idx][2]+fine_size, :]
    opt = data_dic['opt_t' + str(patch_list[idx][0])] \
                  [patch_list[idx][1]:patch_list[idx][1]+fine_size,
                   patch_list[idx][2]:patch_list[idx][2]+fine_size, :]
    opt_cloudy = data_dic['opt_cloudy_t' + str(patch_list[idx][0])] \
                         [patch_list[idx][1]:patch_list[idx][1]+fine_size,
                          patch_list[idx][2]:patch_list[idx][2]+fine_size, :]

    sar, opt, opt_cloudy = Data_augmentation(sar, opt, opt_cloudy,
                                             patch_list[idx][3],
                                             fine_size=fine_size,
                                             random_crop_transformation=random_crop_transformation)
    
    return sar, opt, opt_cloudy

# def save_samples_multiresolution(self, patch_list, output_path, 
    #                              idx=6, epoch=0, real_flag = False):

    # sar, opt, opt_cloudy = Take_patches(patch_list, idx, data_dic = self.data_dic,
    #                                  fine_size=self.fine_size,
    #                                  random_crop_transformation=False)
    # # sar_t0 = patches[0][np.newaxis, ...]
    # # opt_t0_fake = self.sess.run(self.fake_opt_t0_sample,
    # #                             feed_dict={self.SAR: sar_t0})

    # opt_fake = self.sess.run(self.fake_opt_t0_sample,
    #                             feed_dict={self.SAR: sar[np.newaxis, ...],
    #                                        self.OPT_cloudy: opt_cloudy[np.newaxis, ...]})

    # # # # # VISUALYZING THE PATCHES # # # # # 

    # if self.norm_type == 'wise_frame_mean':
    #     scaler_opt = joblib.load('../datasets/' + self.args.dataset_name + '/Norm_params' + \
    #                              '/opt_' + self.opt_name + '_' + 'std' + '.pkl')
    # else:
    #     scaler_opt = joblib.load('../datasets/' + self.args.dataset_name + '/Norm_params' + \
    #                              '/opt_' + self.opt_name + '_' + self.norm_type + '.pkl')
    
    # opt_t0_fake = Denormalization(opt_t0_fake[0,:,:,:], scaler_opt)
    # opt_t0_fake = opt_t0_fake[:, :, [3, 2, 1]] / (2**16 - 1)
    # opt_t0_fake[:, :, 0] = exposure.equalize_adapthist(opt_t0_fake[:, :, 0], clip_limit=0.02)
    # opt_t0_fake[:, :, 1] = exposure.equalize_adapthist(opt_t0_fake[:, :, 1], clip_limit=0.02)
    # opt_t0_fake[:, :, 2] = exposure.equalize_adapthist(opt_t0_fake[:, :, 2], clip_limit=0.02)
    # opt_t0_fake = Image.fromarray(np.uint8(opt_t0_fake*255))
    # opt_t0_fake.save(output_path + '/' + str(patch_list[idx][0]) + '_opt_fake_' + str(epoch) + '.tiff')

    # if labels is not False:

    #     scaler_sar = joblib.load('../datasets/' + self.args.dataset_name + '/Norm_params' + \
    #                              '/sar_' + self.sar_name + '_' + self.norm_type + '.pkl')
    #     sar_t0 = Denormalization(sar_t0[0,:,:,:], scaler_sar)
    #     sar_vh = exposure.equalize_adapthist(sar_t0[:,:,0], clip_limit=0.02)
    #     sar_vh = Image.fromarray(np.uint8(sar_vh*255))
    #     sar_vh.save(output_path + '/' + str(patch_list[idx][0]) + '_sar_vh.tiff')
    #     sar_vv = exposure.equalize_adapthist(sar_t0[:,:,1], clip_limit=0.02)
    #     sar_vv = Image.fromarray(np.uint8(sar_vv*255))
    #     sar_vv.save(output_path + '/' + str(patch_list[idx][0]) + '_sar_vv.tiff')

    #     opt_t0 = patches[1]
    #     opt_t0 = Denormalization(opt_t0, scaler_opt)
    #     opt_t0 = opt_t0[:, :, [3, 2, 1]] / (2**16 - 1)
    #     opt_t0[:, :, 0] = exposure.equalize_adapthist(opt_t0[:, :, 0], clip_limit=0.02)
    #     opt_t0[:, :, 1] = exposure.equalize_adapthist(opt_t0[:, :, 1], clip_limit=0.02)
    #     opt_t0[:, :, 2] = exposure.equalize_adapthist(opt_t0[:, :, 2], clip_limit=0.02)
    #     opt_t0 = Image.fromarray(np.uint8(opt_t0*255))
    #     opt_t0.save(output_path + '/' + str(patch_list[idx][0]) + '_opt_real.tiff')

    #     labels = patches[2]
    #     labels = Image.fromarray(np.uint8(labels*255))
    #     labels.save(output_path + '/' + str(patch_list[idx][0]) + '_labels.tiff')







# ============= SEN12MSCR dataset =============
def Load_SEN2MSCR(data_path, train_ROIs = None, val_ROIs = None, test_ROIs = None, custom_sets = True):

    def Patches_ids(season, sid, d):
        patches = sen12mscr.get_patch_ids(season, sid)
        for patch in patches:
            data_files.append([season, sid, patch])
            set_ids.append(d) 

    sen12mscr = SEN12MSCRDataset(data_path)
    data_files, set_ids = [], []

    if custom_sets:
        for i in train_ROIs:
            Patches_ids(i[0], i[1], 0)
        for i in val_ROIs:
            Patches_ids(i[0], i[1], 1)
        for i in test_ROIs:
            Patches_ids(i[0], i[1], 2)
    
    else:
        seasons = [Seasons.SUMMER]
        for season in seasons:

            scene_ids = sen12mscr.get_scene_ids(season)
            scene_ids = list(np.sort(list(scene_ids)))
            
            for i, sid in zip(range(len(scene_ids)), scene_ids):

                if i < len(scene_ids) * .07: d = 1     # Validation ROIs   (10% of the training set)
                elif i < len(scene_ids) * .7: d = 0    # Train ROIs        (70% of the dataset)
                else: d = 2                            # Test ROIs         (30% of the dataset)

                Patches_ids(season, sid, d)
    
    set_ids = np.array(set_ids)
    train_patches = np.argwhere(set_ids == 0).reshape(-1)
    val_patches = np.argwhere(set_ids == 1).reshape(-1)
    test_patches = np.argwhere(set_ids == 2).reshape(-1)
    print("Training patches: ", train_patches.shape[0])
    print("Validation patches: ", val_patches.shape[0])
    print("Test patches: ", test_patches.shape[0])
    
    return train_patches, val_patches, test_patches, data_files

def Take_Patches_sen12mscr(data_path, data_files, idx, normalize,
                           fine_size, id_transform=-1, random_crop_transformation=False):

    # Loading
    sen12mscr = SEN12MSCRDataset(data_path)
    s1, s2, s2_cloudy, _ = sen12mscr.get_s1s2s2cloudy_triplet(data_files[idx][0], data_files[idx][1], data_files[idx][2], 
                                                             s1_bands=S1Bands.ALL, s2_bands=S2Bands.ONLY_10_20_M,
                                                             s2cloudy_bands=S2Bands.ONLY_10_20_M)
    
    
    s1 = s1.transpose([1, 2, 0]).astype('float32')
    s2 = s2.transpose([1, 2, 0]).astype('float32')
    s2_cloudy = s2_cloudy.transpose([1, 2, 0]).astype('float32')

    s1[np.isnan(s1)] = np.nanmean(s1)
    s2[np.isnan(s2)] = np.nanmean(s2)
    s2_cloudy[np.isnan(s2_cloudy)] = np.nanmean(s2_cloudy)

    s1 = db2linear(s1)                          # convert from dB to linear
    # # Normalization
    s1 = normalize(s1, "s1")
    s2 = normalize(s2, "s2")
    s2_cloudy = normalize(s2_cloudy, "s2")

    # Data Augmentation
    s1, s2, s2_cloudy = Data_augmentation(s1, s2, s2_cloudy, id_transform, fine_size=fine_size,
                            random_crop_transformation=random_crop_transformation)
    
    return s1, s2, s2_cloudy

def Load_Patches_parallel(data_path, data_files, data, normalize, fine_size, im):

    images = Take_Patches_sen12mscr(data_path, data_files, data[im], 
                                    normalize, fine_size, id_transform=-1, 
                                    random_crop_transformation=True)

    return images



def plot_hist(img, bins, lim, name, output_path):
    hist , bins  = np.histogram(img, bins=bins)
    plt.figure(figsize=(6, 4))
    plt.plot(bins[1:], hist/np.prod(img.shape))
    plt.title("{}".format(name))
    plt.xlabel('bins')
    plt.ylabel('count')
    # plt.xlim(lim)
    # plt.ylim([0, 0.011])
    plt.tight_layout()
    plt.savefig(output_path + "/" + name + ".png", dpi=500)
    plt.close()

def save_image(image, file_path, sensor = "s2"):

    if sensor == "s2":        
        image[:,:,0] = exposure.equalize_adapthist(image[:,:,0] , clip_limit=0.01)
        image[:,:,1] = exposure.equalize_adapthist(image[:,:,1] , clip_limit=0.01)
        image[:,:,2] = exposure.equalize_adapthist(image[:,:,2] , clip_limit=0.01)
        image *= 255
        image = Image.fromarray(np.uint8(image))
        image.save(file_path + ".tif")
    
    elif sensor == "s1":
        image = exposure.equalize_adapthist(image[:,:,1] , clip_limit=0.009)
        image = Image.fromarray(np.uint8(image*255))
        image.save(file_path + ".tif")

def generate_samples_sen12mscr(self, output_path, idx, patch_list=None, epoch=0, real_flag = False):

    if patch_list is None:
        s1, s2, s2_cloudy = Take_Patches_sen12mscr(self.data_path, self.data_files, 
                                                    idx, self.norm_routine.Normalize,
                                                    self.image_size, id_transform=0,
                                                    random_crop_transformation=False)
        file_name_opt = "{}_{}_{}_p{}".format(self.data_files[idx][0], "s2", self.data_files[idx][1], self.data_files[idx][2])
        file_name_sar = "{}_{}_{}_p{}".format(self.data_files[idx][0], "s1", self.data_files[idx][1], self.data_files[idx][2])
    else:
        s1, s2, s2_cloudy = Take_patches(patch_list, idx, data_dic = self.data_dic,
                                         fine_size=self.image_size,
                                         random_crop_transformation=False)
        file_name_opt = ""
        file_name_sar = ""
    
    s2_fake = self.sess.run(self.fake_opt_t0_sample,
                            feed_dict={self.SAR: s1[np.newaxis, ...],
                                       self.OPT_cloudy: s2_cloudy[np.newaxis, ...]})
    plot_hist(s2_fake, 1000, [-1, 1], "_histogram FAKE S2 {}".format(epoch), output_path)

    s2_fake = self.norm_routine.Denormalize(s2_fake[0,:,:,:], "s2")
    s2_fake = np.clip(s2_fake, 0.0, self.norm_routine.clips_s2[1]) / self.norm_routine.clips_s2[1]
    s2_fake = s2_fake[:, :, self.visible_bands] 
    file_name = "/{}_{}_{}".format("FAKE", file_name_opt, epoch)
    save_image(s2_fake, output_path + file_name, sensor = "s2")

    if real_flag:

        plot_hist(s2, 1000, [-1, 1], "_histogram REAL S2", output_path)
        s2 = self.norm_routine.Denormalize(s2, "s2") / self.norm_routine.clips_s2[1]
        s2 = s2[:, :, self.visible_bands] 
        file_name = "/{}_{}".format("REAL", file_name_opt)
        save_image(s2, output_path + file_name, sensor = "s2")

        s2_cloudy = self.norm_routine.Denormalize(s2_cloudy, "s2") / self.norm_routine.clips_s2[1]
        s2_cloudy = s2_cloudy[:, :, self.visible_bands] 
        file_name = "/{}_{}".format("REAL_cloudy", file_name_opt)
        save_image(s2_cloudy, output_path + file_name, sensor = "s2")

        s1 = self.norm_routine.Denormalize(s1, "s1")
        s1 = (s1 - self.norm_routine.min_s1) / (self.norm_routine.max_s1 - self.norm_routine.min_s1)
        file_name = "/{}_{}".format("SAR", file_name_sar)
        save_image(s1, output_path + file_name, sensor = "s1")

def generate_samples(self, output_path, idx, patch_list=None, epoch=0, real_flag = False):

    s1, s2, s2_cloudy = Take_patches(patch_list, idx, data_dic = self.data_dic,
                                        fine_size=self.image_size,
                                        random_crop_transformation=False)
    file_name_opt = ""
    file_name_sar = ""

    s2_fake = self.sess.run(self.fake_opt_t0_sample,
                            feed_dict={self.SAR: s1[np.newaxis, ...],
                                       self.OPT_cloudy: s2_cloudy[np.newaxis, ...]})
    plot_hist(s2_fake[0,:,:,:], 1000, [-1, 1], "_histogram FAKE S2 {}".format(epoch), output_path)
    
    s2_fake = self.opt_norm.Denormalize(s2_fake[0,:,:,:]) / self.opt_norm.max_val.max()
    s2_fake = s2_fake[:, :, self.visible_bands] 
    file_name = "/{}_{}_{}".format("FAKE", file_name_opt, epoch)
    save_image(s2_fake, output_path + file_name, sensor = "s2")

    if real_flag:

        plot_hist(s2, 1000, [-1, 1], "_histogram REAL S2", output_path)
        s2 = self.opt_norm.Denormalize(s2) / self.opt_norm.max_val.max()
        s2 = s2[:, :, self.visible_bands] 
        file_name = "/{}_{}".format("REAL", file_name_opt)
        save_image(s2, output_path + file_name, sensor = "s2")

        s2_cloudy = self.opt_norm.Denormalize(s2_cloudy) / self.opt_norm.max_val.max()
        s2_cloudy = s2_cloudy[:, :, self.visible_bands] 
        file_name = "/{}_{}".format("REAL_cloudy", file_name_opt)
        save_image(s2_cloudy, output_path + file_name, sensor = "s2")

        s1 = self.sar_norm.Denormalize(s1) / self.sar_norm.max_val.max()
        file_name = "/{}_{}".format("SAR", file_name_sar)
        save_image(s1, output_path + file_name, sensor = "s1")




class Image_reconstruction(object):

    def __init__ (self, inputs, tensor, output_c_dim, patch_size=256, overlap_percent=0):

        self.inputs = inputs
        self.patch_size = patch_size
        self.overlap_percent = overlap_percent
        self.output_c_dim = output_c_dim
        self.tensor = tensor
    
    def Inference(self, tile):
        
        '''
        Normalize before calling this method
        '''

        num_rows, num_cols, _ = tile.shape

        # Percent of overlap between consecutive patches.
        # The overlap will be multiple of 2
        overlap = round(self.patch_size * self.overlap_percent)
        overlap -= overlap % 2
        stride = self.patch_size - overlap
        
        # Add Padding to the image to match with the patch size and the overlap
        step_row = (stride - num_rows % stride) % stride
        step_col = (stride - num_cols % stride) % stride
 
        pad_tuple = ( (overlap//2, overlap//2 + step_row), ((overlap//2, overlap//2 + step_col)), (0,0) )
        tile_pad = np.pad(tile, pad_tuple, mode = 'symmetric')

        # Number of patches: k1xk2
        k1, k2 = (num_rows+step_row)//stride, (num_cols+step_col)//stride
        print('Number of patches: %d x %d' %(k1, k2))

        # Inference
        probs = np.zeros((k1*stride, k2*stride, self.output_c_dim), dtype='float32')

        for i in range(k1):
            for j in range(k2):
                
                patch = tile_pad[i*stride:(i*stride + self.patch_size), j*stride:(j*stride + self.patch_size), :]
                patch = patch[np.newaxis,...]
                # infer = self.sess.run(self.tensor, feed_dict={inputs: patch})
                if patch.shape[3] > 2:
                    fd = dict(zip(self.inputs, [patch[:,:,:,:2], patch[:,:,:,2:]]))
                else:
                    fd = dict(zip(self.inputs, patch))
                infer = self.tensor.eval(feed_dict=fd)

                probs[i*stride : i*stride+stride, 
                      j*stride : j*stride+stride, :] = infer[0, overlap//2 : overlap//2 + stride, 
                                                                overlap//2 : overlap//2 + stride, :]
            print('row %d' %(i+1))

        # Taken off the padding
        probs = probs[:k1*stride-step_row, :k2*stride-step_col]

        return probs


# ============= METRICS =============
def MAE(y_true, y_pred):
    """Computes the MAE over the full image."""
    return np.mean(np.abs(y_pred - y_true))
    # return K.mean(K.abs(y_pred[:, 0:13, :, :] - y_true[:, 0:13, :, :]))

def MSE(y_true, y_pred):
    """Computes the MSE over the full image."""
    return np.mean(np.square(y_pred - y_true))
    # return K.mean(K.square(y_pred[:, 0:13, :, :] - y_true[:, 0:13, :, :]))

def RMSE(y_true, y_pred):
    """Computes the RMSE over the full image."""
    return np.sqrt(np.mean(np.square(y_pred - y_true)))
    # return K.sqrt(K.mean(K.square(y_pred[:, 0:13, :, :] - y_true[:, 0:13, :, :])))

def SAM(y_true, y_pred):
    """Computes the SAM over the full image."""    
    mat = np.sum(y_true * y_pred, axis=-1)
    mat /= np.sqrt(np.sum(y_true * y_true, axis=-1))
    mat /= np.sqrt(np.sum(y_pred * y_pred, axis=-1))
    mat = np.arccos(np.clip(mat, -1, 1))

    return np.mean(mat)

def PSNR(y_true, y_pred):
    """Computes the PSNR over the full image."""
    # y_true *= 2000
    # y_pred *= 2000
    # rmse = K.sqrt(K.mean(K.square(y_pred[:, 0:13, :, :] - y_true[:, 0:13, :, :])))
    # return 20.0 * (K.log(10000.0 / rmse) / K.log(10.0))
    rmse = RMSE(y_true, y_pred)
    return 20.0 * np.log10(10000.0 / rmse)

def SSIM(y_true, y_pred):
    """Computes the SSIM over the full image."""
    y_true = np.clip(y_true, 0, 10000.0)
    y_pred = np.clip(y_pred, 0, 10000.0)
    ssim = structural_similarity(y_true, y_pred, data_range=10000.0, multichannel=True)

    return ssim

def METRICS(y_true, y_pred, mask=None, ssim_flag=False, dataset="Para_10m"):

    if ssim_flag:
        if dataset == 'Santarem' or dataset == 'Santarem_I1' or dataset == 'Santarem_I2' or dataset == 'Santarem_I3' or dataset == 'Santarem_I4' or dataset == 'Santarem_I5':            
            no_tiles_h = 5
            no_tiles_w = 5
        else:
            no_tiles_h = 5
            no_tiles_w = 4
    
        rows, cols, _ = y_true.shape 
        xsz = rows // no_tiles_h
        ysz = cols // no_tiles_w

        # Tiles coordinates
        h = np.arange(0, rows, xsz)
        w = np.arange(0, cols, ysz)
        if (rows % no_tiles_h): h = h[:-1]
        if (cols % no_tiles_w): w = w[:-1]
        tiles = list(itertools.product(h, w))
        
        if mask is not None:

            if dataset == "Para_10m":
                test_tiles = [1, 3, 6, 8, 9, 11, 14, 15, 16, 17]
            elif dataset == "MG_10m":
                test_tiles = [0, 2, 6, 7, 8, 9, 11, 13, 14, 16]
            elif dataset == "Santarem" or dataset == 'Santarem_I1' or dataset == 'Santarem_I2' or dataset == 'Santarem_I3' or dataset == 'Santarem_I4' or dataset == 'Santarem_I5':
                test_tiles = [0,2,6,8,12,13,15,16,19,21,22,23]

            ssim = []
            for i in test_tiles:
                print(i)
                img1 = y_true[tiles[i][0]:tiles[i][0]+xsz, tiles[i][1]:tiles[i][1]+ysz, :]
                img2 = y_pred[tiles[i][0]:tiles[i][0]+xsz, tiles[i][1]:tiles[i][1]+ysz, :]
                ssim.append(SSIM(img1, img2))

        else:
            # Calculate ssim for the whole image
            test_tiles = range(len(tiles))
            ssim = []
            for i in test_tiles:
                print(i)
                img1 = y_true[tiles[i][0]:tiles[i][0]+xsz, tiles[i][1]:tiles[i][1]+ysz, :]
                img2 = y_pred[tiles[i][0]:tiles[i][0]+xsz, tiles[i][1]:tiles[i][1]+ysz, :]
                ssim.append(SSIM(img1, img2))
    else:
        ssim = [-1.0]
    ssim = np.asarray(ssim)

    if mask is not None:
        y_true = y_true[mask==1]
        y_pred = y_pred[mask==1]

    psnr = PSNR(y_true, y_pred)
    y_true /= 2000
    y_pred /= 2000
    mae  = MAE (y_true, y_pred)
    mse  = MSE (y_true, y_pred)
    rmse = RMSE(y_true, y_pred)
    sam  = SAM (y_true, y_pred)

    # ----------------
    y_true *= 2000
    y_pred *= 2000
    # ----------------

    return mae, mse, rmse, psnr, sam, ssim
    

def Write_metrics_on_file(f, title, mae, mse, rmse, psnr, sam, ssim):

    print("__________ {} __________\n".format(title))
    print("mae, mse, rmse, psnr, sam, ssim")
    print(mae, mse, rmse, psnr, sam, ssim.mean())

    f.write("__________ {} __________\n".format(title))
    f.write("MAE  = %.4f\n"%( mae))
    f.write("MSE  = %.4f\n"%( mse))
    f.write("RMSE = %.4f\n"%(rmse))
    f.write("PSNR = %.4f\n"%(psnr))
    f.write("SAM  = %.4f\n"%( sam))
    f.write("SSIM = %.4f\n"%(ssim.mean()))
    f.write("\n")