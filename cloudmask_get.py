import numpy as np
import scipy
import scipy.signal as scisig
from icecream import ic
import pdb
import matplotlib.pyplot as plt
from osgeo import gdal

# naming conventions:
# ['QA60', 'B1','B2',    'B3',    'B4',   'B5','B6','B7', 'B8','  B8A', 'B9',          'B10', 'B11','B12']
# ['QA60','cb', 'blue', 'green', 'red', 're1','re2','re3','nir', 'nir2', 'waterVapor', 'cirrus','swir1', 'swir2'])
# [        1,    2,      3,       4,     5,    6,    7,    8,     9,      10,            11,      12,     13]) #gdal
# [        0,    1,      2,       3,     4,    5,    6,    7,     8,      9,            10,      11,     12]) #numpy
# [              BB      BG       BR                       BNIR                                  BSWIR1    BSWIR2

# ge. Bands 1, 2, 3, 8, 11, and 12 were utilized as BB , BG , BR , BNIR , BSWIR1 , and BSWIR2, respectively.

def get_rescaled_data(data, limits):
    return (data - limits[0]) / (limits[1] - limits[0])


def get_normalized_difference(channel1, channel2):
    subchan = channel1 - channel2
    sumchan = channel1 + channel2
    sumchan[sumchan == 0] = 0.001  # checking for 0 divisions
    return subchan / sumchan


def get_shadow_mask(data_image):
    # get data between 0 and 1
    data_image = data_image / 10000.

    (ch, r, c) = data_image.shape
    shadow_mask = np.zeros((r, c)).astype('float32')

    BB = data_image[1]
    BNIR = data_image[7]
    BSWIR1 = data_image[11]

    CSI = (BNIR + BSWIR1) / 2.

    t3 = 3 / 4
    T3 = np.min(CSI) + t3 * (np.mean(CSI) - np.min(CSI))

    t4 = 5 / 6
    T4 = np.min(BB) + t4 * (np.mean(BB) - np.min(BB))

    shadow_tf = np.logical_and(CSI < T3, BB < T4)

    shadow_mask[shadow_tf] = -1
    shadow_mask = scisig.medfilt2d(shadow_mask, 5)

    return shadow_mask


def get_cloud_mask(data_image, cloud_threshold, binarize=False, use_moist_check=False):
    data_image = data_image / 10000.
    (ch, r, c) = data_image.shape

    # Cloud until proven otherwise
    score = np.ones((r, c)).astype('float32')
    # Clouds are reasonably bright in the blue and aerosol/cirrus bands.
    score = np.minimum(score, get_rescaled_data(data_image[1], [0.1, 0.5]))
    score = np.minimum(score, get_rescaled_data(data_image[0], [0.1, 0.3]))
    score = np.minimum(score, get_rescaled_data((data_image[0] + data_image[10]), [0.15, 0.2]))
    # Clouds are reasonably bright in all visible bands.
    score = np.minimum(score, get_rescaled_data((data_image[3] + data_image[2] + data_image[1]), [0.2, 0.8]))

    if use_moist_check:
        # Clouds are moist
        ndmi = get_normalized_difference(data_image[7], data_image[11])
        score = np.minimum(score, get_rescaled_data(ndmi, [-0.1, 0.1]))

    # However, clouds are not snow.
    ndsi = get_normalized_difference(data_image[2], data_image[11])
    score = np.minimum(score, get_rescaled_data(ndsi, [0.8, 0.6]))

    box_size = 7
    box = np.ones((box_size, box_size)) / (box_size ** 2)
    score = scipy.ndimage.morphology.grey_closing(score, size=(5, 5))
    score = scisig.convolve2d(score, box, mode='same')

    score = np.clip(score, 0.00001, 1.0)

    if binarize:
        score[score >= cloud_threshold] = 1
        score[score < cloud_threshold] = 0

    return score


def get_cloud_cloudshadow_mask(data_image, cloud_threshold = 0.2):
    cloud_mask = get_cloud_mask(data_image, cloud_threshold, binarize=True)
    shadow_mask = get_shadow_mask(data_image)

    cloud_cloudshadow_mask = np.zeros_like(cloud_mask)
    cloud_cloudshadow_mask[shadow_mask < 0] = -1
    cloud_cloudshadow_mask[cloud_mask > 0] = 1
    
    #pdb.set_trace()
    return cloud_cloudshadow_mask

def load_tiff_image(path):
    # Read tiff Image
    print (path) 
    gdal_header = gdal.Open(path)
    img = gdal_header.ReadAsArray()
    return img

if __name__ == '__main__':

    dataset_name = 'Santarem'
    datasets_dir = 'E:/Jorge/dataset/'
    if dataset_name == 'Santarem':
        
        opt_path = datasets_dir + dataset_name + '/S2/'
        opt_cloudy_path = datasets_dir + dataset_name + '/S2_cloudy/'
        labels_path = datasets_dir + dataset_name


        opt_name_t0 = ['2020/S2_R5_ST_2020_08_09_B1_B7',
                            '2020/S2_R5_ST_2020_08_09_B8_B12']
        opt_cloudy_name_t0 = ['2020/S2_CL_R5_ST_2020_08_24_B1_B7',
                                    '2020/S2_CL_R5_ST_2020_08_24_B8_B12']

        opt_name_t1 = ['2021/S2_R5_ST_2021_07_25_B1_B7',
                            '2021/S2_R5_ST_2021_07_25_B8_B12']
        opt_cloudy_name_t1 = ['2021/S2_CL_R5_ST_2021_07_30_B1_B7',
                                    '2021/S2_CL_R5_ST_2021_07_30_B8_B12']

    t = 't1'
    if t == 't0':
        opt_name = opt_name_t0
        opt_cloudy_name = opt_cloudy_name_t0
    else:
        opt_name = opt_name_t1
        opt_cloudy_name = opt_cloudy_name_t1

    cloudy_flag = False
    if cloudy_flag == False:
        name_id = ''
        for i in range(len(opt_name)):
            ic(opt_path + opt_name[i] + '.tif')        
            img = load_tiff_image(opt_path + opt_name[i] + '.tif').astype('float32')
            if len(img.shape) == 2: img = img[np.newaxis, ...]
            if i:
                s2 = np.concatenate((s2, img), axis=0)
            else:
                s2 = img
        del img       
    else:
        name_id = 'cloudy'
        for i in range(len(opt_cloudy_name)):
            img = load_tiff_image(opt_cloudy_path + opt_cloudy_name[i] + '.tif').astype('float32')
            if len(img.shape) == 2: img = img[np.newaxis, ...]
            if i:
                s2 = np.concatenate((s2, img), axis=0)
            else:
                s2 = img
        del img 
    # path = ""
    # date = "2019"

    # filename = path + 's2_'+date+'.npy'
    # # filename = path + 's2_cloudy_'+date+'.npy'

    # s2 = np.load(filename)

    cloud_cloudshadow_mask = get_cloud_cloudshadow_mask(s2, cloud_threshold = 0.2).astype(np.int8)
    # print("cloud_cloudshadow_mask.shape: ", cloud_cloudshadow_mask.shape)
    ic(np.unique(cloud_cloudshadow_mask, return_counts = True))
    np.save("cloudmask_"+name_id+"_"+t+".npy", cloud_cloudshadow_mask)

    plt.figure()
    plt.imshow(cloud_cloudshadow_mask)
    plt.show()

    plt.figure()
    plt.imshow(cloud_cloudshadow_mask)
    plt.axis('off')
    plt.savefig('cloudmask_'+name_id+'_'+t+'.png', dpi = 500)