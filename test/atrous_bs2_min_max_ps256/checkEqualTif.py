import numpy as np
from osgeo import gdal
from icecream import ic
import pdb
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
def load_tiff_image(path):
    # Read tiff Image
    print (path) 
    gdal_header = gdal.Open(path)
    img = gdal_header.ReadAsArray()
    return img

im1_name = 'MG_10m_dropoutTrue/S2_t0_10bands_Fake_.tif'
im2_name = 'MG_10m_dropoutTrue2/S2_t0_10bands_Fake_.tif'

im1_name = 'MG_10m_dropoutFalseOk/S2_t0_10bands_Fake_.tif'
im2_name = 'MG_10m_dropoutFalseOk2/S2_t0_10bands_Fake_.tif'

im1 = load_tiff_image(im1_name)
im2 = load_tiff_image(im2_name)

ic(np.average(im1), np.std(im1), im1.shape)
ic(np.average(im2), np.std(im2), im2.shape)
ic(np.average(im1 - im2), np.std(im1 - im2))

low_limit = 4000
high_limit = 7000
rgb_chans = [2, 1, 0]

error = mean_squared_error(im1[rgb_chans, low_limit:high_limit, low_limit:high_limit].flatten(), 
        im2[rgb_chans, low_limit:high_limit, low_limit:high_limit].flatten(), squared=False)
ic(error)

dif = np.abs(im1 - im2)
dif = np.transpose(dif, (1, 2, 0))
ic(np.average(dif), np.std(dif))


dif = dif[..., rgb_chans]

def minMaxNorm(x):
    return np.std(x) * (np.max(x) - np.min(x)) + np.min(x)
for chan in range(dif.shape[-1]):
    dif[..., chan] = dif[..., chan] - np.min(dif[..., chan])
    dif[..., chan] = dif[..., chan] / np.max(dif[..., chan])

ic(np.average(dif), np.std(dif))


plt.imshow(dif[low_limit:high_limit, low_limit:high_limit], vmin = 0, vmax = 1)
plt.show()
pdb.set_trace()


# assert np.all(im1 == im2)

