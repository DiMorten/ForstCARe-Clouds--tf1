import numpy as np
from osgeo import gdal
from icecream import ic
import pdb
def load_tiff_image(path):
    # Read tiff Image
    print (path) 
    gdal_header = gdal.Open(path)
    img = gdal_header.ReadAsArray()
    return img
im1 = load_tiff_image('MG_10m/S2_t0_10bands_Fake_.tif')
im2 = load_tiff_image('MG_10m_dropoutTrue/S2_t1_10bands_Fake_.tif')

assert np.all(im1 == im2)

