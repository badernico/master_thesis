import numpy as np
import matplotlib.pyplot as plt
import haci as hci
from skimage.morphology import dilation
from skimage.measure import regionprops
import pandas as pd

fname = '/home/deneke/src/radolan/data/hdfd_miub_drnet00_l3_dbz_v00_20130701000000.nc'

# 0. read data
rx       = hci.read_rx_hdcp2(fname)

# 1. get mask of convectively active grid cells by thresholding
camask   = hci.ca_mask(rx.dbz,thresh=35.0)

# 2. label connected regions above 35dBz
selem = np.zeros((3,3,3))
selem[1,:,:]=1
objmap, nr_obj =  hci.label_objects(camask,selem)

# 3a. get mask of invalid data ...
fillmask = (rx.dbz.values==250)
# 3b. ... and build buffer mask around existing convection/borders
bufmask = hci.buffer_mask(camask|fillmask,radius=15)

# 4a. get mask for new/CI grid cells ...
cimask = hci.ci_mask(camask,bufmask)
# 4b. ... and extract their associated object labels
ci_labels = np.unique(objmap[cimask])

# 5. link CI objects with their subsequent children
objmap,_ = hci.link_objects(objmap,ci_labels,5)

# 6. extract object properties
objprops = hci.object_props(rx.time,objmap,cimask)
