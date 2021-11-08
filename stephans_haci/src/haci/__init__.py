#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This file is part of the Haeberlie-Algorithm for CI detection package
# (HACI) developed within the satellite group at TROPOS.
#
# HACI is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# Authors/Contact
# -Stephan Lenk (lenk@tropos.de)
# -Hartwig Deneke (deneke@tropos.de)

import numpy as np
import xarray as xr
import pandas as pd
import operator
import sys,os
from skimage.morphology import disk as selem_disk, dilation
from skimage.measure import regionprops as regionprops
from scipy.ndimage import label as ndi_label
from tqdm import tqdm

_pkg_dir = os.path.dirname( sys.modules[__name__].__file__ )
_data_dir = os.path.join(_pkg_dir,'share')

#def read_rx_hdcp2(fname,slice=None):
#    '''
#    read RADOLAN RX data in HDCP2 format
#
#    Inputs:
#        * fname: string
#            file name
#        * slice: Slice
#            time slice of which time frame to load
#    '''
#    # read dataset
#    rx = xr.open_dataset(fname,mask_and_scale=False)#,autoclose=True)
#    # extract time slice 
#    if slice is not None:
#        rx = rx.isel(time=slice)
#    # create deep copy of dataset so it can be modified
#    rx = rx.copy(deep=True)
#    #  fixup valid_max and add _FillValue
#    rx.dbz['_FillValue'] = 250
#    rx.dbz['valid_max'] = 249
#    return rx

#def write_table(f, df, fmt, header=None):
#    # check if f is file object
#    if not hasattr(f,'write'):
#        f = open(f,'w')
#        close_file = True
#    else:
#        close_file = False
#    if header is not None:
#        f.write(header+'\n')
#    for r in df.itertuples():
#        f.write(fmt.format(**r._asdict())+'\n')
#    if close_file:
#        f.close()
#    return

def threshold(arr, thresh, op=operator.ge):
    if hasattr(arr,'scale_factor') and hasattr(arr,'add_offset'):
        a = getattr(arr,'scale_factor')
        b = getattr(arr,'add_offset')
        thresh = (thresh-b)/a
    return op(arr,thresh)

def ca_mask(arr, thresh=35.0, fillValue=250):
    # threshold radar reflectivity
    ca_mask = threshold(arr,thresh)
    # mask out fill values
    ca_mask.values[arr.values==fillValue]=False
    return ca_mask

def buffer_mask(mask,radius):
    selem = selem_disk(radius)
    r = selem.shape[0]//2
    ndim = len(mask.shape)
    if ndim==3:
        if radius>5:
            from scipy.signal import fftconvolve
            buf_mask = np.ones_like(mask)
            for i in np.arange(mask.shape[0]):
                buf_mask[i,r:-r,r:-r] = fftconvolve(mask[i,:,:],selem,mode='valid')>0.1
        else:
            from scipy.ndimage import binary_dilation
            buf_mask = binary_dilation(mask,selem[None,:,:],mode='same',boundary_value=1)>0.1
    elif ndim==2:
        if radius>5:
            from scipy.signal import fftconvolve
            buf_mask = np.ones_like(mask)
            buf_mask[r:-r,r,-r] = fftconvolve(mask,selem,mode='same')>0.1
        else:
            from scipy.ndimage import binary_dilation
            buf_mask = binary_dilation(mask,selem,mode='same',boundary_value=1)>0.1
    return buf_mask

def ci_mask(camask,bufmask):
    '''
    Calculate 
    '''
    # initialize mask with zeros
    cimask = np.zeros_like(camask)
    cimask[1:,:,:] = camask[1:,:,:]&~bufmask[0:-1,:,:]
    return cimask

def overlapping_objects(obj1,obj2):
    m = (obj1>0)&(obj2>0)
    id1 = obj1[m]
    id2 = obj2[m]
    idlink = np.stack((id1,id2),axis=-1)
    idlink = np.sort(np.unique(idlink,axis=0),axis=0)
    return (idlink[:,0],idlink[:,1])

def label_objects(objmask,selem=None):
    '''
    Label objects
    '''
    if selem is None:
        selem = np.ones((3,3,3))
    return ndi_label(objmask,selem)

def filter_objects(objmap,labels,invert=False):
    om = np.zeros(np.max(objmap)+1,dtype=np.int)
    if not invert:
        om[labels] = labels
    else:
        om[:] = np.arange(len(om))
        om[labels] = 0
    return om[objmap]

def link_objects(objmap,labels,nstep=5):

    obj_labels = labels.copy()
    
    for i in tqdm(range(nstep)):
        try:
            #print(i)
            # set non-CI objects to zero
            omap = filter_objects(objmap,labels)
            # grow by dilation
            selem = np.ones((3,3))
            omap = dilation(omap,selem[None,:,:])
            # find overlapping objects for times t0 and t1
            label_t0,label_t1 = overlapping_objects(omap[0:-1,:,:],objmap[1:,:,:])
            obj_labels = np.append(obj_labels,label_t1)
            labels = label_t1
        except Exception as e:
            print("Error while linking objects: {}. There are no longer object paths.".format(e))
            break
    # get mask for linked objects
    omask = filter_objects(objmap,obj_labels)>0
    # return newly labeled mask
    return label_objects(omask)

def link_objects_fast(objmap,cimask):
   
    # dilation of object mask to enhace overlap
    selem = np.ones((3,3))
    object_mask_dilated = dilation(np.ma.masked_greater(objmap,0).mask,selem[None,:,:])
    
    # label dilated objectmask 
    object_mask_dilated = ndi_label(object_mask_dilated)[0]
    
    # get object labels of new developed CIs
    object_labels = np.unique(object_mask_dilated[cimask])
    
    # remove all none CI objects
    object_mask_ci = filter_objects(object_mask_dilated,object_labels)
    
    # create mask of undilated objects
    object_mask = (object_mask_ci>0) & (objmap>0)
    
    # return 3D connected objects
    objects = object_mask_dilated * object_mask
    
    return objects

def object_props(time, objmap, cimask):
    
    # get bounding box of objects
    props = regionprops(objmap)
    _id = np.array([p.label for p in props])
    _bbox = np.array([p.bbox for p in props])
    del props

    # get bounding box at t0
    om = objmap.copy()
    om[~cimask] = 0
    props = regionprops(om)
    _bbox_t0 = np.array([p.bbox for p in props])
    del props, om

    # build dataframe
    cols = ['time','id','t0','t1','l00','l01','c00','c01','l0','l1','c0','c1']
    df =np.hstack((_id[:,None],_bbox[:,[0,3]],_bbox_t0[:,[1,4,2,5]],_bbox[:,[1,4,2,5]]))
    df = pd.DataFrame(df,columns=cols[1:])
    df.insert(0,'time',np.array(time)[df.t0])
    idx = df[['time','id']].apply(lambda x: '{0[0]:%Y-%m-%d}-{0[1]:06d}'.format(x),axis=1)
    df.set_index(idx,inplace=True)
    df.insert(2,'flags',0)
    return df

def detect_ci(rx,thresh=35.0,radius=15.0):#,nstep=5):
    '''
    Apply Haeberli algorithm for CI detection
    to RADOLAN RX data.
    '''
    # 1a. get mask of convectively active grid cells by thresholding
    camask = ca_mask(rx.dbz,thresh)

    # 2. label connected regions above 35dBz
    selem = np.zeros((3,3,3))
    selem[1,:,:]=1
    objmap, nr_obj =  label_objects(camask,selem)

    # 3a. get mask of invalid data...
    fillmask = (rx.dbz.values==250)
    # 3b. ... and build buffer mask around existing convection/borders
    bufmask = buffer_mask(camask|fillmask,radius=radius)

    # 4a. get mask of new CI grid cells ...
    cimask = ci_mask(camask,bufmask)
    # 4b. ... and extract their object labels
    ci_labels = np.unique(objmap[cimask])
    
    # 5. link CI objects with their subsequent children
    #objmap = link_objects(objmap,ci_labels,nstep)[0]
    objmap = link_objects_fast(objmap,cimask)

    # 6. extract object properties ...
    objprops = object_props(rx.time,objmap,cimask)
    # ... and return
    return objprops

def colorbar():
    from imageio import imread
    fname = os.path.join(_data_dir,'colorbar.png')
    return imread(fname).squeeze()

_cbar = colorbar()

def dbz2rgb(dbz):
    thresh = np.array([ 
        -32.5,  5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0,
        40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 92.5
    ])
    if hasattr(dbz,'scale_factor') and hasattr(dbz,'add_offset'):
        a = getattr(dbz,'scale_factor')
        b = getattr(dbz,'add_offset')
        thresh = (thresh-b)/a
    return _cbar[np.digitize(dbz.values,thresh)-1]

def time2index(t,dt=5):
    t = (t.hour*60 + t.minute)/dt 
    return int(t)

def recalibrate_value(arr,value):
    if hasattr(arr,'scale_factor') and hasattr(arr,'add_offset'):
        a = getattr(arr,'scale_factor')
        b = getattr(arr,'add_offset')
        return (value-b)/a
    else:
        return value
