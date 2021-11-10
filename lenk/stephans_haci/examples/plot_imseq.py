import numpy as np
import pandas as pd
from scipy.misc import imsave
import haci as hci
import matplotlib.pyplot as plt

def bbox_is_incomplete(bbox,shape):
    if bbox[0].start<0 or bbox[0].stop>shape[0]: return True
    if bbox[1].start<0 or bbox[1].stop>shape[1]: return True
    if bbox[2].start<0 or bbox[2].stop>shape[2]: return True
    return False

def obj2bbox(obj,nt=13,nl=75,nc=75):
    # time slice
    t0 = obj.t0-(nt/2)
    t1 = obj.t0+(nt-nt/2)
    ts = slice(t0,t1)
    # line slice
    l0 = obj.l00-(nl-(obj.l01-obj.l00))/2
    l1 = l0+nl
    ls = slice(l0,l1)
    # column slice
    c0 = o.c00-(nc-(o.c01-o.c00))/2
    c1 = c0+nc
    cs = slice(c0,c1)
    return (ts,ls,cs)

fname = 'ci01/2013-07-01/haci-2013-07-01-ci.dat'
objprops = pd.read_table(fname,delim_whitespace=True,header=0,converters={'flags':lambda x: int(x,16)})
objprops.time = [pd.Timestamp(d+'T'+t) for d,t in zip(objprops.date,objprops.time)]
del objprops['date']
m = (objprops.t1-objprops.t0)==5
objprops = objprops[m]
nl = nc = 75
bbdict = {}
rx = hci.read_rx_hdcp2('/home/deneke/src/radolan/data/hdfd_miub_drnet00_l3_dbz_v00_20130701000000.nc')

for o in objprops.itertuples():
    bbox = obj2bbox(o)
    if bbox_is_incomplete(bbox,rx.dbz.shape): continue
    bbdict[o.id]= bbox
    print bbox
    fname = 'cell-{dt:%Y-%m-%d}-{id:03}.png'.format(dt=o.time,id=o.id)
    print fname
    img = hci.dbz2rgb(rx.dbz[bbox]).reshape((13*nl,nc,4))
    for i in xrange(0,975,75):
        img[i,:,:]  = 255
    img[-1,:,:]  = 255
    img[:,0,:]  = 255
    img[:,-1,:] = 255
    imsave(fname,img)




