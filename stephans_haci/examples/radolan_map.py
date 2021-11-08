from mpl_toolkits.basemap import Basemap
import numpy as np
import matplotlib.pyplot as plt
import pyproj

proj_str = '+proj=stere +lat_0=90 +lat_ts=60 +lon_0=10'        \
           '+x_0=0 +y_0=0 +a=6370040.0 +b=6370040.0 +no_defs'.format(1.0)
p =  pyproj.Proj(proj_str)
xc,yc = p(9.0,51.0)
ll_lon,ll_lat = p(xc-450.0e3,yc-450.0e3,inverse=True)
ur_lon,ur_lat = p(xc+450.0e3,yc+450.0e3,inverse=True)


bmap_param = {
    'projection' :               'stere',
    'rsphere'    : (6370040.0,6370040.0),
    'lat_0'      :                   0.0,
    'lon_0'      :                  10.0,
    'llcrnrlon'  :                ll_lon,
    'llcrnrlat'  :                ll_lat,
    'urcrnrlon'  :                ur_lon,
    'urcrnrlat'  :                ur_lat,
    'resolution' :                   'l',
    'fix_aspect' :                 False
}

nlin = 900
ncol = 900
dpi  = 100

spp = {'left':0.0,'bottom':0.0,'right':1.0,'top':1.0,'wspace':0.0,'hspace':0.0}
fig = plt.figure(figsize=(np.float(ncol)/dpi,np.float(nlin)/dpi),dpi=dpi,frameon=False)
plt.axis('off')
ax = plt.gca()
fig.subplots_adjust(**spp)

m = Basemap(**bmap_param)
m.shadedrelief()
m.drawcoastlines()
m.drawcountries()
#plt.savefig('background.png',dpi=dpi,frameon=False)
plt.show()
