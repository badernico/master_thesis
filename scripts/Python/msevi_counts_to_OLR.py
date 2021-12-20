#! /usr/bin/python

# load modules
import netCDF4 as nc
import glob
import numpy as np
import datetime
import os

# system timezone to UTC
# everything will be done in UTC
# UTC = CET-1h/CEST-2h
os.environ['TZ'] = 'UTC'

# date of interest
# need to find correct file path
doi = '2021-07-30' # year-month-day

# list the to file(s) at the doi
y_oi = doi.split('-')[0]
m_oi = doi.split('-')[1]
d_oi = doi.split('-')[2]

fnames = glob.glob('/Volumes/Elements/data/msevi_rss/raw_netcdf/'+y_oi+'/'+m_oi+'/'+d_oi+'/*')
fnames.sort()

print('START - OLR calculation')

for i in range(len(fnames)):
    # get the i-th filename 
    fname = fnames[i]
    ftext = fnames[i]

    # get date and time information of the file
    ftext = ftext.split("/")[9]
    ftext = ftext.split(".")[1]
    ftext = ftext.split("-")[1]

    year = list(ftext)[0]+list(ftext)[1]+list(ftext)[2]+list(ftext)[3]
    month = list(ftext)[4]+list(ftext)[5]
    day = list(ftext)[6]+list(ftext)[7]
    hour = list(ftext)[8]+list(ftext)[9]
    minute = list(ftext)[10]+list(ftext)[11]
    second = list(ftext)[12]+list(ftext)[13]

    # open netCDF file
    ds = nc.Dataset(fname)

    # get IR_108 and WV_062 channel data in counts unit
    ir_counts = ds.variables['IR_108'][:]
    wv_counts = ds.variables['WV_062'][:]

    # convert counts to spectral radiance (mW m-2 sr-1 cm-1)
    ir_specrad = ir_counts*0.2156+(-10.4)
    wv_specrad = wv_counts*0.0083+(-0.424)

    # convert spectral radiance to radiance
    ir_rad = ir_specrad*65.836/1000
    wv_rad = wv_specrad*88.625/1000

    # calculating outgoing longwave radiation (OLR) (Singh et al., 2007)
    olr = 12.94*ir_rad+16.5*wv_rad+((10.09*wv_rad)/ir_rad)+((12.94*wv_rad)/(ir_rad+0.39))+77.47
    
    # change column order
    olr = olr[np.arange(olr.shape[0]-1,0-1,-1),:]
    
    # put array into a big OLR dataset
    if i==0:
        OLR = np.ma.asanyarray([olr]*len(fnames))
    else:
        OLR[i] = olr

    # SENSING TIME AS UNIX-TIMESTAMP
    # seconds since 1970-01-01 00:00:00 (UTC)
    dt_str = year+'-'+month+'-'+day+' '+hour+':'+minute+':'+second
    dt_object = datetime.datetime.fromisoformat(dt_str)
    timestamp = datetime.datetime.timestamp(dt_object)

    # put array into a big time array
    if i==0:
        times = np.ma.array([int(timestamp)]*len(fnames))
    else:
        times[i] = timestamp

    print(str(round((i+1)/len(fnames)*100,0))+' %')

print('END - OLR calculation')

# GET X,Y DIMENSIONS
# define latitude and longitude array
lat = ds.variables['latitude'][:]
lon = ds.variables['longitude'][:]
lat_dim = np.ma.masked_array(np.arange(lat.min(),lat.max()+0.001,(lat.max()-lat.min())/(lat.shape[1]-1)))
lon_dim = np.ma.masked_array(np.arange(lon.min(),lon.max()+0.001,(lon.max()-lon.min())/(lon.shape[0]-1)))

print('START - producing netCDF file')

# get the 0-th filename 
ftext = fnames[0]

# get date and time information of the file
ftext = ftext.split("/")[9]
ftext = ftext.split(".")[1]
ftext = ftext.split("-")[1]

year = list(ftext)[0]+list(ftext)[1]+list(ftext)[2]+list(ftext)[3]
month = list(ftext)[4]+list(ftext)[5]
day = list(ftext)[6]+list(ftext)[7]
hour = list(ftext)[8]+list(ftext)[9]
minute = list(ftext)[10]+list(ftext)[11]
second = list(ftext)[12]+list(ftext)[13]

# CREATE THE NETCDF FILE
nc_fn = '/Volumes/Elements/data/msevi_rss/olr/'+y_oi+'/'+m_oi+'/'+'OLR_'+year+month+day+'.nc'
nc_ds = nc.Dataset(nc_fn, 'w', format = 'NETCDF4')

# define dimensions
time = nc_ds.createDimension('time', len(times))
lat = nc_ds.createDimension('lat', len(lat_dim))
lon = nc_ds.createDimension('lon', len(lon_dim))

# define variables
olr = nc_ds.createVariable('olr', 'f4', ('time', 'lat', 'lon',))
time = nc_ds.createVariable('time', 'f4', ('time',))
lat = nc_ds.createVariable('lat', 'f4', ('lat',))
lon = nc_ds.createVariable('lon', 'f4', ('lon',))

# define units
olr.units = 'W m^-2'
time.units = 'seconds since 1970-1-1'
lat.units = 'degrees_north'
lon.units = 'degrees_east'

# define other names
olr.long_name = 'OLR'
time.standard_name = 'time'
lat.standard_name = 'latitude'
lon.standard_name = 'longitude'

# define axis
time.axis = 'T'
lat.axis = 'Y'
lon.axis = 'X'

# define others
time.calendar = 'gregorian'
lat.datum = 'WGS84'
lon.datum = 'WGS84'

# define spacing
lat.spacing = str((lat_dim[len(lat_dim)-1]-lat_dim[0])/(len(lat_dim)-1))
lon.spacing = str((lon_dim[len(lon_dim)-1]-lon_dim[0])/(len(lon_dim)-1))

# assign data to variables
lat[:] = lat_dim
lon[:] = lon_dim
time[:] = times
olr[:] = OLR

# close the netCDF
nc_ds.close()
print('END - producing netCDF file')