#! /usr/bin/python

# load modules
from satpy.scene import Scene
from satpy.resample import get_area_def
from satpy.dataset import combine_metadata
from satpy import find_files_and_readers
from satpy.writers import available_writers
from satpy import DataQuery
from datetime import datetime
from pyresample import load_area
from pyresample.geometry import AreaDefinition
import glob
import warnings
import numpy as np
import xarray
import netCDF4
import sys

# date of interest
# need to find correct file path
doi = '2021-07-11' # year-month-day

# list the to file(s) at the doi
y_oi = doi.split('-')[0]
m_oi = doi.split('-')[1]
d_oi = doi.split('-')[2]
fnames = glob.glob('/Volumes/Elements/data/msevi_rss/raw_nat/'+y_oi+'/'+m_oi+'/'+d_oi+'/*')
fnames.sort()

range(len(fnames))

for i in range(len(fnames)):
    # get the i-th filename 
    fname = [fnames[i]]
    ftext = fnames[i]
    
    # get date and time information of the file
    ftext = ftext.split("-")[5]
    ftext = ftext.split(".")[0]
    year = list(ftext)[0]+list(ftext)[1]+list(ftext)[2]+list(ftext)[3]
    month = list(ftext)[4]+list(ftext)[5]
    day = list(ftext)[6]+list(ftext)[7]
    hour = list(ftext)[8]+list(ftext)[9]
    minute = list(ftext)[10]+list(ftext)[11]
    second = list(ftext)[12]+list(ftext)[13]
    
    # read the i-th file: SEVIRI Rapid Scan (native format)
    scn = Scene(reader='seviri_l1b_native', filenames=fname)
    
    # load channels, calibrations and composites
    scn.load(['HRV',1.6,0.6,0.8], calibration='radiance')
    scn.load([3.9,8.7,9.7,12.0,13.4,7.3,10.8,6.2], calibration='brightness_temperature')
    scn.load(['convection'])
    scn.load(['natural_color'])
    
    # resample
    scn = scn.resample("my_germ")
    
    # netCDF file
    scn.save_datasets(filename = 'MSG3-SEVI-MSG15-0100-NA-'+year+month+day+hour+minute+second+'.'+fnames[i].split('.')[1]+'.nc', 
                      base_dir = '/Volumes/Elements/data/msevi_rss/raw_netcdf/'+y_oi+'/'+m_oi+'/'+d_oi+'/', 
                      writer = 'cf', engine = 'netcdf4')
    print(str(round((i+1)/len(fnames)*100,0))+' %')
