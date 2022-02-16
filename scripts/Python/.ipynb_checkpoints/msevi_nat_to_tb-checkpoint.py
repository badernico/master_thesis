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

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


print('Script - START')

# loop over all sensing dates ('data/sensing_dates.txt')
sed=open('/Users/nicobader/Documents/Uni_Leipzig/master_thesis/data/sensing_dates.txt','r').read().split(" ")

for dd in np.arange(0,len(sed),1):
    # date of interest
    # need to find correct file path
    doi = sed[dd] # year-month-day

    # list the to file(s) at the doi
    y_oi = doi.split('-')[0]
    m_oi = doi.split('-')[1]
    d_oi = doi.split('-')[2]
    fnames = glob.glob('/Volumes/Elements/data/msevi_rss/raw_nat/'+y_oi+'/'+m_oi+'/'+d_oi+'/*')
    fnames.sort()

    for i in range(len(fnames)):
        # get the i-th filename 
        fname = [fnames[i]]
        ftext = fnames[i]

        # get date and time information of the file
        ftext = ftext.split("/")[9]
        ftext = ftext.split(".")[0]+'.'+ftext.split(".")[1]
        #year = list(ftext)[0]+list(ftext)[1]+list(ftext)[2]+list(ftext)[3]
        #month = list(ftext)[4]+list(ftext)[5]
        #day = list(ftext)[6]+list(ftext)[7]
        #hour = list(ftext)[8]+list(ftext)[9]
        #minute = list(ftext)[10]+list(ftext)[11]
        #second = list(ftext)[12]+list(ftext)[13]

        # read the i-th file: SEVIRI Rapid Scan (native format)
        scn = Scene(reader='seviri_l1b_native', filenames=fname)

        # load channels, calibrations and composites
        scn.load([3.9,8.7,9.7,12.0,13.4,7.3,10.8,6.2], calibration='brightness_temperature')

        # resample
        scn = scn.resample("my_germ")

        # netCDF file
        scn.save_datasets(filename = ftext+'.nc', 
                          base_dir = '/Volumes/Elements/data/msevi_rss/raw_tb/'+y_oi+'/'+m_oi+'/'+d_oi+'/', 
                          writer = 'cf', engine = 'netcdf4')
        print(doi)
        print(str(round((i+1)/len(fnames)*100,0))+' %')
    print(str(len(sed)-dd)+' sensing date(s) left')
print('Script - END')