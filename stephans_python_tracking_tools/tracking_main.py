# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 13:45:33 2017
last modified: Tue Oct 09 09:03:53 2018
@author: lenk
"""

###############################################################################
#                          import needed packages                             #
###############################################################################
import numpy as np
import pandas as pd
import glob
import sys

sys.path.append("/vols/satellite/home/stephan/utils")
import zeit_utils as zu

                              
sys.path.append("/vols/satellite/home/stephan/utils/tracking")
import tracking_config as tc
import cross_correlation_tracking as cct
import optical_flow_tracking as oft 
import object_tracking as ot   
import tracking_common as tco                       

from analysis_tools import grid_and_interpolation as gi

import datetime as dt

import matplotlib.pyplot as plt

###############################################################################
#                           function definitions                              #
###############################################################################



if __name__ == '__main__':
    # test case
    base_data = tco.load_base_field_data(dt.datetime(2012,7,28,11,0),
                                         dt.datetime(2012,7,28,12,0),
                                         'IR_108')

    # calculate tracks
    start_index = (6,200,200)
    fb_track = tco.calculate_track(base_data,start_index,'farnebaeck',51)
    tvl1_track = tco.calculate_track(base_data,start_index,'tvl1',51)
    cc_track = tco.calculate_track(base_data,start_index,'xcorr',51)
    
    # combine tracks
    times = pd.date_range(start = dt.datetime(2012,7,28,11,0).strftime("%Y%m%dT%H%M"),
                          end = dt.datetime(2012,7,28,12,0).strftime("%Y%m%dT%H%M"),
                          freq="5min")
    t_strings = [t.to_pydatetime().strftime("%Y%m%d%H%M") for t  in times]
    
    track_df = {'time_step': fb_track.time_step.values,
                'col_farnebaeck': fb_track.column_index.values,
                'row_farnebaeck': fb_track.row_index.values,
                'col_tvl1': tvl1_track.column_index.values,
                'row_tvl1': tvl1_track.row_index.values,
                'col_xcorr': cc_track.column_index.values,
                'row_xcorr': cc_track.row_index.values,
                'time':times}
    
    track_df = pd.DataFrame(track_df)
    
   
    field_segmented = ot.find_object_hysteresis_threshold_2d(-base_data,-220,-250)
    
    i = 5
    fig,ax = plt.subplots(1,1)
    ir = ax.imshow(base_data[i],vmin=210,vmax=300,cmap='gray_r')
    obj = ax.imshow(np.ma.masked_less(field_segmented[0][i],2),vmin=1,alpha=0.4)
    fig.colorbar(ir)
    
    obj_track = ot.generate_graph_from_components(field_segmented[0],t_strings)

