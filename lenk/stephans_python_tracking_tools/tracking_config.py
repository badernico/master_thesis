#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 13:58:42 2018

@author: lenk

This file defines needed configurations for the tracking routines.
"""
from standard_config import *
# path definitions ------------------------------------------------------------
tracking_routine_path = "{}/utils/tracking".format(local_home_path)

# data range definitions ------------------------------------------------------
data_range = {'IR_108':{'vmin':210.,'vmax':300},
              'HRV':{'vmin':0.,'vmax':1.},
              'radar':{'vmin':0.,'vmax':85.}}

# configuration of tracking algorithm parameters ------------------------------
farnebaeck_parameters = {'flow': None,
                         'pyr_scale':0.5, 
                         'levels': 5,
                         'win_size':15,
                         'iterations':3,
                         'poly_n':5,
                         'poly_sigma':1.2,
                         'flags':0 }

tvl1_parameters = {'epsilon':0.01,
                   'lambda':0.05,#0.2,
                   'outer_iterations':300,#20,#40,
                   'inner_iterations':2,#5,#7,
                   'gamma':0.1,#0.4,
                   'scales_number':5,
                   'tau':0.25,
                   'theta':0.3,#0.8,
                   'warpings_number':2,#3,#5,
                   'scale_step':0.5,
                   'median_filtering':1,
                   'use_initial_flow':0}

cross_correlation_parameters = {'box_size':51,
                                #'target_box_size':17,
                                'target_box_size':51,
                                'upsample_factor':6}

object_thresholds = {'ir108':{'min':-220,'max':-250},
                     'hrv':{'min':0.3,'max':0.4}}

