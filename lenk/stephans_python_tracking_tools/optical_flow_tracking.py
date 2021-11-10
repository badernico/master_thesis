#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:40:50 2018

@author: lenk

This script contains all routines needed to calculate the optical flow of fields.
"""

# import needed packages ------------------------------------------------------
import cv2
import sys
import numpy as np

sys.path.append("/vols/satellite/home/lenk/utils/tracking_neu")
import tracking_common as tco
import tracking_config as tcf

from analysis_tools import grid_and_interpolation as gi

# functions -------------------------------------------------------------------
def calculate_optical_flow_farnebaeck(field_t0, field_t1, flow_parameters=None):
    """
    Calculate dense optical flow according to the approach after Farnebaeck 
    approach (2003) for the two given fields.

    Inputs:
    field_t0: numpy array, 2d, float32
        field at the first time step
    field_t1: numpy array, 2d, float32
        field at the second time step

    Returns:
    flow: numpy array, 3d
        flow in zonal and meridional direction
    """
    
    # pack fields for some tests
    fields = [field_t0, field_t1]

    for i,field in enumerate(fields):
        # the flow routine expects the inputs to have a data type of float32
        field = field.astype(np.float32)

        # and expects the inputs to be in the range [0,1]
        if np.min(field) != 0 and np.max(field) != 1:
            field = tco.scale_array_min_max(field,0,1)

        fields[i] = field

    # load flow parameter set
    if flow_parameters is None:
        flow_parameters = tcf.farnebaeck_parameters

    # calculate flow
    flow = cv2.calcOpticalFlowFarneback(fields[0],
                                        fields[1],
                                        flow_parameters['flow'],
                                        flow_parameters['pyr_scale'],
                                        flow_parameters['levels'],
                                        flow_parameters['win_size'],
                                        flow_parameters['iterations'],
                                        flow_parameters['poly_n'],
                                        flow_parameters['poly_sigma'],
                                        flow_parameters['flags'])

    return flow

def calculate_optical_flow_tvl1(field_t0, field_t1, flow_parameters=None):
    """
    Calculate dense optical flow according to the approach after 
    Zach et al (2007) for the two given fields.

    Inputs:
    field_t0: numpy array, 2d, float32
        field at the first time step
    field_t1: numpy array, 2d, float32
        field at the second time step

    Returns:
    optflow: numpy array, 3d
        flow in zonal and meridional direction
    """
    
    # pack fields for some tests
    fields = [field_t0, field_t1]

    for i,field in enumerate(fields):
        # the flow routine expects the inputs to have a data type of float32
        field = field.astype(np.float32)

        # and expects the inputs to be in the range [0,1]
        if np.min(field) != 0 and np.max(field) != 1:
            field = tco.scale_array_min_max(field,0,1)

        fields[i] = field

    # load flow parameter set
    if flow_parameters is None:
        flow_parameters = tcf.tvl1_parameters
    
    # set flow parameters
    initial_flow=None
    optflow=cv2.optflow.createOptFlow_DualTVL1()
    
    optflow.setEpsilon(flow_parameters['epsilon'])
    optflow.setLambda(flow_parameters['lambda'])
    optflow.setOuterIterations(flow_parameters['outer_iterations'])
    optflow.setInnerIterations(flow_parameters['inner_iterations'])
    optflow.setGamma(flow_parameters['gamma'])
    optflow.setScalesNumber(flow_parameters['scales_number'])
    optflow.setTau(flow_parameters['tau'])
    optflow.setTheta(flow_parameters['theta'])
    optflow.setWarpingsNumber(flow_parameters['warpings_number'])
    optflow.setScaleStep(flow_parameters['scale_step'])
    optflow.setMedianFiltering(flow_parameters['median_filtering'])
    optflow.setUseInitialFlow(flow_parameters['use_initial_flow'])
    
    if initial_flow is not None:
      optflow.setUseInitialFlow(True)

    # calculate flow
    flow = optflow.calc(fields[0],fields[1],initial_flow)

    return flow

def calculate_optical_flow_dis(field_t0, field_t1,mode=1):
    """
    Calculate dense optical flow according to the dense inverse search approach after 
    Koeppke et al. (2016) for the two given fields.

    Inputs:
    field_t0: numpy array, 2d, float32
        field at the first time step
    field_t1: numpy array, 2d, float32
        field at the second time step
    mode: int, default=2
        preset for the opencv DISOptical_flow module, 0 = ultrafast, 1 = fast, 2 = medium

    Returns:
    flow: numpy array, 3d
        flow in zonal and meridional direction
    """
    
    # pack fields for some tests
    fields = [field_t0, field_t1]

    for i,field in enumerate(fields):
        # the flow routine expects the inputs to have a data type of uint8
        field = tco.transform_array2picture(field)

        fields[i] = field

    # create flow object
    optflow=cv2.DISOpticalFlow_create(mode)

    # calculate flow
    flow = optflow.calc(fields[0],fields[1],None)

    return flow

# calculate optical flow
def calculate_optical_flow(field_stack,method,filtering=False, flow_parameters=None):
    """
    Calculate dense optical flow according to the TV_L1 approach after Zach et 
    al. (2007) or to the Farnebaeck approach (2003) for a given 3d stack of 
    fields, applying a bilateral filter to smooth the input fields beforehand if
    selected.

    Parameters
    ----------
    field_stack: 3d array of floats, with field_stack(time,row,cols)
              input field for the time steps
    method: string, farnebaeck or tvl1
        string, which defines the optical flow method
    filtering: boolean
        switch for the usage of the bilateral filtering
    flow_parameters: dictionary, default = None
        tracking algorithm parameters
    
    Returns
    -------
    3d array of floats, with (time,[zonal_flow],[meridional_flow])
        3d array with the flow fields. The first array represents zonal flow and
        the second one the meridional flow.
    """

    # define possbile methods
    methods = ['farnebaeck','tvl1','dis']
    
    if method in methods:
        if filtering==True:
            # apply bilateral filter to smooth the input fields a bit
            for i,f in enumerate(field_stack):
                field_stack[i] = cv2.bilateralFilter(f.copy().astype(np.float32),3,3,75)

                    
        if method == 'farnebaeck' or method == 'farneback':
            flow = [calculate_optical_flow_farnebaeck(field_stack[i-1],field_stack[i],flow_parameters)
                    for i in range(1,len(field_stack))]
         
        elif method == 'tvl1':
            flow = [calculate_optical_flow_tvl1(field_stack[i-1],field_stack[i],flow_parameters)
                    for i in range(1,len(field_stack))]
        elif method == 'dis':
            flow =  [calculate_optical_flow_dis(field_stack[i-1],field_stack[i],1)
                     for i in range(1,len(field_stack))]
        else:
            print("Given method is unknown. Only 'farnebaeck','tvl1' and 'dis' are possible.")
            return

        return flow
    else:
        print("Given method is unknown. Only 'farnebaeck','tvl1' and 'dis'are possible.")
        return
