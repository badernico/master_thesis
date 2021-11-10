#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:42:40 2018

@author: lenk

This file contains common routines used for the tracking.
"""

import numpy as np
import xarray as xr
import pandas as pd
import glob
import logging

from standard_config import *

import sys
sys.path.append("{}/utils".format(local_home_path))
import zeit_utils as zu

from l15_msevi.msevi import MSevi
from analysis_tools import grid_and_interpolation as gi

sys.path.append("{}/utils/tracking".format(local_home_path))
import cross_correlation_tracking as cct
import optical_flow_tracking as oft  
import object_tracking as obt  

import scipy.ndimage as ndi
from skimage import filters

import networkx as nx
                 

# write error to log file
def log_error(error,experiment_no,log_path):
    experiment_path = "/vols/satellite/home/lenk/proj/2017-10_tracking/experimente"
    log_file = "{ep}/log/error_log_{no}.log".format(ep=experiment_path,no=experiment_no)

    logging.basicConfig(filename=log_file, level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s %(message)s')
    logger=logging.getLogger(__name__)

    logger.error(error)

# scale a field into a given range ------------------ -------------------------
def scale_array_min_max(array_data,range_min=0,range_max=1):
    """
    Scales a array into the chosen range.
    
    Inputs:
    -------
    array_data: numpy array of floats or integers, 2d
        array to scale
    range_min: int or float, default = 0
        minimum value of the range to scale array to,
    range_max: int or float, default = 1
        maximum value of the range to scale array to,
        
    Returns:
    --------
    scaled_array: numpy array of floats, 2d
    """
    # get array extrema
    array_min = np.min(array_data)
    array_max = np.max(array_data)

    # derive conversion parameters
    a = (range_max - range_min) / (array_max - array_min)
    b = range_max - a * array_max

    # scale array
    scaled_array = a * array_data + b
    
    return scaled_array

# transform array into the range [0,255] and turn it into uint8 for opencv ----
def transform_array2picture(input_array):
    """
    Transforms a given numpy array into the range [0,255].

    Inputs:
    -------
    input_array: numpy array, 2d or 3d, float or int
        array to be transformed

    Returns:
    --------
    picture: numpy array with shape of input array, uint8
        transformed array
    """
    picture = scale_array_min_max(input_array,0,1)*255.999

    return picture.astype("uint8")

def transform_picture2array(picture,vmin,vmax):
    """
    Transforms a given numpy in the range [0,255] to the given range.

    Inputs:
    -------
    picture: numpy array, 2d or 3d, uint8
        array to be transformed
    vmin: float
        minimum value of the range to transform the array in
    vmax: float
        maximum value of the range to transform the array in

    Returns:
    --------
    output_array: numpy array with shape of input array, float
        transformed array
    """
    output_array = picture / 255.999
    output_array = scale_array_min_max(output_array,vmin,vmax)

    return output_array

# load paths to the trajectory data of all manual tracks ----------------------
def load_track_paths(track_directory):
    """
    Loads the paths of all trackes stored under the track path


    Parameters
    ----------
    track_directory: str
                     Directory, where the track files have been stored into.

    Returns
    -------
    list
        List with the track paths

    """

    path = "%s/*" % track_directory
    tracks = glob.glob(path)

    tracks.sort()

    return(tracks)

# load paths to the data 25x25 pixel around trajectroy points -----------------
def load_track_data_paths():
    """
    Loads the paths of the trackdata for all the manual tracks.

    Parameters
    ----------
    none

    Returns
    -------
    list
        List with the track paths

    """
    tracks = []

    with open("/vols/satellite/home/lenk/data/trackdata/analysetracks_liste", 'r') as file:
        for line in file:
            zeile = line.split("\n")
            tracks.append(zeile[0])

    return tracks

# load satellite data ---------------------------------------------------------
def load_satellite_data(time,channel,region='eu',scan_type='rss'):
    """
    Loads satellite data of a given time and channel.

    Parameters
    ----------
    time: datetime object
          date and time of which satellite data shall be loaded

    channel: string
             channel to load

    region: str, default: 'eu'
           region to get satellite data from

    scan_type: str, 'pzs' or 'rss', default: 'rss'
            scan type to load data from
    """
    if region=='germ' or region=='de' and scan_type =='rss':
         region =((216, 456), (1676, 2076))
    elif region=='germ' or region=='de' and scan_type =='pzs':
         region =((216, 456), (1876, 2276))

    s = MSevi(time=time,region=region,scan_type=scan_type,chan_list=[channel])

    if channel[0:2] == 'IR'and channel != 'IR_016' or channel[0:2] == 'WV' :
        s.rad2bt()
        return(s.bt[channel])
    else:
        s.rad2refl()
        return(s.ref[channel])

# load radar data -------------------------------------------------------------
def load_radar_data(date_time):
    """
    Loads radar data of a given time.

    Parameters
    ----------
    date_time: datetime object
          date and time of which radar data shall be loaded

    """

    radolan_path = "/vols/talos/datasets/radolan/rx_hdcp2/"

    year,month,day = date_time.strftime("%Y"), \
                     date_time.strftime("%m"), \
                     date_time.strftime("%d")

    time_idx = (date_time.hour*60 + date_time.minute) / 5

    file_name = "hdfd_miub_drnet00_l3_dbz_v00_%s%s%s000000.nc" % (year,month,day)

    file_path = '%s/%s/%s' % (radolan_path, year, file_name)

    radolan_data = xr.open_dataset(file_path)

    return radolan_data['dbz'][time_idx].data


def load_base_field_data(start_time,end_time,field):
    """
    Load a given type of field data starting from a given time step for a given
    amount of time.

    Parameters
    ----------
    ci_time: datetime object
                date and time of the starting point

    start_time: datetime object
                first time step in the track

    end_time: datetime object
                last time step in the track

    field: str
           string of which data to load, possible strings are:
               * all MSG channels composed of VIS, WV_ or IR_ and the channel center
                 wavelength (006,008,016,039,062,076,087,097,108,120,134) or HRV
               * Radolan radar data, 'radar' or 'radolan'

    Returns
    -------
    3d stack of numpy arrays (time,rows,cols)
            stack with data in the time frame -n_time_steps<=start_time<=n_time_steps


    """

    # create a time list with all nedded time steps
    tlist = zu.zeitliste2(start_time,end_time,5)

    field_data = []
    if field[0] == 'R' or field[0] == 'r':
        for t in tlist:
            field_data.append(load_radar_data(t))
    else:
        for t in tlist:
            field_data.append(load_satellite_data(t,field,region='germ'))

    return np.array(field_data)

# calculate the angular error between a derived trackpoint and a reference point
def calc_angular_error(r_ref,c_ref,r,c,origin_r=0,origin_c=0):
    """
    Calculates the angular error between a point and a reference point according to:

    .. math::
        \\epsilon_\mathrm{a} = \\mathrm{arccos} \\left(
                                \\frac{c \cdot c_\mathrm{ref} + r \cdot r_\mathrm{ref}}
                                      {\\sqrt{r^2 + c^2} \cdot \\sqrt{c_\mathrm{ref}^2+r_\mathrm{ref}^2}}
                                \\right)

    Parameters
    ----------
    r_ref: float
        row index of the reference point
    c_ref: float
        column index of the reference point
    r: float
        row index of the point to compare
    c: float
        column index of the point to compare
    origin_r: float, default = 0
        row index of the origin point
    origin_c: float, default = 0
        column index of the origin point    
    

    Returns
    -------
    float
        calculated angular error
    """
    
    # re-calculate vectors to start from origin
    r = r - origin_r
    r_ref = r_ref - origin_r
    c = c - origin_c
    c_ref = c_ref - origin_c
    #wf = np.arccos((c*c_ref + r*r_ref)/(np.sqrt(r**2+c**2+1) * np.sqrt(c_ref**2+r_ref**2+1)))
    wf = np.arccos((c*c_ref + r*r_ref)/(np.sqrt(r**2+c**2) * np.sqrt(c_ref**2+r_ref**2)))
    
    if np.any(np.isnan(wf)):
        wf.values[np.where(np.isnan(wf))] = 0
        
    return np.rad2deg(wf)

# calculate the end point error between a derived trackpoint and a reference point
def calc_end_point_error(r_ref,c_ref,r,c,size_r, size_c):
    """
    Calculates the end point error between a point and a reference point according to:

    .. math::
        \\epsilon_\mathrm{e} = \\sqrt{(c_\mathrm{ref}-c)^2 + (r_\mathrm{ref}-r)^2}

    Parameters
    ----------
    r_ref: float
        row index of the reference point
    c_ref: float
        column index of the reference point
    r: float
        row index of the point to compare
    c: float
        column index of the point to compare
    size_r: float
        size of a base grid pixel in row direction in km
    size_c: float
        size of a base grid pixel in column direction in km

    Returns
    -------
    float
        calculated end point error
    """
    return np.sqrt(((c_ref-c)*size_c)**2+((r_ref-r)*size_r)**2)

# create a track for a given start point using different tracking techniques
def calculate_track_oneway(field_stack,index,direction,method,boxsize=51):
    """
    Calculate dense optical flow according to the TV_L1 approach after Zach et 
    al. (2007) for a given 3d stack of fields, using a cutout around the track 
    points to speed up the flow calulation.

    Parameters
    ----------
    field_stack: 3d array of floats, with field_stack(time,rows,cols)
              3d stack of fields to derive the flow from
    index: tuple with index(time,row,col)
           index of where the starting point is located
    direction: string
          direction in which to calcualte the track from the starting point,
          forward or backward
    method: string
            method used for tracking, possible: farnebaeck, tvl1, xcorr
    boxsize: int, defualt = 51
             size of the box to cutout around the track points


    Returns
    -------
    track
        list with coordinates tuples of the trajectory points, with (time step,row,col)
    """
    # set up valid methods
    valid_methods = ['farnebaeck','tvl1','xcorr','cc','obj','dis']
    
    # create list to save results in
    track = []
    
    # extract starting coordinates
    t00 = index[0]
    r = int(np.rint(index[1]))
    c = int(np.rint(index[2]))
    
    # extract shape
    ntimes, nrows, ncols = field_stack.shape

    # append starting coordinates to track
    track.append((0,r,c))
    
    # create subset of the field data to do the tracking on
    if direction in ['forward','for','vor','+','f','v']:
        fields = field_stack[t00:]
        d_time = 1
    elif direction in ['backward','back','rueck','-','b','r']:
        fields = field_stack[:t00+1][::-1]
        d_time = -1
    else:
        print("Unknown tracking direction. Only 'forward' and 'backward' are valid.")
        return
    
    # loop over the field data
    t0 = t00
    
    if method in valid_methods and method == 'obj':
        try:
            if direction in ['backward','back','rueck','-','b','r']:
                t0 = 0
                start = t0
            else:
                t0 = t00 -2
                start = index[0]
            
            object_arrays, object_tracks = obt.track_object(fields,track_point='centroid')
            
            # find the object track that is closest to starting point
            distances = []
            nodes = []
            for obj_track in object_tracks:
               
               node_names = [o.split("_") for o in sorted(obj_track.nodes)]
               time_steps = np.array([int(n[0]) for n in node_names])
               obj_ids = np.array([n[1] for n in node_names])
               
               target_nodes = obj_ids[np.where(time_steps==start)]
               
               target_nodes = np.array(['{}_'.format(start) + nd for nd in target_nodes])
               
               for tn in target_nodes:
                   r = obj_track.node[tn]['row']
                   c = obj_track.node[tn]['col']
                   distances.append(np.sqrt((index[1]- r)**2 + (index[2]- c)**2))
                   nodes.append(tn)
               
               wanted_node = np.array(nodes)[np.where(distances == np.min(distances))]
               
               if obj_track.has_node(wanted_node[0]):
                   wanted_track = obj_track
               else:
                   continue
            # transform wanted track into directed graph
            wanted_track = nx.DiGraph(wanted_track)
           
            # sort wanted track and only take the most direct way
            direct_track = list(nx.dfs_preorder_nodes(wanted_track,wanted_node[0]))[0:len(fields)]
            
            track = [] 
                           
            for track_point in direct_track:
               c = wanted_track.node[track_point]['col']
               r = wanted_track.node[track_point]['row']
                
               t_idx = ((t0 + d_time)+1 - t00)*5
               
               track.append((t_idx,r,c))
               t0 += d_time
                
            if len(track) < len(fields):
              for i in np.arange((len(fields) - len(track)),0,-1):
                  t_idx = ((t0 + d_time)+1 - t00)*5 
                  track.append((t_idx,np.nan,np.nan))
                  
                  t0 += d_time
            elif len(track) > len(fields):
                track = track[:len(fields)]
            else:
                track = track
        except:
            t0 = t00 - 2
            track = []
            for i in np.arange(len(fields)):
                t_idx = ((t0 + d_time)+1 - t00)*5 
                track.append((t_idx,np.nan,np.nan))
                  
                t0 += d_time
                
               
    else:
        for i in range(len(fields)-1):
            # create the cutouts
#            cutout0 = gi.cutout_field4box(fields[i],
#                                          (int(np.rint(r)),int(np.rint(c))),
#                                          boxsize)
#            cutout1 = gi.cutout_field4box(fields[i+1],
#                                          (int(np.rint(r)),int(np.rint(c))),
#                                          boxsize)
            
            r_prev = r
            c_prev = c
            ri = int(np.rint(r_prev))
            ci = int(np.rint(c_prev))
    
            if method in valid_methods:
                # calculate the movement between the time steps
                if method in ['farnebaeck','farneback','tvl1','dis']:
                    #step_flow = oft.calculate_optical_flow(field_stack=np.array([cutout0,cutout1]),
                    #                                       method=method,filtering=False)

                    step_flow = oft.calculate_optical_flow(field_stack=np.array([fields[i],fields[i+1]]),
                                       method=method,filtering=False)
                    
                    
                    # calculate new position of the track point
                    #center = boxsize / 2   # ist das falsch? -> Ja, ist es!
                    
                    #r = np.clip(r + step_flow[0][...,1][center,center],0,nrows)
                    #c = np.clip(c + step_flow[0][...,0][center,center],0,ncols)
                    
                    r = np.clip(r + step_flow[0][...,1][ri,ci],0,nrows)
                    c = np.clip(c + step_flow[0][...,0][ri,ci],0,ncols)
                    
                    # append coordinates to track
                    t_idx = ((t0 + d_time) - t00)*5
                    
                    track.append([t_idx,r,c])
                    
                    t0 += d_time
                    
                elif method in ['xcorr','cc']:
                    shift= cct.calc_cross_correlation_point_shift((r,c),cutout0,cutout1)
                    
                    #r = np.clip(r + shift[0],0,nrows)
                    #c = np.clip(c + shift[1],0,ncols)
                    
                    # append coordinates to track
                    t_idx = ((t0 + d_time) - t00)*5
                    
                    #track.append([t_idx,r,c])
                    track.append([t_idx,shift[0],shift[1]])
                    
                    t0 += d_time
                    
                
            else:
                print("Unknown tracking method. Only %s are valid." % valid_methods)
                return
   
    #if direction in ['backward','back','rueck','-','b','r']: 
    #    track = track[:-1]
          
    return track

# calculate track forward and backward
def calculate_track(base_data,start_index,method,box_size=51):
    """
    Calculates a track from a given starting point on the given data using the 
    specified method for the specified time range forward and backward from the
    starting index.
    
    Parameters:
    -----------
    base_data: numpy array, 2d or 3d with (time,row,col)
        data used to track on
    start_index: tuple with (time_index, row, col)
        index of the starting point, the time index is relative to the shape of
        the base data
    method: string, possible: farnebaeck, tvl1, xcorr, obj
        specifier for the tracking method to be used
    box_size: int
        size of the cutout around the track points used to derive the movement
        from
        
    Returns:
    --------
    combined_track: pandas data frame
        track of the point over the given base data
    """
    
    # calculate the track
    forward_track = calculate_track_oneway(base_data,
                                           start_index,
                                           '+',
                                           method,
                                           box_size)
    backward_track = calculate_track_oneway(base_data,
                                            start_index,
                                            '-',
                                            method,
                                            box_size)
    
    # combine forward an backward track
    combined_track = backward_track[::-1][:-1] + forward_track
  
    
    track_dict = {'time_step':[step[0] for step in combined_track],
                  'row_index':[step[1] for step in combined_track],
                  'column_index':[step[2] for step in combined_track]}
    
    combined_track = pd.DataFrame(track_dict)
    
    return combined_track

# label connections between clusters
def label_connections( cluster_t0, cluster_t1, time_name0 = '', time_name1 = ''):
    """
    Labels the connections between two given labeled fields.

    Parameters
    ----------
    cluster_t0: numpy array, 2d, float
        labeled array at first time step

    cluster_t1: numpy array, 2d, float
        labeled array at second time step

    time_name0: string, default = ''
        time label for the first time step

    time_name1: string, default = ''
        time label for the second time step

    Returns
    -------
    connections: dictionary
        labeled connections between the given fields
    """

    # get slices .....................................................
    oslices = ndi.measurements.find_objects( cluster_t0 )

    # set index
    index = range(1, len(oslices) + 1)

    # loop over slices ...............................................
    connections = {}
    for ilab0 in index:
        try:
        
            lab_slice = oslices[ilab0 - 1]
            
            cutout0 = cluster_t0[lab_slice]
            cutout1 = cluster_t1[lab_slice]
            
            mask0 = (cutout0 == ilab0)
            label1_list = list(set( cutout1[mask0] ) - set([0,]))
    
            label1_list_str = [ '%s_%s'% (time_name1, str(l).zfill(4))  for l in label1_list]
            
            key = '%s_%s' % (time_name0, str(ilab0).zfill(4))
            connections[key] = label1_list_str
        except:
            continue
    
    return connections

# morph a given field with a given flow
def morph_field(field, column_shift, row_shift, method = 'forward'):
    
    '''
    Applies morphological transformation of field f given a displacement 
    field.
    
    Inputs:
    -------
    field: 2d array
       field that is transformed (source)
    column_shift: 2d array, same shape as field
       displacement vector between source and target stage in column direction
    row_shift: 2d array, same shape as field
       displacement vector between source and target stage in row direction

    Output:
    -------
    field_trans: transformed field
    
    '''


    # get shape of field and corresponding index set -----------------
    nrows, ncols = field.shape
    irow, icol = gi.make_index_set(nrows, ncols)
    # ================================================================

    # get index shift from real-values flow field --------------------
    # care that row and column are transposed here!!!!
    ishift_row = np.rint(row_shift).astype(np.int)
    ishift_col = np.rint(column_shift).astype(np.int)
    
    if method == 'forward':
        ir = irow - ishift_row
        ic = icol - ishift_col
    elif method == 'backward':
        ir = irow + ishift_row
        ic = icol + ishift_col
        

    ir = np.clip(ir, 0, nrows - 1)
    ic = np.clip(ic, 0, ncols - 1)
    # ================================================================

    return field[ir, ic]    
    
