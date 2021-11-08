#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 8 14:40:50 2018
Last modified on Thu Feb 14 15:13:28 2019

@author: Stephan Lenk, stephan.lenk@tropos.de

This script contains all routines needed to calculate cross correlation tracking.
"""

# import needed packages ------------------------------------------------------
from skimage.feature import register_translation
from skimage.util import view_as_windows
import numpy as np

import sys
sys.path.append("/vols/satellite/home/lenk/utils/tracking_neu")
import tracking_config as tcf

import operator

# functions -------------------------------------------------------------------

# derive integer divisors of a given number
def get_integer_divisors(number):
    divisors = []
    i=1
    
    while i <= number:
        if number % i == 0:
            divisor = number//i        
            divisors.append(divisor)
            i += 1

        else:
            i += 1

    divisors.reverse()
    return divisors

def partitionate_array(array, box_size):
    """
    Partionate a given array into boxes of given size.

    Parameters
    ----------
    array: 2d array of integers or floats
        source array to partionate
    box_size: integer
        size of the boxes to partitionate the array in, has to be an integer divisor of the array shape to avoid border effects

    Returns
    -------
    windowed_array: numpy array of shape (array.shape[0], array.shape[1], box_size, box_size)
    """
    array_windowed = view_as_windows(array,(box_size,box_size))

    return array_windowed

def get_greycoprops4array2(array,window_size,n_grey_levels):
    from skimage.util import view_as_windows
    
    array_shape = array.shape
    out_shape = (2,array_shape[0],array_shape[1])
    out = {'contrast':np.zeros(out_shape),
           'homogeneity':np.zeros(out_shape),
           'ASM':np.zeros(out_shape),
           'correlation':np.zeros(out_shape)}
    
    array_windows = view_as_windows(array,(window_size,window_size))
    
    i = 0
    for x in np.arange(0,array_windows.shape[0]):
        for y in np.arange(0,array_windows.shape[1]):
            
            glcm = greycomatrix(array_windows[x,y],[1],[0,0.5*np.pi],n_grey_levels+1,symmetric=True,normed=True)

            for k in out.keys():
                value = greycoprops(glcm,k)

                out[k][0][x,y] = value[0][0]
                out[k][1][x,y] = value[0][1]
            update_progress(i / ((array_shape[0]-window_size) * (array_shape[1]-window_size)))    
            i += 1
    return out

# calculate cross correlation shift
def calc_cross_correlation_shift(source,target,cc_parameters=None):
    """
    Calculate the shift between two images using cross correlation given by
    scikit image's register translation.

    Parameters
    ----------
    source: 2d array of floats
            source image to look for in the target
    target: 2d array of floats
          target image in which to look for the source

    cc_parameters: dictionary
        parameters to use for the cross correlation algorithm, consits of
        'box_size' for the size of the box to calculate the shift for.
        'target_box_size' for the size of the target area to look for the 
        first box in and 'upsample_factor' to detect sub-pixel movements

    padding: int
        width of zero padding around the boxes to mitigate cross correlation
        artifacts
    
    Returns
    -------
    shift: 3d array
        row wise and column wise shift
    """
    
    # load cross correlation parameters if not defined
    if not cc_parameters:
        cc_parameters = tcf.cross_correlation_parameters

    # partitionate the arrays
    if cc_parameters['box_size'] == 0:
        cc_parameters['box_size'] = source.shape[0]
        #cc_parameters['target_box_size'] = source.shape[0]
    elif cc_parameters['box_size'] > source.shape[0]:
        print("Error: The given box size is larger than the arrays.")
        return

    # cutout source boxes
    source_windowed = partitionate_array(source, cc_parameters['box_size'])
    target_windowed = partitionate_array(target, cc_parameters['box_size'])

    # calculate shifts
    shift = np.zeros((source.shape[0],source.shape[1],2))

    for x in np.arange(0,source_windowed.shape[0]):
        for y in np.arange(0,source_windowed.shape[1]):
            
            ishift, _ , _ = register_translation(source_windowed[x,y],
                                                 target_windowed[x,y],cc_parameters['upsample_factor'])
            
            shift[:,:,0][x + cc_parameters['box_size']//2,
                     y + cc_parameters['box_size']//2] = ishift[0]
            shift[:,:,1][x + cc_parameters['box_size']//2,
                     y + cc_parameters['box_size']//2] = ishift[1]                

    return shift
    
def calc_cross_correlation_point_shift(point,source,target,cc_parameters=None,
								       padding=4,direction='+'):
	"""
	Calculate the shift derived by cross correlation tracking for a given point.
	
	Parameters
    ----------
    point: tuple of int or float, (row index , column index)
		   row and column coordinates of the starting point
    source: 2d array of floats
            source image to look for in the target
    target: 2d array of floats
          target image in which to look for the source
    direction: str, valid = ['forward','for','+',backward','bck','-']
			   indication of the direction to track into

    cc_parameters: dictionary
        parameters to use for the cross correlation algorithm, consits of
        'box_size' for the size of the box to calculate the shift for.
        'target_box_size' for the size of the target area to look for the 
        first box in and 'upsample_factor' to detect sub-pixel movements

    padding: int
        width of zero padding around the boxes to mitigate cross correlation
        artifacts
    
    Returns
    -------
    new_position: tuple, (row index, column index)
				  new position of the starting point
    """
    
	# load cross correlation parameters if not defined
	if not cc_parameters:
		cc_parameters = tcf.cross_correlation_parameters
        
	# calculate flow fields
	shift_fields = calc_cross_correlation_shift(source,target,cc_parameters,padding)
    
	# get integer coordinates of the point
	point_irow = np.clip(int(np.rint(point[0])),0,source.shape[0])
	point_icol = np.clip(int(np.rint(point[1])),0,source.shape[1])
    
	# move starting point
	row_shift = shift_fields[0][point_irow,point_icol]
	col_shift = shift_fields[0][point_irow,point_icol]
    
	if direction in ['forward','for','+']:
		new_row = point[0] + row_shift
		new_col = point[1] + col_shift
	elif direction  in ['backward','bck','-']:
		new_row = point[0] - row_shift
		new_col = point[1] - col_shift
		
	return (new_row,new_col)
