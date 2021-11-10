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

# split given arrays into sub-boxes
def partitionate_array(array, box_size):
    """
    Partionate a given array into boxes of given size.

    Parameters
    ----------
    array: 2d array of integers or floats
        source array to partionate
    box_size: integer
        size of the boxes to partitionate the array in, has to be an integer divisor of the array shape

    Returns
    -------
    splits: list of array   
        boxes of the source array
    idx_row: list of arrays
        row index coordinates of the boxes relative to the source array
    idx_col: list of arrays
        col index coordinates of the boxes relative to the source array
    """
    
    try:
        # get size of the boxes
        shape_row = array.shape[0]
        shape_col = array.shape[1]

        # get number of boxes to subdivide array into
        n_boxes_row = shape_row / box_size
        n_boxes_col = shape_col / box_size

        # create index set to track the parts
        index_set = np.meshgrid(np.arange(0,shape_row),
                                np.arange(0,shape_col),indexing='ij')

        # split array 
        splits_row = np.split(array, n_boxes_row,axis = 0)
        splits_row_id = np.split(index_set[0], n_boxes_row,axis = 0)
        splits_col_id = np.split(index_set[1], n_boxes_row,axis = 0)

        splits = []
        idx_row = []
        idx_col = []

        for i,sp in enumerate(splits_row):
            splits.extend(np.array_split(sp, n_boxes_col, axis=1))
            idx_row.extend(np.array_split(splits_row_id[i], n_boxes_col, axis=1))
            idx_col.extend(np.array_split(splits_col_id[i], n_boxes_col, axis=1))

        return splits, idx_row,idx_col
    except ValueError:
        int_divisors = get_integer_divisors(array.shape[0])
        print("The given box size is not an integer divisor of the array shape. Only {div} are valid box sizes".format(div=int_divisors))
        return

# calculate cross correlation shift
def calc_cross_correlation_shift(source,target,cc_parameters=None,padding=4):
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
        cc_parameters['target_box_size'] = source.shape[0]
    elif cc_parameters['box_size'] > source.shape[0]:
        print("Error: The given box size is larger than the arrays.")
        return
    elif cc_parameters['target_box_size'] > source.shape[0]:
        print("Error: The given target box size is larger than the arrays.")
        return

    # cutout source boxes
    source_boxes, source_rows, source_cols = partitionate_array(source,cc_parameters['box_size'])

    if 'box_size' == 'target_box_size':
        inflation = 0
    else:
        #inflation = (cc_parameters['target_box_size'] - cc_parameters['box_size']) // 2
        inflation = (cc_parameters['box_size'] - cc_parameters['target_box_size']) // 2

    # loop over all boxes and calculate the shift
    shift = np.zeros((2,source.shape[0],source.shape[1]))

    for i, source_box in enumerate(source_boxes):
        # pad source box with zeros to have the same size as the target box
        source_box = np.pad(source_box,(padding+inflation,padding+inflation),'constant')

        # create target cutout
        nrows = source_rows[i].shape[0]
        ncols = source_rows[i].shape[1]
        
        r1_min = np.min(source_rows[i]) - inflation
        r1_max = np.max(source_rows[i]) + inflation + 1
        c1_min = np.min(source_cols[i]) - inflation
        c1_max = np.max(source_cols[i]) + inflation + 1
        
        # check if target box extends over the source box margin and adjust accordingly
        if r1_min < 0:
            row_pad_min = padding - r1_min
            r1_min = 0
        else:
            row_pad_min = padding
        
        if r1_max > source.shape[0]:
            row_pad_max = padding + r1_max - source.shape[0]
            r1_max = source.shape[0]
        else:
            row_pad_max = padding
            
        if c1_min < 0:
            col_pad_min = padding - c1_min
            c1_min = 0
        else:
            col_pad_min = padding
        
        if c1_max > source.shape[1]:
            col_pad_max = padding + c1_max - source.shape[1]
            c1_max = source.shape[1]
        else:
            col_pad_max = padding        
        
        target_box = np.pad(target[r1_min:r1_max,c1_min:c1_max],
                           [(row_pad_min,row_pad_max),(col_pad_min, col_pad_max)],'constant')

        ishift = register_translation(source_box,
                                      target_box,
                                      cc_parameters['upsample_factor'],
                                      'real')[0]
        
        shift[0][source_rows[i],source_cols[i]] = ishift[0]
        shift[1][source_rows[i],source_cols[i]] = ishift[1]

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
