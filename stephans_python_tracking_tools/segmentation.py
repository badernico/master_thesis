#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:31:18 2020

@author: lenk

This file contains routines for segmentation.
"""

import xarray as xr
import scipy.ndimage as ndi
import operator as op
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt
from skimage import feature, filters, measure, morphology

from scipy import ndimage as ndi

import click

import networkx as nx
import pandas as pd

import operator

###############################################################################

def apply_hysteresis_threshold(image, low, high):
    """Apply hysteresis thresholding to `image`.
    This algorithm finds regions where `image` is greater than `high`
    OR `image` is greater than `low` *and* that region is connected to
    a region greater than `high`.
    Parameters
    ----------
    image : array, shape (M,[ N, ..., P])
        Grayscale input image.
    low : float, or array of same shape as `image`
        Lower threshold.
    high : float, or array of same shape as `image`
        Higher threshold.
    Returns
    -------
    thresholded : array of bool, same shape as `image`
        Array in which `True` indicates the locations where `image`
        was above the hysteresis threshold.
    Examples
    --------
    >>> image = np.array([1, 2, 3, 2, 1, 2, 1, 3, 2])
    >>> apply_hysteresis_threshold(image, 1.5, 2.5).astype(int)
    array([0, 1, 1, 1, 0, 0, 0, 1, 1])
    References
    ----------
    .. [1] J. Canny. A computational approach to edge detection.
           IEEE Transactions on Pattern Analysis and Machine Intelligence.
           1986; vol. 8, pp.679-698.
           DOI: 10.1109/TPAMI.1986.4767851
    """
    if hasattr(low,'__len__'):
        low = np.clip(low, a_min=None, a_max=high)  # ensure low always below high
        mask_low = np.array([image[i] > low[i] for i in range(len(image))])
        mask_high = np.array([image[i] > high[i] for i in range(len(image))])
        # Connected components of mask_low
        labels_low, num_labels = ndi.label(mask_low)
        # Check which connected components contain pixels from mask_high
        sums = ndi.sum(mask_high, labels_low, np.arange(num_labels + 1))
        connected_to_high = sums > 0
        thresholded = connected_to_high[labels_low]  
    else:
        low = np.clip(low, a_min=None, a_max=high)  # ensure low always below high
        mask_low = image > low
        mask_high = image > high
        # Connected components of mask_low
        labels_low, num_labels = ndi.label(mask_low)
        # Check which connected components contain pixels from mask_high
        sums = ndi.sum(mask_high, labels_low, np.arange(num_labels + 1))
        connected_to_high = sums > 0
        thresholded = connected_to_high[labels_low]
    
    return thresholded

def find_object_hysteresis_threshold(field_data,min_threshold,max_threshold):
    """
    Segmentation of a given data field using hysteresis thresholding to find 
    objects.
    
    Inputs
    ------
        * field_data: data to segment, 2d or 3d array (time on first axis)
                      numpy array
        * min_threshold: minimum threshold for segmentation
                         float or numpy array with floats
        * max_threshold: maximum threshold for segmentation
                         float or numpy array with floats
                         
    Returns                         
    -------
        * labeled segmented data 
        * maximum label
    """
    
    if hasattr(min_threshold,'__len__') > 1:
        masked_data = np.array([apply_hysteresis_threshold(field_data[i],
                                                           min_threshold[i],
                                                           max_threshold[i]) 
                                for i in range(len(field_data))])
    else:
        masked_data = apply_hysteresis_threshold(field_data,
                                                 min_threshold,
                                                 max_threshold)
    #labeled_data, nlabel_data = ndi.label(masked_data*1,structure=structure_element)
    labeled_data, nlabel_data = ndi.label(masked_data*1,structure=np.ones((3,3,3)))
    
    return labeled_data, nlabel_data

def find_object_hysteresis_threshold_2d(field_data,min_threshold,max_threshold):
    """
    Segmentation of a given data field using hysteresis thresholding to find 
    objects.
    
    Inputs
    ------
        * field_data: data to segment, 2d or 3d array (time on first axis)
                      numpy array
        * min_threshold: minimum threshold for segmentation
                         float or numpy array with floats
        * max_threshold: maximum threshold for segmentation
                         float or numpy array with floats
                         
    Returns                         
    -------
        * labeled segmented data 
        * maximum label
    """
  
    if hasattr(min_threshold,'__len__') > 1:
        masked_data = np.array([apply_hysteresis_threshold(field_data[i],
                                                           min_threshold[i],
                                                           max_threshold[i]) 
                                for i in range(len(field_data))])
    else:
        masked_data = apply_hysteresis_threshold(field_data,
                                                 min_threshold,
                                                 max_threshold)
    
    labeled_data = []
    nlabel_data = []
    
    for data in masked_data:
        ld, nl = ndi.label(data*1,structure=np.ones((3,3)))
        labeled_data.append(ld)
        nlabel_data.append(nl)
    
    return labeled_data, nlabel_data

def smooth_watershed(field,cloud_mask=None,threshold=0.2,smoothing_neighbourhood=15,intensity_smoothing=25,
                     spatial_smoothing=1,extreme_type='max',extrema_separation=4):
    """
    Segments a given field using a bilateral filter and watershed segmentation.
    
    Inputs:
    -------
        * field: numpy array, 2d or 3d
            field to be segmented
        * cloud_mask: numpy array, 2d or 3d, default=None
            NWCSAF cloud mask to use for the separation of foreground and background, if not given a 
            threshold is used
        * threshold: float, default=0.2
            threshold to use for the separation of foreground and background, 
        * smoothing_neighbourhood: int
            parameter of the bilateral filter, neighbourhood to smooth for
        * intensity_smoothing: int
            parameter of the bilateral filter, strength of the smoothing in intesity space
        * spatial_smoothing: int
            parameter of the bilateral filter, strength of the smoothing in spatial space
        * extreme_type: str
            type of extremes to look for, minima (min) or maxmia (mx)
        * extrema_separation: int
            minimum distance between two extrema used a starting points for the watershed segmentation
    Output:
    -------
        * segmented_field: numpy array, 2d or 3d
    """
    import numpy as np
    from cv2 import bilateralFilter
    from skimage.feature import peak_local_max
    from skimage.morphology import watershed
    
    valid_extremes = ['min','max']
    
    # smooth field
    field_smoothed = bilateralFilter(field.astype("float32"),
                                     smoothing_neighbourhood,
                                     intensity_smoothing,
                                     spatial_smoothing)
    
    # determine foreground and background
    if np.any(cloud_mask):
        field_masked = np.ma.masked_where(cloud_mask<2,field)
        
        if extreme_type in valid_extremes:
            if extreme_type == 'max':
                extreme_coordinates = peak_local_max(field_masked,min_distance=extrema_separation)
                
            elif extreme_type == 'min':
                extreme_coordinates = peak_local_max(-field_masked,min_distance=extrema_separation)
            else:    
                print("{} ist not a valid extreme type. Only 'min' and 'max' are valid.".format(extreme_type))
                return
    else:
        if extreme_type in valid_extremes:
            if extreme_type == 'max':
                # detect local maxima
                field_masked = np.ma.masked_less(field_smoothed,threshold)

                extreme_coordinates = peak_local_max(field_masked,min_distance=extrema_separation)
                
            elif extreme_type == 'min':
                # detect local minima
                field_masked = np.ma.masked_greater(field_smoothed,threshold)
                
                extreme_coordinates = peak_local_max(-field_masked,min_distance=extrema_separation)
            else:    
                print("{} ist not a valid extreme type. Only 'min' and 'max' are valid.".format(extreme_type))
                return
            
    # create array with markers for watershed segmentation
    markers = np.zeros_like(field_masked)
    
    for j,p in enumerate(extreme_coordinates):
        markers[p[0],p[1]] = j+1

    # perform watershed segmentation
    segmented_field = watershed(field_masked, markers, mask=~field_masked.mask)
    
    return segmented_field

def quantise_field(field,lower_limit,upper_limit,delta):
    """
    Quantises a given array into discrete steps within given limits and step width.
    
    Parameters
    ----------
    field: numpy array, 2d
        array to quantise
    lower_limit: float
        lower limit for the quantisation
    upper_limit: float
        upper limit for the quantisation
    delta: float
        width of the quantisation steps, if delta > 0 the quantisation will be based on the field maxima and if 
        delta < 0 on the field minima
    """
    
    quant = np.zeros_like(field)
    
    too_small = np.where(field <= lower_limit)
    too_large = np.where(field > upper_limit)
    in_range = np.where(np.logical_and(field > lower_limit,field <= upper_limit) )
    
    if delta < 0:
        quant[too_small] = np.rint((lower_limit-upper_limit) / delta)
        quant[too_large] = 0
        quant[in_range] = np.rint((field[in_range] - upper_limit) / delta)
        
        return quant
    elif delta > 0:
        quant[too_large] = np.rint((upper_limit-lower_limit)/delta)
        quant[too_small] = 0
        quant[in_range] = np.rint((field[in_range] - lower_limit) / delta)
        
        return quant
    else:
        print("Delta has to be either < 0 or > 0.")
        return
    
def quantisation_segmentation(field,lower_limit,upper_limit,delta):
    """
    Segment an array which is quantsied into descrete steps beforehand by 
    connected component labeling.
    
    Parameters:
        * field: array like
            filed to be segmented
        * lower_limit: float or int
            lower limit of the range of the filed to be quantised in
        * upper_limit: float or int
            upper limit of the range of the filed to be quantised in
        *delta: float
            width of the quantisation steps, if delta > 0 the quantisation will
            be based on the field maxima and if delta < 0 on the field minima
            
    Returns:
        * segmented_array: array like
            segmented array
    """
    
    # quantise array
    quantised_array = quantise_field(field,lower_limit,upper_limit,delta)
    
    # the lowest level usually corresponds to some kind of background
    masked_array = np.ma.masked_less(quantised_array,0)
    
    # segmentation
    segmented_array, nlabels = ndi.label(masked_array)
    
    
def find_peaks(field):
    """
    Find peaks in a given field, using morphological reconstruction,
    idea taken from http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_holes_and_peaks.html.
    
    Parameters:
        * field: array like
            field to find peaks in
            
    Returns:
        * peaks: array like
            peaks of the field
    """            
    
    from skimage.morphology import reconstruction
    
    # create seed to find peaks
    seed = np.copy(field)
    seed[1:-1,1:-1] = field.min()
    
    # create mask
    mask = field
    
    # create dilation
    rec = reconstruction(seed,mask, method='dilation')
    
    peaks = field - rec
    
    return peaks

def find_holes(field):
    """
    Find holes in a given field, using morphological reconstruction,
    idea taken from http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_holes_and_peaks.html.
    
    Parameters:
        * field: array like
            field to find holes in
            
    Returns:
        * holes: array like
            holes of the field
    """            
    
    from skimage.morphology import reconstruction
    
    # create seed to find holes
    seed = np.copy(field)
    seed[1:-1,1:-1] = field.max()
    
    # create mask
    mask = field
    
    # create erosion
    filled = reconstruction(seed,mask, method='erosion')
    
    holes = field - filled
    
    return holes

def segment_morphological_reconstruction(field, extreme='min', threshold=None):
    """
    Segments a given field by using morphological reconstruction.
    
    Parameters:
        * field: array like
            field to segment
        * extreme: string, 'min' or 'max'
            kind of the extremes to look for in the field
        * threshold: float or int, default = None
            threshold used to separate foreground and background, if none is given
            the threshold will be selected dynamically
        
    Returns:
        * field_labeled: array like
            segmented and labeled field
    """
    
    valid_extremes = ['min','-','max','+']
    
    if extreme in valid_extremes:
        if extreme in valid_extremes[:2]:
            field_transformed = find_holes(field)
            
            if not threshold:
                threshold = np.percentile(field_transformed,0.99)
                
            field_segmented = np.zeros_like(field)
            field_segmented[field_transformed <= threshold] = 1
               
        elif extreme in valid_extremes[2:]:
            field_transformed = find_peaks(field)
                
            if not threshold:
                threshold = np.percentile(field_transformed,0.99)
                
            field_segmented = np.zeros_like(field)
            field_segmented[field_transformed >= threshold] = 1
                
        else:
            print("{} is not a valid extreme identifier. Try using 'min' or 'max'.".format(extreme))
            return
        
        field_labeled, nlabel = ndi.label(field_segmented)
        
        return field_labeled
    else:
        print("{} is not a valid extreme identifier. Try using 'min' or 'max'.".format(extreme))
        return


def segmentation_dilation(field_data,threshold,gauss_sigma=0,dilation_size=3,op=operator.gt):
    """
    Segmentation of a given probably smoothed field using a given threshold. 
    The segmentation is performed using connected component labeling field 
    dilated with the given dilation size.

    Inputs:
        * field_data: array like, 2d or 3d, dim = (time,x,y)
            data to segment
        * threshold: int or float
            threshold used to sparate foreground and background
        * gauss_sigma: int, default = 0
            smoothng factor of the gaussian smoothing
        * dilation size: int, default = 3
            size to dilate the field with
        * op: operator of operator library, default: operator.gt
            operator to use for the segmentation

    Returns:
        * fields_labelled_dil: array like, shape of input field_data
            labelled and dilated given field
    """
    # smoothing of the given filed
    field_smoothed = ndi.gaussian_filter(field_data,gauss_sigma)

    # thresholding
    field_thresholded = op(field_smoothed,threshold)

    # labelling
    fields_labelled = []
    labels = []

    for field in field_thresholded:
        field_labelled,l = ndi.label(field)
        fields_labelled.append(field_labelled)
        labels.append(l)

    # dilation of the labelled fields
    fields_labelled_dil = np.array([ndi.morphology.grey_dilation(fl,size=(dilation_size,dilation_size))
                                   for fl in fields_labelled])

    return fields_labelled_dil

# segment a given field with local thresholds ---------------------------------
def segment_local_threshold(field,minimum_depth = 10,tmax = 273.15, tmin = 220,
                            thresh_min = 240, spread = 5,sigma=0,tlimit=300):
    """
    Segment a given field using local minima and threshold derived from them.

    Inputs:
        * field: array-like, float or int, 2d or 3d
            array to find the local minima in
        * minimum depth: float or int, default = 10
            depth a local minimum has to have to be considered
        * tmax: float or int, default = 273.15
            maximum temperature to consider as an object minimum
        * tmin: float or int, default = 220
            minimum temperature for the calculation of the threshold
        * thresh_min: float or int, default = 240
            minimum possible value for the object boundary
        * spread: float or int, default = 5
            spread value to calculate the object boundary from the local min
        * sigma: float or int, default = 0
            sigma of a possible Gaussian smoothing if the input field
        * tlimit: float or int, default = 300
            temperature threshold until which local minima are considered
    * tlimit: float or int, default = 260
        maximum temperature a local minimum is allowed to have to be considered

    Returns:
        * segmented_field: array-like, same shape as field
            field with labeled areas
    """

    # find local minima
    local_minima_field, nlabels = get_labeled_local_minima(field,minimum_depth,sigma)

    # get individual labels of the localminima
    minima_label = np.arange(1,nlabels)

    # create dictionary to store the values of the local minima
    minima_values = {ml:[] for ml in minima_label}
    
    # collect the position and the values of the local minima
    for ml in minima_label:
        ml_location = np.where(local_minima_field==ml)
    
        if np.all(local_minima_field[ml_location[0][0],ml_location[1][0]] < tlimit): 
            minima_values[ml] = [ml_location[0][0], 
                                 ml_location[1][0],
                                 field[ml_location][0],
                                 local_threshold(field[ml_location][0])]
        else:
            continue

    # calculate the local threshold for the local minima and store them
    segmented_field = np.zeros_like(field,dtype="uint16")
    
    for local_minimum in minima_values.keys():
        if np.isnan(minima_values[local_minimum][3]):
            continue
        else:
            # create masks with areas which inbetween the thresholds
            lower_mask = np.ma.masked_less_equal(field,minima_values[local_minimum][3])
            upper_mask = np.ma.masked_greater_equal(field,minima_values[local_minimum][2])
            
            #threshold_mask = np.ma.masked_less_equal(field,
            #                                         minima_values[local_minimum][3]).mask*1 
    
            threshold_mask = upper_mask.mask & lower_mask.mask
            
            # fill possible holes in the threshold_mask
            threshold_mask = ndi.binary_fill_holes(threshold_mask)
            
            # label those areas
            threshold_mask_labeled = ndi.label(threshold_mask*1)[0]
    
            # get label for the wanted object
            obj_id = np.unique(threshold_mask_labeled[minima_values[local_minimum][0],
                                                      minima_values[local_minimum][1]])#[0]
    
            # get the points of this object
            obj_location = np.where(threshold_mask_labeled==obj_id)
            
            # check, if there is already an object
            segment_values = np.unique(segmented_field[obj_location])
            
            # if all are zero, we simply label these values as belonging to the object
            if np.all(segment_values == 0):
                segmented_field[obj_location] = local_minimum
            # if not we only label those values, which are zero
            else:
                zero_loc = np.where(segmented_field[obj_location]==0)
                loc = (obj_location[0][zero_loc],obj_location[1][zero_loc])
                segmented_field[loc] = local_minimum

    # return segmented field
    return segmented_field

# Object definition with local minima and local threshold =====================
# object definition by local threshold ----------------------------------------
def local_threshold(t, tmax = 273.15, tmin=220,thresh_min = 240, spread = 5):
    """
    Calculate threshold for a given local minimum.

    Inputs:
        * t: float or int
            temperature of the local minimum
        * tmax: float or int, default = 273.15
            maximum temperature to consider as an object minimum
        * tmin: float or int, default = 220
            minimum temperature for the calculation of the threshold
        * thresh_min: float or int, default = 240
            minimum possible value for the object boundary
        * spread: float or int, default = 5
            spread value to calculate the object boundary from the local min
    Returns:
        * T_thresh: float or int
            threshold temperature for the object boundary  
    """

    # calculate factor for spread term
    k  = t - tmin - spread

    # calculate threshold based on local minimum value
    T_thresh = np.max((t + spread,
                       t + spread + k*(t - tmax)/ (tmin - tmax),
                       thresh_min))
    return T_thresh

# find local minima and label them --------------------------------------------
def get_labeled_local_minima(field,minimum_depth=10,gauss_sigma=1):
    """
        Find local minima with a fiven depth and label them.

        Inputs:
            * field: array-like, float or int, 2d or 3d
                array to find the local minima in
            * minimum depth: float or int, default = 10
                depth a local minimum hast ot have to be considered
        * gauss_sigma: int or float
            sigma paramter of Gaussian smoothing
        Returns:
            array with labeled minima and number of labels
    """
    # smooth input field using a 2d gaussian filter
    field = ndi.gaussian_filter(field,gauss_sigma)    

    # find minima
    minima = morphology.h_minima(field,minimum_depth)
    
    # return labeled minima
    return ndi.label(minima)


# segementation using local minima and watershed transformation ---------------
def watershed_local_min_segmentation(data_array, depth=6, tmin=220, tmax=273.15, tlevel=240,spread=5,smoothing_factor=1):
    """
    Segmentation using local minima extended by watershed segmentation to fix some issues.

    Inputs:
        * data_arry: array-like, 2d
            array from which the local minima should be derived
        * depth: int
            depth for h_minima, depth a local minimum has to have to be considered
        * tmin: float or int, default: 220
            minimum temperature for spread in segementation using local thresholds
        * tmax: float or int, default: 273.15
            maximum temperature for spread in segementation using local thresholds
        * tlevel: float or int, default: 240
            leveling temperature for spread in segementation using local thresholds
        * spread: float or int, default: 5
            spread for segementation using local thresholds
    """
    # gaussian smoothing of data array
    data_array_smooth = filters.gaussian(data_array,smoothing_factor)
    
    # find local minima
    local_mins = morphology.h_minima(data_array_smooth, depth)
    
    # label the local minima
    lmins_labeled = ndi.label(local_mins)[0]
    
    # collect local values, threshold values and locations of the local minima
    #lmin_properties = get_lmin_properties(lmins_labeled,data_array)

    # get mask with possible object locations
    mask = segment_local_threshold(data_array_smooth,
                                   depth,tmax,tmin,tlevel,spread,
                                   smoothing_factor,300)
    mask = np.ma.masked_greater(mask,0).mask
    #mask = create_mask_from_thresholds(lmins_labeled,lmin_properties,data_array)

    # create mask using Otsu's approach to fix some problems
    #otsu_mask = np.ma.masked_greater(data_array,
    #                               filters.threshold_otsu(data_array))

    # combine masks
    #mask = mask & ~otsu_mask.mask
    
    # perform watershed transformation with the local minima as markers using the mask
    watershed_segmentation = morphology.watershed(data_array, lmins_labeled, mask=mask, watershed_line=False)
    
    # return segmented field
    return watershed_segmentation

def get_lmin_properties(labeled_local_minima,data_array,tmin=220,tmax=273.15,tlevel=240,spread=5):
    """
    Derive properties of local minima.

    Inputs:
        * labeled_local_minima: array-like, 2d, int
            numpy array with local minima labeled by a unique number
        * data_array: array-like, same shape as labeled_local_minima
            array to derive values from
        * tmin: float or int, default: 220
            minimum temperature for spread in segementation using local thresholds
        * tmax: float or int, default: 273.15
            maximum temperature for spread in segementation using local thresholds
        * tlevel: float or int, default: 240
            leveling temperature for spread in segementation using local thresholds
        * spread: float or int, default: 5
            spread for segementation using local thresholds

    Returns:
        * properties: dictionary with value, threshold, column and row
            properties of the local minimum
    """

    # extract labels of local minima
    lmin_labels = np.unique(labeled_local_minima)[1:]
    
    # collect local values, threshold values and location of the local minima
    properties = {label:{'value':[],
                         'threshold':[],
                         'column':[],
                         'row':[]} for label in lmin_labels}

    for label in lmin_labels:
        label_loc = np.where(labeled_local_minima==label)
        
        if label_loc[0].size > 1:
            label_loc = (np.array([np.min(label_loc[0])]), np.array([np.min(label_loc[1])]))
            
        thresh = local_threshold(data_array[label_loc],tmax,tmin,tlevel,spread)

        if type(thresh) == int or type(thresh) == float:
            properties[label]['threshold'] = thresh
        else:
            properties[label]['threshold'] = thresh[0]
        
        properties[label]['value'] = data_array[label_loc][0]
        properties[label]['column'] = label_loc[0][0]
        properties[label]['row'] = label_loc[1][0]
        
    return properties

def create_mask_from_thresholds(labeled_local_minima,lmin_properties,data_array):
    """
    Create mask from local threshold to segment after.

    Inputs:
        * labeled_local_minima: array-like, 2d, int
            numpy array with local minima labeled by a unique number
        * lmin_properties: dictionary
            properties of the local minima: value, threshold, column, row
        * data_arry: array-like, same shape as labeled_local_minima
            array from which the local minima have been derived

    Returns:
        * label_mask: array-like, in
            array with locations of possible objects
    """
    label_masks = [ ]
        
    for label in lmin_properties.keys():
        # mask where data values are below threshold
        mask = np.zeros_like(labeled_local_minima)
        mask = np.ma.masked_where(data_array < lmin_properties[label]['threshold'],mask)

        # label those areas
        m_labeled = ndi.label(mask.mask*1)[0]
        min_label = m_labeled[lmin_properties[label]['column'],lmin_properties[label]['row']]

        # select that area, where the local minimum is located inside
        omask = np.full_like(data_array,False)
        omask[np.where(m_labeled==min_label)] = True

        # append to list of masks
        label_masks.append(omask)
        
    # combine masks
    mask_sum = np.sum(label_masks,axis=0)

    # run through all levels in the combined mask and check for local minima
    lmin_mask = np.ma.masked_greater(labeled_local_minima,0).mask
    level_masks = []
    for level in np.unique(mask_sum)[1:]:
        level_mask = np.ma.masked_equal(mask_sum,level).mask
   
        level_labeled = ndi.label(level_mask*1)[0]

        level_objects = np.zeros_like(level_mask)

        # check, which local minima are located within objects at this level
        represented_lmins = lmin_mask & level_mask

        retained_objects = level_labeled[np.where(represented_lmins==True)]

        for ro in retained_objects:
            level_objects[np.where(level_labeled==ro)]=1

        level_masks.append(level_objects)
    
    label_mask =  np.ma.masked_greater(np.sum(level_masks,axis=0),0).mask
    
    # return combined mask
    return label_mask
