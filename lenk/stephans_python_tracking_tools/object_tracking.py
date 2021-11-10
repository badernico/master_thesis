#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 14:27:18 2018

@author: lenk

This file contains routines used for the object tracking.
"""

import xarray as xr
import operator as op
import numpy as np
import pandas as pd
import networkx as nx
import tqdm

import matplotlib.pyplot as plt

from skimage import feature, filters, measure, morphology
import scipy.ndimage as ndi

import click

import segmentation as seg

from standard_config import *

import sys
sys.path.append("{}/utils/tracking".format(local_home_path))
import optical_flow_tracking as oft
import tracking_common as tco

###############################################################################

# graph routines ###############################################################
def split_complex_graphs(graph_list, min_node_number = 5 ):
    
    """
    Sorts Graphs into isolated and complex graphs.
    
    Parameters
    ----------
    graph_list: list of networkx graphs
        list with the graphs to be filtered
    min_node_number: int
        minimum number of nodes a graph has to have to be considered
    
    
    Returns
    -------
    filtered_list: list of networkx graphs
        filtered list of the graphs
        
    """
    
    isolated_graphs = []
    
    click.echo(click.style('Filtering {} graphs, this make take a while...'.format(len(graph_list)),
                           fg='white',bg='blue'))
    
    with click.progressbar(graph_list,length=len(graph_list), label='Splitting complex graphs') as bar:
        for i, graph in enumerate(graph_list):
            bar.update(i)

            # all edges are remove for split and mergers
            splitted_graphs = split_and_merge_identification(graph, edge_removal = True)
            
            
            # again connect component analysis
            subgraphs = list(nx.connected_component_subgraphs( splitted_graphs ))

            isolated_subgraphs, complex_subgraphs = sort_graphs( subgraphs, min_node_number = min_node_number )
            
            #print len( complex_subgraphs )
            
            isolated_graphs += isolated_subgraphs 
            
    return isolated_graphs

def split_and_merge_identification(g_in, edge_removal = False):

    '''
    Identifies splits and merges as well as inits and end points.
    
    It set true, all edges of split and merge events are removed.
    
    
    Parameters
    -----------
    g_in : networkx graph object
        object graph with temporal connections
        
        
    Returns
    --------
    g : networkx graph object, optional if edge_removal = True
        same as g_in, but with all merge / split connections removed
        
    '''

    g = g_in.copy()
    
    itimes = nx.get_node_attributes(g, 'itime')
    node_category = {}
    edge_set = []

    for nodename in sorted( g.nodes() ):
        n = g.node[nodename]
    
        node_neighbors = g.neighbors(nodename)
    
        # get time of node
        itime0 = itimes[nodename]
    
        # we get the time index of the neighbors
        itime_neighbors = [itimes[nname] for nname in node_neighbors]
        itime_neighbors = np.array( itime_neighbors )
    
        # we check if we have one or two times
        # and we check the multiplicity for split and merge analysis
        timeset, number_of_connections = np.unique( itime_neighbors, return_counts=True )
    
        number_of_times = len(timeset)
    
        # are we at beginning, in-between or at the end?
        cat_string = ''
        if number_of_times == 1:
            if itime0 > timeset[0]:
                # end point
                cat_string += 'e'
        
                if number_of_connections > 1:
                    # final merge
                    cat_string += 'm'
    
            elif itime0 < timeset[0]:
                # starting point
                cat_string += 'i'
            
                if number_of_connections > 1:
                    # init split
                    cat_string += 's'
         
        # we are in-between
        else:
            if number_of_connections[0] > 1:            
                # in-between merge
                cat_string += 'm'
        
            if number_of_connections[1] > 1:
                # in-between split
                cat_string += 's'
    
        node_category[nodename] = cat_string 
        
        # prepare edge removal if wanted

        # for splits
        if 's' in cat_string:
            
                
            # get set of neighbors that are part of split
            neighbor_set =  np.array(node_neighbors)[ itime_neighbors == timeset[-1]]
                
            # combine it with the "mother" node name
            edge_set += [ (nodename, neighborname) for neighborname in neighbor_set ]

        # for merges
        if 'm' in cat_string:
            
                
            # get set of neighbors that are part of split
            neighbor_set =  np.array(node_neighbors)[ itime_neighbors == timeset[0]]
                
            # combine it with the "mother" node name
            edge_set += [ (nodename, neighborname) for neighborname in neighbor_set ]
    
    
    if edge_removal:
        # remove edges
        g.remove_edges_from(edge_set)

    nx.set_node_attributes(g, 'node_category', node_category)   
        
    return g

def sort_graphs( graph_list, min_node_number = 5 ):
    """
    Sorts Graphs into isolated and complex graphs.
    
    Parameters
    ----------
    graph_list: list of networkx graphs
        list with the graphs to be filtered
    min_node_number: int
        minimum number of nodes a graph has to have to be considered
    
    
    Returns
    -------
    filtered_list: list of networkx graphs
        filtered list of the graphs
        
    """
    
    isolated_graphs = []
    complex_graphs = []
    
    click.echo(click.style('Filtering {} graphs, this make take a longer while...'.format(len(graph_list)),
                           fg='white',bg='blue'))
    
    with click.progressbar(graph_list,length=len(graph_list), label='Sorting graphs') as bar:
        for i, graph in enumerate(graph_list):
            bar.update(i)
            
            
            # first, check if only graphs with isolated objects should be considered
            if graph.number_of_nodes() >= min_node_number:
                time_vector = []
                    
                for nodename in graph.nodes():
                    node = graph.node[nodename]
                    time = node['itime']
                    n_nodes = graph.number_of_nodes()
                    
                    time_vector.append(time)
                    
                if n_nodes == len(np.unique(time_vector)):
                    isolated_graphs.append( graph )
                else:
                    complex_graphs.append( graph ) 

            
    return isolated_graphs, complex_graphs

def get_attributes_from_graph(graph):
    """
    Gets the centroid positions of the graph nodes out of the node attributes.
    
    Parameters
    ----------
    graph: networkx graph
        graph to extract the node positions from
        
    Returns
    -------
    positions: list
        list with the node positions
        
    """
    
    attributes = {k:[] for k in graph.node[graph.nodes()[0]].keys()}
     
    # get the time step and nodes per time step
    for nodename in sorted(graph.nodes()):
        node = graph.node[nodename]
        
        for k in node.keys():
            attributes[k].append(node[k])
            
    return attributes

def sort_graph_by_timesteps(graph):
    """
    Sorts graph nodes according to their attached time steps.
    
    Parameters
    ----------
    graph: networkx graph
        networkx graph with time step information attached to it in variable itime
        
    Returns
    -------
    time_sorted_graph: dictionary
        graph sorted according to the time steps
    """
    
    time_sorted_graph = {}
    
    # get the time step and nodes per time step
    for nodename in graph.nodes():
        node = graph.node[nodename]
        
        if not node['itime'] in time_sorted_graph:
            time_sorted_graph[node['itime']] = []
            time_sorted_graph[node['itime']].append(nodename)
        else:
            time_sorted_graph[node['itime']].append(nodename)
        
    return time_sorted_graph

# object properties ############################################################
def get_object_centroid_from_area(clusters,data,object_id,threshold,median=True):
    """
    Derive minimum position to add to the graph.
    
    Parameters
    ----------
    clusters: numpy array, 2d or 3d, int
        labeled data
    data: numpy array, 2d or 3d, float
        data of which the clusters have been derived
    itime: int
        index of the desired timestep
    object_id: int
        object index of the desired object
    threshold: float
        threshold to define are of which to get the centroid
    median: boolean
        switch whether to use median or minimum and maximum to derive the 
        centroid position
   
    Returns
    -------
    ocol: float
        column index of the object centroid
    orow: float
        row index of the object centroid
    """
    cluster_masked = np.ma.masked_not_equal(clusters,object_id)
    data_masked = np.ma.masked_where(cluster_masked.mask==True,data)
        
    oidx = np.where(data_masked < threshold)
            
    if not median:
        orow = (np.min(oidx[0]) + np.max(oidx[0])) / 2.
        ocol = (np.min(oidx[1]) + np.max(oidx[1])) / 2.
    else:
        orow = np.median( oidx[0] )
        ocol = np.median( oidx[1] )    
    
    return (ocol,orow)

def get_object_centroid(clusters,itime,object_id, median = True):
    """
    Derive centroid position to add to the graph.
    
    Parameters
    ----------
    clusters: numpy array, 2d or 3d, int
        labeled data
    itime: int
        index of the desired timestep
    object_id: int
        object index of the desired object
    median: boolean
        switch whether to use median or minimum and maximum to derive the 
        centroid position
        
    Returns
    -------
    ocol: float
        column index of the object centroid
    orow: float
        row index of the object centroid
    oidx: numpy array, 2d
        position indices of the object points
    """
    
    oidx = np.where(clusters[itime] == object_id)
    
    if not median:
        orow = (np.min(oidx[0]) + np.max(oidx[0])) / 2.
        ocol = (np.min(oidx[1]) + np.max(oidx[1])) / 2.
    else:
        orow = np.median( oidx[0] )
        ocol = np.median( oidx[1] )
    
    return (ocol,orow,oidx)

def get_object_properties(clusters,data,itime,object_id, median = True):
    """
    Derive object proeprties to add to the graph.
    
    Parameters
    ----------
    clusters: numpy array, 2d or 3d, int
        labeled data
    data: numpy array 2d or 3d, float
        array to derive properties from
    itime: int
        index of the desired timestep
    object_id: int
        object index of the desired object
            
    Returns
    -------
    min: float
        minimum intensity of the object 
    max: float
        maximum intensity of the object
    mean: float
        mean intensity of the object
    median: float
        median intensity of the object
    object_locations: 2d numpy array
        locations of the object pixels
    """
    
    oidx = np.where(clusters[itime] == object_id)
    
    values = data[oidx]
    object_min = np.min(values)
    object_max = np.max(values)
    object_mean = np.mean(values)
    object_median = np.median(values)
          
    return (object_min,object_max,object_mean,object_median,oidx)
    
def get_object_sizes(labeled_fields,time_sorted_graph):
    """
    Determine the sizes of given objects.
    
    Parameters:
        * labeled_fields: 3d array of integers
            fields with the objects belonging to the graph
        * time_sorted_graph: dictionary
            dictionary with the time steps and the nodes belonging to the object
            and the time step
    Returns:
        * path_nodes: list
            with graph nodes with largest size
    """
    
    path_nodes = []

    for t in time_sorted_graph.keys():
        elements = []
        sizes = []
        
        for element in time_sorted_graph[t]:
            obj_id = int(element.split("_")[1])
            
            obj_size = len(np.where(labeled_fields[t]==obj_id)[0])
            
            sizes.append(obj_size)
            elements.append(element)

        max_id = np.where(np.array(sizes)== np.max(np.array(sizes)))
        
        path_nodes.append(elements[max_id[0][0]])
        
    return path_nodes
    
def get_bounding_box(labelled_fields,path_nodes):
    """
    Get the bounding boxes of a desired object in an labelled array.
    
    Parameters:
        labeled_fields: 3d or 2d array of integers
            fields to look for the object in
        path_nodes: list of strings
            list with the object identifiers for the time steps to find the
            right object
    
    Returns:
        bounding_boxes: dictionary
            dicitonary with the relative time steps and the bounding box 
            coordinates    
    """
    
    # intialise the boxes
    bounding_boxes = {t:[] for t in np.arange(0,len(labelled_fields),1)}

    for element in path_nodes:
        time_id = int(element.split("_")[0])
        obj_id = int(element.split("_")[1])
        bbox = ndi.find_objects(labelled_fields[time_id]==obj_id)[0]

        bounding_boxes[time_id] = bbox
        
    return bounding_boxes

def create_bbox_dataframe(bboxes,centroids,times,lon_data,lat_data,time_data):
    """
    Create pandas dataframe out of some given bounding box data.
    
    Inputs:
        * bboxes: dictionary {time:row_slice,column_slice}
            dictionary with time the bounding box coordinates
        * centroids: dictionary {time:(row coordinate,column coordinate)}
            dictionary with the centroid coordinates of the bounding boxes
        * times: array like
            relative time steps of the bounding boxes
        * lon_data: array like, 2d
            longitude coordinates of the cutout pixels
        * lat_data: array like, 2d
            latitude coordinates of the cutout pixels
        * time_data: array like
            UTC time belonging to the relative time steps
            
    Returns:
        * bbox_df: pandas data frame
              pandas data frame of the input data
    """
    
    col_min = []
    col_max = []
    row_min = []
    row_max = []
    itime = []
    time = []
    cent_row = []
    cent_col = []
    
    lon_min = []
    lon_max = []
    lat_min = []
    lat_max = []
    cent_lon = []
    cent_lat = []
    
    for t_idx in bboxes.keys():
        if len(bboxes[t_idx]) != 0:
            bbox = bboxes[t_idx]
            col_min.append(bbox[0].start)
            col_max.append(bbox[0].stop)
            
            row_min.append(bbox[1].start)
            row_max.append(bbox[1].stop)
            
            itime.append(times[t_idx])
            time.append(time_data[t_idx])
            
            lon_min.append(lon_data[bbox[0].start,bbox[1].start])
            lon_max.append(lon_data[bbox[0].stop,bbox[1].stop])
            
            lat_min.append(lat_data[bbox[0].start,bbox[1].start])
            lat_max.append(lat_data[bbox[0].stop,bbox[1].stop])
            
            cent_row.append(centroids[t_idx][1])
            cent_col.append(centroids[t_idx][0])
            
            cent_lon.append(lon_data[int(np.rint(centroids[t_idx][1])),
                                     int(np.rint(centroids[t_idx][0]))])
            cent_lat.append(lat_data[int(np.rint(centroids[t_idx][1])),
                                     int(np.rint(centroids[t_idx][0]))])
        else:
            continue
        
    bbox_df = pd.DataFrame({'itime':itime,'time':time,
                            'col_min':col_min,'col_max':col_max,
                            'row_min':row_min,'row_max':row_max,
                            'lon_min':lon_min,'lon_max':lon_max,
                            'lat_min':lat_min,'lat_max':lat_max,
                            'centroid_row':cent_row,
                            'centroid_col':cent_col,
                            'centroid_lon':cent_lon,
                            'centroid_lat':cent_lat})
    
    return bbox_df

def get_bounding_boxes_from_dict(field_data,extreme,threshold):
    """
    Derive bounding box data of objects in given fields.
    
    Inputs:
        * field_data: dictionary of arrays
            data to find the objects in and to derive the bounding boxes from
        * extreme: string, 'min' or 'max'
            identifier for which exreme values to look for in the segmentation
            process
        * threshold: int
            theshold to separate background and foreground during the segmentation
            process
    
    Returns:
        * bboxes: dictionary {time:row_slice,column_slice}
            dictionary with tiem steps and bounding box pixel coordinates
    """
    # segment data
    t_ref = field_data.keys()
    t_ref.sort()
    
    data_segmented = np.array([seg.segment_morphological_reconstruction(field_data[t],extreme,threshold)
                               for t in t_ref])  
    
    # connect the objects of all time steps
    connections, objects = generate_graph_from_components(data_segmented,
                                                             np.arange(0,len(data_segmented),1))
    
    # determine the interesting object, it should be in the centre of the 
    # centre time step
    wanted_id = get_wanted_object_id(data_segmented)
    target_object_id = '{}_{:04d}'.format(len(data_segmented)//2,wanted_id)
    
    # look for the graph wich cointains this object
    wanted_object_graph = get_wanted_graph(objects,target_object_id)
    
    # it may be that this is a complex graph, but ideally we only want the
    # path of a single cell
    wanted_object_graph_sorted = sort_graph_by_timesteps(wanted_object_graph)
    
    object_track = get_main_track(data_segmented,wanted_object_graph_sorted,target_object_id)
        
    # get the bounding boxes
    bboxes = get_bounding_box(data_segmented,object_track)
    
    return bboxes

def get_bounding_boxes(field_data,extreme,threshold):
    """
    Derive bounding box data of objects in given fields.
    
    Inputs:
        * field_data: array like, 2d or 3d
            data to find the objects in and to derive the bounding boxes from
        * extreme: string, 'min' or 'max'
            identifier for which exreme values to look for in the segmentation
            process
        * threshold: int
            theshold to separate background and foreground during the segmentation
            process
    
    Returns:
        * bboxes: dictionary {time:row_slice,column_slice}
            dictionary with tiem steps and bounding box pixel coordinates
    """
    # segment data
    data_segmented = np.array([seg.segment_morphological_reconstruction(d,extreme,threshold)
                                        for d in field_data])
    
    # connect the objects of all time steps
    connections, objects = generate_graph_from_components(data_segmented,
                                                             np.arange(0,len(data_segmented),1))
    
     # determine the interesting object, it should be in the centre of the 
    # centre time step
    wanted_id = get_wanted_object_id(data_segmented)
    target_object_id = '{}_{:04d}'.format(len(data_segmented)//2,wanted_id)
    
    # look for the graph wich cointains this object
    wanted_object_graph = get_wanted_graph(objects,target_object_id)
    
    # it may be that this is a complex graph, but ideally we only want the
    # path of a single cell
    wanted_object_graph_sorted = sort_graph_by_timesteps(wanted_object_graph)
    
    object_track = get_main_track(data_segmented,wanted_object_graph_sorted,target_object_id)
        
    # get the bounding boxes
    bboxes = get_bounding_box(data_segmented,object_track)
    
    return bboxes

def get_centroid_from_bb(bbox_data):
    """
    Get centroid of bounding boxes.
    
    Inputs:
        * bbox_data: dictionary
            corner coordinates of the bounding boxes
    Returns:
        * centoroids:
            centroid coordinates for the bounding boxes
    """
    centroids = {k:[] for k in bbox_data.keys()}
    for k in bbox_data.keys():
        if len(bbox_data[k]) > 0:
            cent_col = bbox_data[k][0].start + (bbox_data[k][0].stop - bbox_data[k][0].start) / 2.
            cent_row = bbox_data[k][1].start + (bbox_data[k][1].stop - bbox_data[k][0].start) / 2.
            
            centroids[k] = (cent_col,cent_row)
        else:
            continue
        
    return centroids

def bb_intersection_over_union(boxA, boxB):
    """
    Calcualte intersection over union for two bounding boxes.
    
    Inputs:
        * boxA: coordinate tuple (row_min, col_min, row_max, col_max)
              coordinates of the first bounding box  
        * boxB: coordinate tuple (row_min, col_min, row_max, col_max)
              coordinates of the second bounding box
    Returns:
        * iou: float
              intersection over union of the two bounding boxes
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
         
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
         
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
         
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
         
    # return the intersection over union value
    return iou

def pixel_intersection_over_union(object_a, object_b):
    intersection = np.logical_and(object_a, object_b)
    union = np.logical_or(object_a,object_b)
    
    return np.where(intersection==1)[0].size / float(np.where(union==1)[0].size)

def iou_difference(iou0,iou1):
    return iou1 - iou0
    
# create ellipses around minima to track them ---------------------------------
def get_track_ellipses(field,ellipsis_width=5,ellipsis_height=9,
                       ellipsis_rotation=0):
    """
        Create ellipses around given points
    
        Inputs:
            * field: array_like, 2d or 3d
                array with labeled points to create ellipses around
            * ellipsis_width: int, default = 5
                width of the ellipsis
            * ellipsis_height: int, default = 9
                height of the ellipsis
            * ellipsis_rotation: int
                rotation of the ellipsis
        Returns:
            * ellipses_filed: array-like, same shape as field
                array with ellipses
    """
    from skimage.draw import ellipse

    # create dummy field to fill
    ellipses_field = np.zeros_like(field, dtype="uint8")
    
    # check if field is labeled, if not, label it
    if np.max(field) == 1:
        field = ndi.label(field)[0]
    
    # run through all unique poits and create ellipses
    for l in np.unique(field):
        min_loc = np.where(field == l)
    
        rr, cc = ellipse(min_loc[0][0],
                         min_loc[1][0],
                         ellipsis_height, 
                         ellipsis_width, 
                         rotation=ellipsis_rotation)
        
        rr = np.clip(rr,0,field.shape[0]-1)
        cc = np.clip(cc,0,field.shape[1]-1)
        ellipses_field[rr,cc] = l
    
    ellipses_field = ~np.ma.masked_less(ellipses_field,1).mask*1
    
    return ellipses_field   

# get properties in the neighbourhood of a given point
def lmin_ellipses(field,depth=10,sigma=1,width=3,height=3):
    """
    Finds local minima in a given field and returns elliptical areas
    around that minima definied by width and height.

    Inputs:
	* field: array-like
	   Field to find the local minima in
	* depth: int or float
	   Depth a local minimum has to have to be considered
	* sigma: int or float
	   Sigma parameter of a Gaussian filter used to smooth the field 
	   before looking for local minima
	* width: int
	   width of the ellipsis defining the neighbourhood
	* height: int
	   height of the ellipsis defuining the neighbourhood

    Returns:
	* ellipsis: array-like, shape of field
	   ellipses defining the neighbourhood around the local minima
    """
    ellipses = []


    for i,f in enumerate(field):
        lmins = seg.get_labeled_local_minima(f,10,1)
        lmin_ellipses = get_track_ellipses(lmins,ellipsis_width=width,ellipsis_height=height)

        ellipses.append(lmin_ellipses)
        
    return ellipses
    
def get_object_properties4tracking(objects):
    """
    Get object centroids, areas and equivalent diameters for a given labeled field.
    
    Inputs:
        * objects: array-like, int
            labeled objects to get properties from
    Returns:
        * dictionary 
            centroid coordinates, areas and eqivalent diameter of the objects
    """
    # import regionprops from scikit.image
    from skimage.measure import regionprops
    
    # calculate object properties
    regions = regionprops(objects)

    # get object labels
    object_labels = np.unique(objects)[1:]

    # collect centroid coodinates, areas and equivalent diameters
    properties = { oid : {'centroid': [], 'area':[],'diameter':[]} for oid in object_labels}

    for i, props in enumerate(regions):
        properties[object_labels[i]]['centroid'] = props.centroid # row and column
        properties[object_labels[i]]['area'] = props.bbox_area
        properties[object_labels[i]]['diameter'] = props.equivalent_diameter
    
    return properties

def get_object_intersections(labeled_field_t0,labeled_field_t1):
    """
        Derive area of the intersection and the symmetrical difference of objects in
        given labeled arrays.

        Inputs:
            * labeled_field_t0: array-like, int
                labeled array of the first time step
            * labeled_field_t1: array-like, int
                labeled array of the second time step

        Returns:
            * object_intersections: dict
    """

    # check how many objects we have in the arrays
    objects_t0 = np.unique(labeled_field_t0)[1:]
    objects_t1 = np.unique(labeled_field_t1)[1:]

    for o0 in objects_t0:
        obj0 = np.zeros_like(o0)

        obj0[labeled_fieldß==o0] = 1

        for o1 in objets_t1:
            obj1 = np.zeros_like(o0)

            obj1[labeled_field_t1==o0] = 1


    A_area = np.count_nonzero(A)
    B_area = np.count_nonzero(B)

    intersection_area = np.count_nonzero(A & B)
    symmetrical_difference = np.count_nonzero(A ^ B)

    A_not_B_area = np.count_nonzero(A ^ (A & B))
    B_not_a_area = np.count_nonzero(B ^ (A & B))
    
    return (intersection_area, symmetrical_difference, A_area, B_area,A_not_B_area, B_not_a_area)
    
# tracking routines ############################################################
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
    
            label1_list_str = [ '{:03d}_{:04d}'.format(time_name1, l)  for l in label1_list]
            #label1_list_str = [ '%s_%s'% (time_name1, str(l).zfill(4))  for l in label1_list]
            
            key = '{:03d}_{:04d}'.format(time_name0, ilab0)
            #key = '%s_%s' % (time_name0, str(ilab0).zfill(4))
            connections[key] = label1_list_str
        except:
            continue
    
    return connections

def generate_graph_from_components(clusters,time,min_connections=1,
                                   track_point='centroid', 
                                   dilation_size=3,method='cost'):
    """
    Generates networkx graphs from labeled data.
    
    Parameters
    ----------
    clusters: numpy array, 2d or 3d, int
        labeled data to generate graphs from
    time: list, time like
        list with the time steps belonging to the cluster layers
    min_connections: int, default=1
        minimum  number of connections to be considered
    track_point: string, possible options 'centroid', 'min','max', default = 'centroid'
        specification of which object point to track
    dilation_size: int, default = 3
        size of the structuring element for the dilation to stabilise the
        segmentation
    method: string, 'overlap' or 'cost', default = 'cost''
        method to use for linking objects, simple overlap or more advanced cost function
    
    Returns
    -------
    G: networkx graph
        graph with all nodes of the cluster stack
    graphs: list of networkx graphs
        graphs with connected components
    """
    # setup for the point options
    valid_track_points = ['centroid','c','min','mn','max','mx']

    # create empty Graph
    G = nx.Graph()
    
    # loop over all labeled arrays
    ntime = len(clusters)
    
    for i in range(ntime - 1):
        try:
            c0 = clusters[i]

            # derive optical flow
            flow = oft.calculate_optical_flow_dis(clusters[i],clusters[i+1])

            # morph c0
            c0 = tco.morph_field(c0,flow[:,:,0],flow[:,:,1])
                    
            # create time strings to label graph nodes with
            t0 = time[i]
            t1 = time[i+1]
        
            # derive connections
            if method in ['overlap','cost']:
                if method == 'overlap':
                    c1 = morphology.dilation(clusters[i+1],
                                             np.ones((dilation_size,dilation_size)))
                    connections = label_connections(c0,c1,t0,t1)
                else:
                    c1 = clusters[i+1]
                    connections = find_successor_lakshmanan(c0,c1,t0,t1,2)
            else:
                print("Invalid method. Only overlap and cost are valid.")
                          
            # fill graph with nodes and edges from the connections
            for c in connections.keys():
                # objects with no connections are not interesting and masked out
                if len(connections[c]) >= min_connections:
                    if not c in G:
                        # get coorindates of the node as centroid of the object
                        oid = int(c.split("_")[-1])
                        
                        if track_point in valid_track_points:
                            if track_point in ['centroid','c']:
                                ocol, orow, oidx = get_object_centroid(clusters,i,oid,
                                                                      median=True)
                            elif track_point in ['max','mx']:
                                orow, ocol, oidx = np.where(clusters == np.max(clusters))
                            elif track_point in ['min','mn']:
                                orow, ocol, oidx = np.where(clusters == np.min(clusters))
                        else:
                            print("{} is not a valid point to track.".format(track_point))
                            return
                        
                        G.add_node(c,col=ocol,row=orow, itime = i, label = oid,location = oidx)
                       
                    for element in connections[c]:
                        if not element in G:
                            # get coorindates of the node as centroid of the object
                            oid = int(element.split("_")[-1])
                                 
                            ocol, orow, oidx = get_object_centroid(clusters,i + 1,oid,
                                                             median=True)
                            
                            G.add_node(element,col=ocol,row=orow, itime = i + 1,
#                                       label = oid, location = oidx)
                                        label = c, location = oidx)
                            G.add_edge(c,element)
                        else:
                            G.add_edge(c,element)
                else:
                    continue
        except Exception as e:
            error_message = "ERROR {} while deriving graphs.".format(e)
            print(error_message)
            continue
                
    # create subgraphs with connected components from the graph
    #subgraphs = list(nx.connected_component_subgraphs(G))
    subgraphs = list((G.subgraph(c).copy() for c in nx.connected_components(G)))
    
    return G, subgraphs

# function to track an object via overlap tracking
def track_object(field_stack,time_list=None,lower_threshold=None,upper_threshold=None,
                 track_point='centroid'):
    """
    Track objects via overlap tracking
    
    Parameters
    ----------
    field_stack: numpy array, 2d or 3d, float
        data with features to track
    time_list: list, time like
        list with the time steps belonging to the cluster layers
    lower_threshold: float, default = None
        lower threshold to use for the segmentation, if None it is calulated
        from the field stack
    upper_threshold: float, default = None
        upper threshold to use for the segmentation, if None it is calulated
        from the field stack    
    track_point: string, possible options 'centroid', 'extreme'
        selection of which object point to track
        
    Returns
    -------
    G: networkx graph
        graph with all nodes of the cluster stack
    graphs: list of networkx graphs
        graphs with connected components
    """
    
    # setup for track points
    valid_track_points = ['centroid','c','extreme','e','ex']
    
    # check if a time list exists, if not create a relative one
    if not time_list:
        time_list = np.arange(0,len(field_stack),1)
    # check if input is a brightness temperature or a reflectance
    if np.min(field_stack) > 1:
        if not upper_threshold:
            #upper_threshold = filters.threshold_yen(field_stack[6])
            upper_threshold = filters.threshold_otsu(field_stack[6])
        if not lower_threshold:
            #lower_threshold = np.nanpercentile(field_stack[6],1)
            lower_threshold = np.nanpercentile(field_stack[6],5)
            
        obj = seg.find_object_hysteresis_threshold_2d(-field_stack,
                                                      -upper_threshold,
                                                      -lower_threshold)
        
        if track_point in valid_track_points:
            if track_point in ['centroid','c']:
                track_point = 'centroid'
            elif track_point in ['extreme','e','ex']:
                track_point = 'min'
            else:
                print("Unknown object point to track.")
                return
        else:
            print("{} is not a valid object point to track.".format(track_point))
            return
        #obj = [-o for o in obj]
    else:
        if not lower_threshold:
            lower_threshold = filters.threshold_yen(field_stack[6])
        if not upper_threshold:
            upper_threshold = np.nanpercentile(field_stack[6],99)
        obj = seg.find_object_hysteresis_threshold_2d(field_stack,
                                                      lower_threshold,
                                                      upper_threshold)
        
        if track_point in valid_track_points:
            if track_point in ['centroid','c']:
                track_point = 'centroid'
            elif track_point in ['extreme','e','ex']:
                track_point = 'max'
            else:
                print("Unknown object point to track.")
                return
        else:
            print("{} is not a valid object point to track.".format(track_point))
            return
               
    connections, object_connections = generate_graph_from_components(obj[0],time=time_list,min_connections=1)
    
    return obj[0], object_connections

def get_object_overlap_sizes(labeled_field,graph_nodes_time,points_prev):
    """
    Determine the sizes of given objects.
    
    Parameters:
        * labeled_field: 2d array of integers
            field with the objects belonging to the graph
        * graph_nodes_time: list
            graph nodes of a certain time step
        * points_prev: numpy array
            locations of the points of the previous object
    Returns:
        * wanted_id: string
            id of 
            the wanted object
    """
    
    overlap_sizes = []
    elements = []
    
    for element in graph_nodes_time:
        obj_id = int(element.split("_")[1])

        obj_points = np.where(labeled_field.ravel()==obj_id)[0]

        overlap = np.intersect1d(points_prev,obj_points)
        overlap_sizes.append(len(overlap))
        elements.append(element)

    overlap_sizes = np.array(overlap_sizes)
    real_overlap = np.where(overlap_sizes>0)
    elements = np.array(elements)[real_overlap]
    overlap_sizes = overlap_sizes[real_overlap]

    try:
        wanted_id = elements[np.argmax(overlap_sizes)]
        return str(wanted_id)
    except:
        return

def extract_main_graph_from_complex(labeled_fields,time_sorted_graph,start_node,method='overlap'):
    """
    Extract the main path of the object from a complex graph using either the
    object size or the size of object overlaps.
    
    Parameters:
        labeled_fields: 3d array of integers
            fields with the objects belonging to the graph
        time_sorted_graph: dictionary
            dictionary with the time steps and the nodes belonging to the object
            and the time step
        start_node: string
            identifier of the object node to start from
        method: string, 'size' or 'overlap' or 'cost'
            method to determine which object to choose
        dilation_size: int
            size of the structuring element for the dilation
            
    Returns:
        path_nodes: list
            list with time identifier and object identifier
    """
    
    valid_methods = ['size','s','overlap','o','cost','c']
    
    if method in valid_methods:
        if method in valid_methods[:2]:
            path_nodes = get_object_sizes(labeled_fields,time_sorted_graph)

            return path_nodes
        
        elif method in valid_methods[2:3]:
            path_nodes = []
            
            t_center = len(labeled_fields) // 2
            
            t_back = np.arange(t_center-1,-1,-1)
            t_fwd = np.arange(t_center+1,len(labeled_fields),1)
            
            #start_id = int(time_sorted_graph[t_center][0].split("_")[1])
            start_id = int(start_node.split("_")[1])
            
            start_points = np.where(labeled_fields[t_center].ravel()==start_id)

            points_prev = start_points
            prev_id = start_node
            obj_back = [start_node]
            obj_fwd = []
            
            # backward
            for i,t in enumerate(t_back):
                if len(time_sorted_graph[t]) >1 :
                    wanted_id = get_object_overlap_sizes(labeled_fields[t],
                                                         time_sorted_graph[t],
                                                         points_prev)
                    prev_id = wanted_id
                    obj_back.append(prev_id)
                else:
                    prev_id = time_sorted_graph[t][0]
                    obj_back.append(time_sorted_graph[t][0])
            
                points_prev = np.where(labeled_fields[t].ravel() == int(prev_id.split("_")[1]))

            # forward
            points_prev = start_points
            prev_id = start_node
            
            for i,t in enumerate(t_fwd):
                if len(time_sorted_graph[t]) >1 :
                    wanted_id = get_object_overlap_sizes(labeled_fields[t],
                                                       time_sorted_graph[t],
                                                       points_prev)
                    prev_id = wanted_id
                    obj_fwd.append(prev_id)
                else:
                    prev_id = time_sorted_graph[t][0]
                    obj_fwd.append(prev_id)
                    
                points_prev = np.where(labeled_fields[t_fwd[i]].ravel() == int(prev_id.split("_")[1]))
                    
            path_nodes = obj_back[::-1] + obj_fwd
            
            return path_nodes
        else:
          print("'{}' is not a valid method. Try 'size' or 'overlap'.".format(method))
          return          
               
        
    else:
        print("'{}' is not a valid method. Try 'size' or 'overlap'.".format(method))
        return

def get_wanted_object_id(labelled_fields):
    """
    Selects the object which contains the central point of the cutout in the 
    centre of all available field layers.
    
    Input:
        * labelled_fields: array like, 3d
            fields to find the object in
    Returns:
        * wanted_id: int
            identifier of the wanted object
    """
    t_centre = len(labelled_fields)//2
    row_shape = labelled_fields[t_centre].shape[0]
    col_shape = labelled_fields[t_centre].shape[1]
    
    object_labels = np.unique(labelled_fields[t_centre])[1:]
    wanted_id = 0
     
    for o in object_labels:
        idx = np.where(labelled_fields[t_centre]==o)
        
        if row_shape//2 in idx[0] and col_shape//2 in idx[1]:
            wanted_id = o
        else:
            wanted_id = 0
            
    # check if the wanted_id still is zero, if yes check the distances of object centroids
    if wanted_id==0:
        distances = dict()

        centroids = ndi.measurements.center_of_mass(labelled_fields[t_centre],
                                                    labelled_fields[t_centre],
                                                    object_labels)

        for i,c in enumerate(centroids):
            distances[object_labels[i]] = np.sqrt((row_shape//2-c[0])**2 + (col_shape//2-c[1])**2)

        # check for the minimum distance
        min_dist = np.min(np.array(list(distances.values())))

        if min_dist > 10:
            wanted_id = 0
        else:
            wanted_id = object_labels[np.argmin(np.array(list(distances.values())))]
        
    return wanted_id

def get_wanted_objects(labelled_fields,path_nodes):
    """
    Get the bounding boxes of a desired object in an labelled array.
    
    Parameters:
        labeled_fields: 3d or 2d array of integers
            fields to look for the object in
        path_nodes: list of strings
            list with the object identifiers for the time steps to find the
            right object
    
    Returns:
        bounding_boxes: dictionary
            dicitonary with the relative time steps and the bounding box 
            coordinates    
    """
    
    # separate path time steps and object ids
    try:
        time_ids = np.array([int(element.split("_")[0]) for element in path_nodes])
        obj_ids = np.array([int(element.split("_")[1]) for element in path_nodes])
        
        # run through all labelled fields and only retain the wanted object
        for tid, field in enumerate(labelled_fields):
            if tid not in time_ids:
                labelled_fields[tid] = np.zeros_like(labelled_fields[tid])
            else:
                idx = np.where(time_ids == tid)
                obj_id = obj_ids[idx]
            
                
                labelled_fields[tid][np.where(labelled_fields[tid] != obj_id)] = 0
            
                labelled_fields[tid][np.where(labelled_fields[tid] == obj_id)] = 1
    except:
        labelled_fields = np.zeros_like(labelled_fields)
        
    return labelled_fields

# routines for estimating tracking parameters
def get_wanted_graph(object_graphs, target_object_id):
    """
    Find the graph, which contains the wanted object.
    
    Inputs:
        * object_graphs: list of networkx graphs
            graphs in which to look for the object
        * target_object_id: int
            identifier of the wnaed object
    Returns:
        * wanted_object_graph: networkx graph
            graph with the wanted object
    """
    wanted_object_graph = []

    for obj in object_graphs:
        if target_object_id in obj.nodes():
            wanted_object_graph = obj
        else:
            continue
        
    return wanted_object_graph

def get_main_track(labelled_field,time_sorted_graph,target_object_id,method='size'):
    """
    Select the longest track with a wanted object out of a complex graph.
    
    Inputs:
        * labelled_field: array like, 2d or 3d
            field in which to look for the object
        * time_sorted_graph: dictionary
            complex graph sorted by time steps
        * target_objct_id: int
            identifier of the object to look for
        * method: string, 'size' or 'overlap', default = 'size'
            method to use for the identification of the right objects
    Returns:
        * object_track: networkx graph
            wanted object track
    """
     # check if there are splits or merges in the graph
    n_objects = []
    
    for t in time_sorted_graph:
        n_objects.append(len(time_sorted_graph[t]))
        
    if np.any(np.array(n_objects) > 1):
        object_track = extract_main_graph_from_complex(labelled_field,
                                                       time_sorted_graph,
                                                       target_object_id,
                                                       method)
   
    else:
        object_track = [time_sorted_graph[i][0] for i in time_sorted_graph.keys()]
        
    return object_track

def extracted_wanted_object_from_field(field_data,extreme,threshold):
    """
    Exctract the wanted object form given field data. The field is segmented using
    morphological reconstruction and a given threshold. The wanted object is
    defined as the object which includes the centre of the data cutout which is 
    in the centre of the field data stack.
    
    Inputs:
        * field_data: numpy array, 2d or 3d
            data to segment and extract the object from
        * extreme: string, either 'min' or 'max'
            identifier for the extreme to focus segmentation on
        * threshold: float
            threshold used to separate foreground and background
    Returns:
        * wanted_object: numpy array, shape of field_data
            data stack with only the wanted object
    """
    
    try:
        if type(field_data)==dict:
            # segment data
        
            data_segmented = np.array([seg.seg.segment_morphological_reconstruction(field_data[k],extreme,threshold)
                                                for k in sorted(field_data.keys())])
            
        elif type(field_data) == np.ndarray or type(field_data) == list:
            data_segmented = np.array([seg.seg.segment_morphological_reconstruction(d,extreme,threshold)
                                                for d in field_data])
            
        else:
            print("{} is not a valid type for the field data. Only dicitonaries, numpy arrays or lists are valid".format(type(field_data)))
            return
        
        # connect the objects of all time steps
        connections, objects = generate_graph_from_components(data_segmented,
                                                                 np.arange(0,len(data_segmented),1))
        
        # determine the interesting object, it should be in the centre of the 
        # centre time step
        wanted_id = get_wanted_object_id(data_segmented)
        target_object_id = '{}_{:04d}'.format(len(data_segmented)//2,wanted_id)
        
        # look for the graph wich cointains this object
        wanted_object_graph = get_wanted_graph(objects,target_object_id)
        
        # it may be that this is a complex graph, but ideally we only want the
        # path of a single cell
        wanted_object_graph_sorted = sort_graph_by_timesteps(wanted_object_graph)
        
        object_track = get_main_track(data_segmented,wanted_object_graph_sorted,target_object_id)
            
        # get the objects
        wanted_object = get_wanted_objects(data_segmented,object_track)
        
        return wanted_object
            
    except:
        wanted_object = np.zeros_like(field_data)
        return wanted_object

def get_track_from_labeled_field(labeled_field,time=None,min_connections=1,
                                 track_point='centroid',dilation_size=3,method='cost'):
    """
    Get track of connected objects from a segmented and labeled stack of fields. 

    Inputs:
        * labeled_field: array like, 3d, (time,x,y)
            field stack to find object track in
        * time: list
            list with the time identifiers of the fields
        * min_connections: int
            minimum nunber of connections to retain in the track graphs
        * track_point: string, valid = ['centroid','c','min','mn','max','mx']
            identifier for which point to select as object track point
        * dilation size: int
            size of the dilation to stabilise the segmentation_dilation
        * method: string, 'overlap' or 'cost', default = 'cost'
            method to connect objects
    Returns:
        * main_track: list
            list of time steps with object ids
    """
    if not np.any(time):
        t_len = len(labeled_field)
        time = np.arange(0,t_len)
    else:
        t_len = len(time)
        
    # find connections
    connections = generate_graph_from_components(labeled_field,
                                                 time,min_connections,
                                                 track_point,dilation_size,
                                                 method=method)
    
    # get id of the object closest to the cetre of the cutout of the centre time step
    object_id = get_wanted_object_id(labeled_field)

    # find graph which contains the wanted object
    object_graph = get_wanted_graph(connections[1],"{:03d}_{:04d}".format(t_len//2,object_id))

    # sort graph nodes by time    
    wanted_graph = sort_graph_by_timesteps(object_graph)

    # find main track of the object
    main_track = get_main_track(labeled_field,wanted_graph,"{:03d}_{:04d}".format(t_len//2,object_id))

    return main_track

def get_wanted_object_field(labeled_field,track):
    object_field = dict()
    
    for i, obj in enumerate(track):
        t_idx = int(obj.split("_")[0])
        oid = int(obj.split("_")[1])

        cut = np.zeros_like(labeled_field[i])

        obj_locations = np.where(labeled_field[t_idx] == oid)

        cut[obj_locations] = 1
        object_field[t_idx] = cut
        
    return object_field

# extended object tracking after Lakshmanan and Smith (2010)
def equiv_diameter(area):
    """
    Calculate equvalent diameter for a given object area.
    
    Inputs:
        * area: float
    """
    return np.sqrt((4*area) / np.pi)

def intersections(A,B):
    A_area = np.count_nonzero(A)
    B_area = np.count_nonzero(B)

    intersection_area = np.count_nonzero(A & B)
    symmetrical_difference = np.count_nonzero(A ^ B)

   
    return (intersection_area, symmetrical_difference, A_area, B_area)

def intersection_term(object0,object1):
    """
    Derive intersection term for the object tracking cost function.
    
    Inputs:
        * object0: array-like, int, range = [0,1]
            object to look for fitting sucessor candidates
        * object1: array-like, int, range = [0,1]
            possible successir candidate for object0
            
    Returns:
        * intersection term
    """
    # derive intersection, symmetrical difference, object areas
    (intersection,difference,object0_area,object1_area) = intersections(object0,object1)
    
    try:
        # calculate and return intersection term
        return difference# / intersection
    except:
        # if there is no instersection the term in not defined, thus return np.nan
        return np.nan

def centroid_cost(row_t0,row_t1,column_t0,column_t1,area_t0,area_t1,object_t0,object_t1):
    """
    Cos function to find best matching successor object to an existing one.
    
    Inputs:
        * row_t0: int or float
            row coordinate of the centroid of the preceeding object
        * row_t1: int or float
            row coordinate of the centroid of a possible succeeding object
        * column_t0: int or float
            column coordinate of the centroid of the preceeding object
        * column_t1: int or float
            column coordinate of the centroid of a possible succeeding object
        * area_t0: int or float
            area of the preceeding object
        * area_t1: int or float
            area of possible succeeding object
        * object_t0: array_like, int, range: [0,1]
            array with the object to find a succeessor for labeled by 1
        * object_t1: array_like, int, range: [0,1]
            array with a possible succeessor labeled by 1
    Returns:
        cost: float
            cost of assuming an object as a succesor of the selected one
    """
    # calculate metrics
    # centroid distances
    dist_row = (row_t0-row_t1)**2
    dist_column = (column_t0-column_t1)**2
    
    # areas
    scaled_area = area_t0 / np.pi
    dist_area = np.abs(area_t0-area_t1) / np.max(np.array([area_t0,area_t1]))
    
    # equivalent diameters
    diameter_t0 = equiv_diameter(area_t0)
    diameter_t1 = equiv_diameter(area_t1)

    dist_diameter = np.abs(diameter_t0-diameter_t1) / np.max(np.array([diameter_t0,diameter_t1]))
    
    # intersection term 
    intersection = intersection_term(object_t0,object_t1)
    # calculate cost
    cost = dist_row + dist_column + scaled_area * (dist_area + dist_diameter) + intersection
    
    return cost

def find_successor_lakshmanan(objects_t0,objects_t1,t0,t1,search_factor=2):
    """
    Find possible successor candiates for a given object usign a cost function.

    Inputs:
        * objects_t0: array-like, int
            labeled array for the first time step, ideally advection corrected
        * objects_t1: array_like, int
            labeled array for the second time step
        * t0: int
            time index of the first time step
        * t1: int
            time index of the second time step
        * search_factor: int
            factor to broaden diameter around object centroid
    Outputs:

    """
    # find object properties
    properties_t0 = get_object_properties4tracking(objects_t0)
    properties_t1 = get_object_properties4tracking(objects_t1)

    # connect objects
    connections = dict()
    for i0 in properties_t0.keys():
        # extract object centroid, area, diameter and label
        x0 = properties_t0[i0]['centroid'][0]
        y0 = properties_t0[i0]['centroid'][1]
        area0 = properties_t0[i0]['area']
        diameter0 = properties_t0[i0]['diameter']
        
        # determine object label
        label0 = i0
        
        cost = dict()
        for i1 in properties_t1.keys():
            x1 = properties_t1[i1]['centroid'][0]
            y1 = properties_t1[i1]['centroid'][1]
            area1 = properties_t1[i1]['area']
            diameter1 = properties_t1[i1]['diameter']
            
            # calculate centroid distance
            dist = np.linalg.norm(np.array([x0,y0])-np.array([x1,y1]))
            
            # check if centroid is within the search radius
            if dist <= diameter0 * search_factor:
                # determine object label
                label1 = i1

                # derive cutouts
                cmin = int(np.rint(np.clip(x0-diameter0 * search_factor,0,objects_t0.shape[0])))
                cmax = int(np.rint(np.clip(x0+diameter0 * search_factor,0,objects_t0.shape[0])))
                rmin = int(np.rint(np.clip(y0-diameter0 * search_factor,0,objects_t0.shape[0])))
                rmax = int(np.rint(np.clip(y0+diameter0 * search_factor,0,objects_t0.shape[0])))

                cutout_t0 = objects_t0[cmin:cmax,rmin:rmax].copy()
                cutout_t1 = objects_t1[cmin:cmax,rmin:rmax].copy()

                # only keep one object
                cutout_t0[cutout_t0!=label0] = 0
                cutout_t0[cutout_t0==label0] = 1

                cutout_t1[cutout_t1!=label1] = 0
                cutout_t1[cutout_t1==label1] = 1

                # calculate cost function
                cost["{:03d}_{:04d}".format(t1,label1)] = centroid_cost(x0,x1,y0,y1,area0,area1,
                                                                        np.ma.masked_where(cutout_t0==0,cutout_t0).mask,
                                                                        np.ma.masked_where(cutout_t1==0,cutout_t1).mask)

        # look for the object with the least cost, that is the successor
        try:
            min_idx = np.argmin(np.array(list(cost.values())))

            connections["{:03d}_{:04d}".format(t0,label0)] = [list(cost.keys())[min_idx]]

        except:
            continue
            
    return connections

# select wanted track from list of tracks
def select_wanted_track_from_list(tracks,object_id):
    """
    Find a wanted track in a list of tracks.

    Inputs:
        * tracks: list of list
            tracks to select one for
        * object_id: string, "ttt_oooo" with t = time index and o = object index
            identifier of the object to find a tack for

    Returns:
        * wanted_track: list
            track witch wanted object inside
    """
    wanted_track = []

    for track in tracks:
        if object_id in track:
            wanted_track = track
            
            return wanted_track
        else:
            continue

# link objects
def link_objects(object_connections):
    """
    Link objects given by time step and object label.

    Inputs:
        * object_connections: dictionary {time_index: {object_id_t0: ['object_id_t1']}}
            dictionary with object connections

    Returns:
        * tracks: list
            list with tracks denoted by time_label
    """
    tracks = []

    visited_oids = []
    for tid in object_connections.keys():
        for oid in object_connections[tid].keys():
            if oid in visited_oids:
                continue
            else:
                track = [oid]

                t = int(oid.split("_")[0])

                while t < len(object_connections.keys()):
                    try: 
                        oid = object_connections[t][oid][0]
                        track.append(oid)
                        visited_oids.append(oid)

                        t = int(oid.split("_")[0])
                    except:
                        break
                tracks.append(track)

    return tracks

# wrapper for tracking using segmentation with local minima and watershed 
# and tracking via cost function
def track_objects_lmin_cost(data_array,data_field='ir108',flow_field='wv073',minimum_depth=6,tmin=220,tmax=273.15,tlevel=240,spread=5,search_factor=3,smoothing_factor=1):
    """
    Create objects using local thresholding and track them via a cost function.

    Inputs:
        * data_array: array-like, 2d or 3d
            array to derive values from
        * data_field: string, default = 'ir108'
            field in data_array to segment
        * flow_filed: string, default = 'wv073'
            field in data array to derive optical flow from
        * minimum_depth: int
            depth a local minimum has to have to be considered
        * tmin: float or int, default: 220
            minimum temperature for spread in segementation using local thresholds
        * tmax: float or int, default: 273.15
            maximum temperature for spread in segementation using local thresholds
        * tlevel: float or int, default: 240
            leveling temperature for spread in segementation using local thresholds
        * spread: float or int, default: 5
            spread for segementation using local thresholds
        * search_factor: float or int, default = 3
            factor to enlarge search radius around an object to find successor
        * smoothing_factor: float or int, default = 1
            sigma parameter of gaussian smoothing to stabilise sementation

    Returns:
        * objects: array-like, int, same shape as data_array
            segemented data array
        * tracks: list of lists
            object tracks
    """
    # segment data arrays
    objects = []

    for i in range(len(data_array[data_field])):
        obj = seg.watershed_local_min_segmentation(data_array[data_field][i],minimum_depth,tmin,tmax,tlevel,spread,smoothing_factor)
        objects.append(obj)

    # link the object
    object_connections = {}

    for i in range(1,len(objects)):
        flow = oft.calculate_optical_flow_dis(data_array[flow_field][i-1], data_array[flow_field][i])
        o0 = tco.morph_field(objects[i-1], flow[:,:,0], flow[:,:,1], method='forward')
        
        #object_connections[i-1] = find_successor_lakshmanan(objects[i-1],objects[i],i-1,i,search_factor)
        object_connections[i-1] = find_successor_lakshmanan(o0,objects[i],i-1,i,search_factor)
    
    tracks = link_objects(object_connections)

    # return object tracks
    return objects,tracks

def interpolate_grid(data_array,lon0,lat0,lon1,lat1,**kwargs):
    """
    Interpolate given grid denoted by lon0 and lat0 to different grid
    given by lon1 and lat1.

    Inputs:
        * data_array: array_like, float or int
            array with data to interpolate
        * lon0: array_like, float, same shape as data_array
            longitude coordinates of the elements in data_array
        * lat0: array_like, float, same shape as data_array
            latitude coordinates of the elements in data_array
        * lon1: array_like, float
            longitude coordinates to interpolate to
        * lat1: array_like, float
            latitude coordinates to interpolate to
        * **kwargs: arguments
            arguments for scipy.interpolate.griddata

    Returns:
        * interpolated_array: array-like, same shape as lon1 and lat1
            interpolated data_array
    """
    from scipy.interpolate import griddata

    interpolated_array = griddata((lon0.ravel(),lat0.ravel()), 
                                  data_array.ravel(), 
                                  (lon1, lat1), 
                                  **kwargs)

    return interpolated_array

def collect_object_values(data_array,objects):
    """
    Collect data values of given objects and put them into an data frame.

    Inputs:
        * data_array: array-like
            numpy array with values to collect
        * objects: dictionary, dict{time_index: [array-like]} , int, same shape as data_array, range = [0,1]
            segmented data_array where 1 denotes the object to collect the values for
    Returns:
        * collected_data: pandas data frame with ['values','field','time']
            object data collected into a pandas data frame
    """

    # interpolate objects to RX and RADOLAN grid

    obj_hrv = dict()
    obj_rx = dict()

    for i in objects:
        ohr = objects[i].repeat(3,axis=0).repeat(3,axis=1)
        obj_hrv[i] = ohr
        
        obj_rx[i] = interpolate_grid(ohr,
                                     data_array['hlon'],
                                     data_array['hlat'],
                                     data_array['rlon'],
                                     data_array['rlat'],
                                     method='nearest')

    # run through objects and collect values
    values = []
    times = []
    field = []

    for i in objects.keys():
        object_locations = np.where(objects[i]!=0)
        object_locations_hrv = np.where(obj_hrv[i]!=0)
        object_locations_rx = np.where(obj_rx[i]!=0)

        for f in list(data_array.keys())[:16]:
            if f == 'hrv':
                val = data_array[f][i][object_locations_hrv]
            elif f == 'rx':
                val = data_array[f][i][object_locations_rx]
            else:
                val = data_array[f][i][object_locations]
            
            values.extend(val.tolist())
            
            times.extend([(i*5)-30]*len(val))
            field.extend([f]*len(val))
    
    # put everything into a data frame
    collected_data = pd.DataFrame({'value':values,'time':times,'field':field})

    # return data frame
    return collected_data

def aggregate_object_values(object_values,field='ir108',fraction=10,function='min'):
    aggregated_values = []
    
    functions = {'min':np.nanmin,
                 'max':np.nanmax,
                 'mean':np.nanmean,
                 'median':np.nanmedian}
    
    for t in object_values['time'].unique():
        subset = object_values[object_values['time'] == t]
        n_values = len(subset[subset['field'] == field])

        # determine region with the coldest pixels in the IR 10.8 µm channel
        #if field != 'ir108' and field != 'rx':
        #    pixel_locations = 

        if n_values < 15:
            aggregated_values.append(functions[function](subset[subset['field'] == field]['value'].astype("float")))
        else:
            values_sorted = sorted(subset[subset['field'] == field]['value'].astype("float"))
            
            if function == 'min':    
                aggregated_values.append(np.mean(values_sorted[:int(np.rint(n_values/fraction))]))
            elif function == 'max':
                aggregated_values.append(np.mean(values_sorted[-int(np.rint(n_values/fraction)):]))
            else: 
                aggregated_values.append(functions[function](subset[subset['field'] == field]['value'].astype("float")))
                
    return aggregated_values

