#!/bin/python

import numpy as np
import 
def rearrange_dictionary(input_dictionary):
    """
    Rearrange a dictionary with {key: [row, column, value]} into
    {rows:[rows], cols:[cols], values:[values]} or a dicitonary 
    with {key: [row, col]} into {rows:[rows], cols:[cols]}.

    Inputs:
        * input_dictionary: python dictionary
            dictionary to be re-arranged
    Outputs:
        * rearranged dicionary
    """

    if len(input_dictionary[list(input_dictionary.keys())[0]]) >= 3:
        rows = []
        cols = []
        values = []

        for k in list(input_dictionary.keys()):
            rows.append(input_dictionary[k][1])
            cols.append(input_dictionary[k][0])
            values.append(input_dictionary[k][2])
            
        rearranged_dictionary = {'rows':rows,
                                 'cols':cols,
                                 'values':values}
            
    else:
        rows = []
        cols = []

        for k in list(input_dictionary.keys()):
            rows.append(input_dictionary[k][1])
            cols.append(input_dictionary[k][0])
            
        rearranged_dictionary = {'rows':max_rows,
                                 'cols':max_cols}
    
    return rearranged_dictionary

def create_grid_from_points(points,array_shape):
    """
    Create a 2d numpy array from point coordinates with values not euqal to zero
    denoting the point locations in the array. 

    Inputs:
        * points: list
            List with arrays with point coordinates
        * shape: tuple (int,int)
            shape of the numpy array to create
    Returns:
        * point_grid: numpy array of the given shape
            Array with the points labelled.
    """
    point_grid = np.zeros(shape)

    for p in list(points.keys()):
        ro = int(np.rint(points[p][0]))
        co = int(np.rint(points[p][1]))

        point_grid[ro,co] = p
        
    return point_grid

def RMS_vectors(vector1, vector2):
    """
    Calculate root mean squared of the components of two vectors.
    
    Inputs:
        * vector1: array like, int or float
        * vector2: array like, int or float, same shape as vector1
    Returns:
        * vector norm of the rms
    """
    rms = np.sqrt((vector1**2+vector2**2)/ vector1.shape[0])
    
    return np.linalg.norm(rms)

def calculate_vector_pair_deviations(vectors, vector_combination_list):
    """
    Calculate the deviations between given vector pairs using the root mean squared of the vector elements.

    Inputs:
        * vectors: dictionary
            dicitonary with the vector coordinates
        * vector_combination_list: list
            list with unique combinations aof vectors

    Returns:
        * deviations: dictionary
            dicitonary with the deviations of all vector combinations
    """
    deviations = {v:[] for v in vector_combination_list}
        
    for v in vector_combination_list:
        v0 = v.split('_')[0]
        v1 = v.split('_')[1]

        deviations[v] = RMS_vectors(vectors[v0],vectors[v1])
        
    return deviations

def determine_cutout_slice(points,point_id,lower_limit,upper_limit,radius=50):
    """
    Create a slice to cutout a area of the given radius around a given point.

    Inputs:
        * points: dictionary {point_id: [row, col]}
            Dictionary with point coordinates
        * point_id: int
            Identifier of the desired point
        * lower_limit: int
            Lower limit of the possible coordinates for the cutout
        * upper_limit: int
            upper limit of the possible coordinates for the cutout 
        * radius: inf, default = 50
            radius of the cutout
    Returns:
        * cutout_slice: tuple of slices with (slice(row_min, row_max),slice(col_min,col_max))
    """
    r_min = int(np.rint(np.clip(points[point_id][0] - radius,lower_limit,upper_limit)))
    r_max = int(np.rint(np.clip(points[point_id][0] + radius + 1,lower_limit,upper_limit)))
    c_min = int(np.rint(np.clip(points[point_id][1] - radius,lower_limit,upper_limit)))
    c_max = int(np.rint(np.clip(points[point_id][1] + radius + 1,lower_limit,upper_limit)))
    
    cutout_slice = (slice(r_min,r_max),slice(c_min,c_max))
    
    return cutout_slice

def get_point_location(data,data_labeled,obj,label,percentile=99,filter_size=20,return_value=True):
    """
    Find location of the maximum within a given object.
    
    Inputs:
        * data: array like, int or float
            data to find maximum in
        * data_labeled: array like, int
            data labeled to denote different objects
        * obj: slice returned by scipy.ndimage.find_objects
            boundaries of the current object
        * label: int
            label of the object to look for
        * percentile: int, default=99
            percentile to form a threshold to find local maxima above
        * filter_size: int, default=20
            size of the maximum filter used for the maximum detection
        * return_value: boolean, default=True
            switch to select if also the value of the selected maximum is to be returned
    """
    data_cutout = data[obj].copy()
    label_cutout = data_labeled[obj].copy()
    label_cutout[label_cutout!=label] = 0
    
    not_obj_locations = np.where(label_cutout==0) 
    
    threshold = np.percentile(data_cutout.ravel(),percentile)
    
    data_cutout[not_obj_locations] = 0
    data_cutout[np.where(data_cutout<threshold)]=0
    
    max_filter = ndi.maximum_filter(data_cutout,size=filter_size)
    
    max_location = np.where(max_filter==np.max(max_filter))
    
    max_r = np.min(max_location[0]) + ((np.max(max_location[0]) - np.min(max_location[0])) / 2.)
    max_c = np.min(max_location[1]) + ((np.max(max_location[1]) - np.min(max_location[1])) / 2.)
    
    r = obj[0].start + max_r
    c = obj[1].start + max_c
    
    if return_value==True:
        value = data[int(np.rint(r)),int(np.rint(c))]
        
        return (r,c,value)
    else:
        return (r,c)

def extract_point_locations(grid,limits,point_id,fill=False,return_cutout=True):
    """
    Extract the coordinates of points in an selected area around a desired point and optionally a cutout around the point from a labelled grid.
    Values greater than zero are interpreted as point locations.

    Inputs:
        * grid: 2d numpy array
            grid to extract point and point environment from
        * limits: tuple of slices
            limits for the optional cutout
        * point_id: int
            identifier of the desired point
        * fill: boolean, default=False
            if True the values of the cutout are filled with zero except for the desired point,
            if False the cutout is returned as gotten from the grid
        * return_cutout: boolean, default=True
            if True a cutout defined by the given limits is returned together with the point coordinates,
            if False only the point coordinates are returned
    Returns:
        * points: list of numpy arrays [array([row, col])]
            coordinates of the points within the cutout 
        * cutout: 2d numpy array
            cutout of the grid defined by the limits           
    """
    cutout = grid[limits].copy()
    
    if fill==True:
        cutout[np.where(cutout!=point_id)] = 0
        
    point_locations = np.where(cutout!=0)
    
    # reformat point_locations to (row, column)
    if len(point_locations[0]) > 1:
        points = []

        for i in range(len(point_locations[0])):
            points.append(np.array([point_locations[0][i], point_locations[1][i]]))

    else:
        points = [np.array([point_locations[0][0], point_locations[1][0]])]
    
    if return_cutout==True:
        return cutout, points
    else:
        return points

def calculate_mean_vector(vector0,vector1):
    """
    Calculate the mean vector of two vectors.
    
    Input:
        * vector0: numpy array, float or int
        * vector1: numpy array, float or int
    
    Returns:
        * mean_vector: numpy array, float or int
    """
    
    mean_vector = 0.5 * (vector0 + vector1)
    
    return mean_vector

def get_location_vector(point, origin):
    """
    Calculate the location vector for a given point and origin.

    Inputs:
        * point: numpy array, array([row, column])
            row and column coordinates of the point
        * origin: numpy array, array([row, column])
            row and column coordinates of the origin
    Returns:
        * location_vector: numpy array, array([row, column])
            location vector of the point relative to the origin
    """
    location_vector = point - origin
    
    return location_vector

def determine_unique_combinations(element_list,starting_point='v0'):
    """
    Determine unique combinations of the elements of a given list.

    Inputs:
        * element_list: list
            list with elements of which unique combinations are to be determined
        * starting_point: string, int, default='v0'
            if this is set, only combinations containing this element are returned

    Returns:
        * combinations: list
            list with the unique combinations of the list elements
    """
    combinations = []

    if starting_point:
        v = starting_point
        for w in element_list:
            if not v == w:
                if not '{}_{}'.format(w,v) in combinations:
                    combinations.append('{}_{}'.format(v,w))
    else:
        for v in element_list:
            for w in element_list:
                if not v == w:
                    if not '{}_{}'.format(w,v) in combinations:
                        combinations.append('{}_{}'.format(v,w))
                        
    return combinations

def truncation(sigma,width):
    """
    Calculate truncation width for a Gaussian filter from sigma and filter width.

    Inputs:
        * sigma: float or int
            sigma value of the Gaussian filter
        * width: int
            width of the Gaussian filter

    Returns:
        t: int or float
            truncation width
    """
    t = (((width-1)/2)-0.5)/sigma
    
    return t

def prepare_data(data, filter_sigma=2, filter_width=10, exclude_below=20, exclude_above=92.5,mask_threshold=5,grid_shape=(900,900)):
    """
    Prepare the data for the tracking. The data are smoothed using a 2d Gaussian filter and mask using the given values.
    
    Inputs:
        * data: 2d numpy array
            data to be prepared
        * filter_sigma: float or int
            sigma value for the Gaussian filter
        * filter_width: int
            width of the Gaussian filter
        * exclude_below: float or int
            grid values below this value are masked out
        * exclude_above: float or int
            grid values greater euqal this value are masked out
        * mask_threshold: float or int
            only values above this value are considered for the determination of objects
        * grid_shape: tuple (row, column)
             shape of the grid to be returned with points labeled
    Returns:
        * points: dictionary, {point_id:[row,column]}
            dictionary with point locations
        * point_grid: 2d numpy array of grid_shape
            numpy array with the point locations labelled
    """
    data_filtered = data.copy()
    data_filtered[np.where(data_filtered<exclude_below)] = 0
    data_filtered[np.where(data_filtered>=exclude_above)] = 0
    
    data_smooth = ndi.filters.gaussian_filter(data_filtered,2,truncate=truncation(filter_sigma,filter_width))
    
    data_binary = np.ma.masked_greater(data_smooth,mask_threshold).mask*1
    
    data_labelled, nlabel = ndi.label(data_binary)
    
    objects = ndi.measurements.find_objects(data_labelled)
    
    points = {i+1:[] for i in range(len(objects))}

    for i,o  in enumerate(objects):
        point_loc = get_point_location(data_smooth, data_labelled,o,i+1)

        points[i+1] = point_loc
        
    point_grid = create_point_grid(points,grid_shape)
                 
    return points, point_grid 

def select_next_vector(grid_t0,grid_t1,point_dict_t0,point_dict_t1,
                       point_id,search_radius=50,maximum_decay=35,maximum_increase=40):
    """
    Select the vector which is the shift to the next object location in time.
    
    Inputs:
        * grid_t0: numpy array, int
            numpy array with the locations of the points in the first time step labeled
        * grid_t1: numpy array, int
            numpy array with the locations of the points in the second time step labeled
        * point_dict_t0: dictionary
            dictionary with the labels, locations and values of the points in the first time step
        * point_dict_t1: dictionary
            dictionary with the labels, locations and values of the point in the second time step
        *point_id: int
            label of the pointto be analysed
        * search_radius: int, default = 50
            radius of grid steps to be looked in for objects to connect to the object of the first time step
        * maximum_decay: float or int, default = 35
            maximum decay of intensity of an object can have to still be considered in the next time step
        * maximum_increase: float or int, default = 40
            maximum increase of intensity an object can have to still be considered in the next time step
    Returns:
        * return_array: numpy array with (row, column, object in in t1)
    """
    
    # get limits of the cutout   
    cutout_limits = determine_cutout_slice(point_dict_t0,point_id,0,np.max(grid_t0.shape)-1)
    
    # get cutouts and coordinates of the points
    cutout_t0, p_t0 = extract_point_locations(grid_t0,cutout_limits,point_id,True) 
    cutout_t1, p_t1 = extract_point_locations(grid_t1,cutout_limits,point_id,False)     
        
    # check the possible new objects for increase or decay of intensity:
    # get labels of possible in the second time step
    labels_t1 = np.unique(cutout_t1)[1:]

    obj_t1_values = {int(l):[] for l in labels_t1}

    reference_value = point_dict_t0[point_id][2]
    for l in labels_t1:
        location = np.where(cutout_t1==l)
        value = point_dict_t1[l][2]

        if value - reference_value < maximum_increase and reference_value - value < maximum_decay:
            obj_t1_values[l] = (location[0][0],location[1][0],value)
        else:
            obj_t1_values[l] = (location[0][0],location[1][0],np.nan)
    
    # get the location vectors to the points in t1 relative to the point in t0
    loc_vec = dict()

    #for i,p in enumerate(p_t1):
    for i, k in enumerate(list(obj_t1_values.keys())):
        if not np.isnan(obj_t1_values[k][2]):
            vec = get_location_vector(np.array([obj_t1_values[k][0],obj_t1_values[k][1]]),p_t0)
            loc_vec['v{}'.format(i)] = vec
    
    if len(list(loc_vec.keys())) == 1:
        vec = loc_vec[list(loc_vec.keys())[0]][0]
        return_array = np.array([vec[0],vec[1],list(obj_t1_values.keys())[0]])
        
        return return_array
    else:
        # run through all vector pairs in t1 and store all unique combinations containing the starting point
        vector_combinations = determine_unique_combinations(list(loc_vec.keys()))

        # run through the combinations and calculate RMS
        deviations = calculate_vector_pair_deviations(loc_vec,vector_combinations)    

        # get combination with minimum RMS
        min_idx = np.where(np.array(list(deviations.values()))==np.min(np.array(list(deviations.values()))))

        min_combi = list(deviations.keys())[min_idx[0][0]]

        # calculate mean vector of that combination
        v0 = min_combi.split('_')[0]
        v1 = min_combi.split('_')[1]

        v_m = calculate_mean_vector(loc_vec[v0],loc_vec[v1])

        # select one vector based on a circle around the mean vector and a factor
        for k in np.arange(1.2,3.5,0.1):
            vector_evaluation = dict()

            for v in [v0,v1]:
                evaluation = np.linalg.norm(loc_vec[v] - k*v_m) - np.linalg.norm(k*k*v_m)
                vector_evaluation[v] = evaluation

            if np.any(np.array(list(vector_evaluation.values())) <= 0):
                cand_idx = np.where(np.array(list(vector_evaluation.values())) <= 0)

                cand_vec = np.array(list(vector_evaluation.keys()))[cand_idx]

                if len(cand_vec[0]) > 1:

                    rms = {v:[] for v in [v0,v1]}
                    for cand in cand_vec:
                        rms[cand] = RMS_vectors(loc_vec[cand],v_m)
                    
                    rms_values = np.array(list(rms.values()))
                    
                    min_idx = np.where(rms_values == np.min(rms_values))

                    cand_vec = np.array(list(rms.keys()))[min_idx]
                
                vec = loc_vec[cand_vec[0]][0]
                obj_location = np.array([p_t0[0][0] + vec[0], p_t0[0][1] + vec[1]])
                obj_id = cutout_t1[obj_location[0],obj_location[1]]
                
                return_array = np.array([vec[0],vec[1],obj_id])

                return return_array
            else:
                continue
