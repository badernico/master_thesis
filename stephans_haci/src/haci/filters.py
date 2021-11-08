#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This file is part of the Haeberlie-Algorithm for CI detection package
# (HACI) developed within the satellite group at TROPOS.
#
# HACI is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
# Authors/Contact
# -Stephan Lenk (lenk@tropos.de)
# -Hartwig Deneke (deneke@tropos.de)

import pandas as pd

def duration_filter(objprops, min_dur = 6):
    '''
    Filter for persisent Objects.

    Parameters:
        * objprops: pandas dataframe
            properties of the HACI objects
        * min_dur: int
            minimum duration an object has to have to be considered

    Returns:
        * objprops with new column for duration
        * list with booleans indicating whether the minimun duration criterion was met
    '''


    t_duration = objprops.t1 - objprops.t0

    objprops = objprops.assign(dt = t_duration)
    return (objprops,t_duration >= min_dur)

def range_filter(objprops, 
                 field_shape = (288,900,900), 
                 time_edge = 12, 
                 space_edge = 30):

    '''
    Filters objects too close to bounds.

    Parameters:
        * objprops: pandas dataframe
            properties of the HACI objects
        * field_shape: tuple (time, rows, columns)
            shape of the filed the HACI objects have been derived from
        * time_edge: int, default=12
            minimum distance an object has to have from the time boundaries 
        * space_edge: int, default=30
            minimum distance an object has to be separated from the spatial boundaries
    Returns:
        * list with booleans indicating whether the criteria have been met
    '''


    ntime, nrow, ncol = field_shape

    # temporal filter
    m_time1 = (objprops.t0 > time_edge)
    m_time2 = (objprops.t1 < (ntime - time_edge))
    
    m_time_range = m_time1 & m_time2


    # Filtering based on Bounding Box
    
    # lines or rows
    m_lrange = (objprops.l00 > space_edge) &  (
        objprops.l01 < nrow -  space_edge)
    
    # columns
    m_crange = (objprops.c00 > space_edge) &  (
        objprops.c01 < ncol -  space_edge)

    
    # combination of all
    m_range = m_lrange & m_crange & m_time_range


    return m_range


def area_growth_filter(objprops,
                       max_init_area = 100,
                       min_area_change = 200):

    '''
    Calculates and compares areas from bounding boxes.

    Parameters:
        * objprops: pandas dataframe
            properties of the HACI objects
        * max_init_area: int, default=100,
            maximum starting are an object is allowed to have
        * min_area_change: int, default=200
            minimum change of the area during its life time an object has to have
    Returns:
        * objprops with new columns for starting area, maximum area and
          area change
        * list with booleans indicating whether the criteria have been met
    '''


    area0 = (objprops.l01 - objprops.l00) * (objprops.c01 - objprops.c00)
    area_max = (objprops.l1 - objprops.l0) * (objprops.c1 - objprops.c0)
    darea = area_max - area0

    m_area = (area0 < max_init_area) & (darea > min_area_change)

    objprops = objprops.assign(area0 = area0,
                               area_max = area_max,
                               darea = darea)

    return (objprops, m_area)

def area_ratio_filter(objprops,
                      area_low_threshold=20,
                      area_high_threshold=35,
                      min_area_ratio=0.3):
    """
    Calculates and compares the ratio of the area with more than higher
    threshold and with more than the lower one

    Parameters:
        * objprops: pandas dataframe
            properties of the HACI objects
        * area_low_threshold: int, default = 20
            threshold value to determine the surrounding area of the wanted 
            object
        * area_high_threshold: int, default = 35
            threshold value to dermine the area of the object, should be the
            same as the valeu used to defien the object
        * min_area_ratio: float, default = 0.3
            minimum ratio of the areas
    Returns:
        * objprops with new column for the area ratio
        * list with booleans indicating whether the criteria have been met
    """
    from haci import data_io as hdio
    from . import haci_config
    from . import time2index
    from . import recalibrate_value
    import datetime as dt
    from tqdm import tqdm
    import numpy as np

    # the object properties area usually returned per day, so we read the RADOLAN RX data for the 
    # specific day
    date = "{}T{:02d}{:02d}".format(objprops.iloc[0].date.replace("-",""),
                                    (objprops.iloc[0].t0*5)//60, 
                                    (objprops.iloc[0].t0*5)%60)
    
    date = pd.Timestamp(date).to_pydatetime()

    rx_filename = haci_config.rx_path_fmt.format(root=haci_config.rx_root,dt=date)

    rx = hdio.read_rx_hdcp2(rx_filename)

    # recalibrate threshold values
    area_low_threshold = recalibrate_value(rx.dbz,area_low_threshold)
    area_high_threshold = recalibrate_value(rx.dbz,area_high_threshold)

    # loop over all HACI objects in the objprops file and compare area ratios
    ratios = []
    for i, obj in tqdm(objprops.iterrows(), total=objprops.shape[0], desc="Filtering HACI objects for area ratios..."):
        try:
            # create cutout around the object
            cutout = rx.dbz.data[obj.t0][obj.l0:obj.l1,obj.c0:obj.c1]
            
            # determine areas
            area_low = np.where(cutout>=area_low_threshold)
            area_high = np.where(cutout>=area_high_threshold)
            
            try:
                ratio = area_high[0].size / area_low[0].size
            except:
                ratio = np.nan
            
            ratios.append(ratio)
        except:
            ratios.append(np.nan)

    m_ratio = np.array(ratios) > min_area_ratio

    objprops = objprops.assign(area_ratio = ratios)
    return (objprops, m_ratio)
