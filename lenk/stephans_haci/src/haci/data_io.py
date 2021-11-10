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

import xarray as xr

def read_rx_hdcp2(fname,slice=None):
    '''
    read RADOLAN RX data in HDCP2 format

    Inputs:
        * fname: string
            file name
        * slice: Slice
            time slice of which time frame to load
    '''
    # read dataset
    rx = xr.open_dataset(fname,mask_and_scale=False)#,autoclose=True)
    # extract time slice 
    if slice is not None:
        rx = rx.isel(time=slice)
    # create deep copy of dataset so it can be modified
    rx = rx.copy(deep=True)
    #  fixup valid_max and add _FillValue
    rx.dbz['_FillValue'] = 250
    rx.dbz['valid_max'] = 249
    return rx

def write_table(f, df, fmt, header=None):
    # check if f is file object
    if not hasattr(f,'write'):
        f = open(f,'w')
        close_file = True
    else:
        close_file = False
    if header is not None:
        f.write(header+'\n')
    for r in df.itertuples():
        f.write(fmt.format(**r._asdict())+'\n')
    if close_file:
        f.close()
    return
