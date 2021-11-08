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

import os

# general path cofigurations --------------------------------------------------
data_root = '/vols/satellite/datasets'

# configurations for the HDCP2 RADOLAN RX files -------------------------------
# folder of the NetCDF4 files
rx_root = os.getenv('RADOLAN_RX_HDCP2_ROOT',
                    '{}/ground/radolan/rx_hdcp2/'.format(data_root))

# stencil for the file name
rx_path_fmt = '{root}/{dt:%Y}/hdfd_miub_drnet00_l3_dbz_v00_{dt:%Y%m%d}000000.nc'

