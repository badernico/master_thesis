library(ncdf4)


# one input file as example
fname <- '/Volumes/Elements/data/msevi_rss/tobac_tb/2021/06/OLR_20210620.nc'
#nc_open(fname)

y <- ncvar_get(nc_open(fname), 'lat') # latitude
x <- ncvar_get(nc_open(fname), 'lon') # longitude

# grid spacing in º
dx <- (max(x, na.rm = TRUE)-min(x, na.rm = TRUE))/length(x)
dy <- (max(y, na.rm = TRUE)-min(y, na.rm = TRUE))/length(y)


# south: 1º <- 78 km (45 ºN)
# north: 1º <- 58 km (58 ºN)
deg_to_m_max <- 78000
deg_to_m_min <- 58000

# dx spacing upper and lower boundary
dx_min <- dx*deg_to_m_min
dx_max <- dx*deg_to_m_max

# dy spacing upper and lower boundary
dy_min <- dy*deg_to_m_min
dy_max <- dy*deg_to_m_max

# means of above values
dx_mean <- (dx_min+dx_max)/2
dy_mean <- (dy_min+dy_max)/2

# mean grid spacing
dxy <- (dx_mean+dy_mean)/2



