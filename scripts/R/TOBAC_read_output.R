library(ncdf4)
library(rasterVis)
library(raster)
library(sp)
library(maps)
library(maptools)

## FEATURES
ncfname <- '/Volumes/Elements/data/tobac/Save/2021/06/21/Features.nc'
nc_open(ncfname)

frame <- ncvar_get(nc_open(ncfname), 'frame')
idx <- ncvar_get(nc_open(ncfname), 'idx')
hdim_1 <- ncvar_get(nc_open(ncfname), 'hdim_1')
hdim_2 <- ncvar_get(nc_open(ncfname), 'hdim_2')
num <- ncvar_get(nc_open(ncfname), 'num')
threshold_value <- ncvar_get(nc_open(ncfname), 'threshold_value')
feature <- ncvar_get(nc_open(ncfname), 'feature')
time <- ncvar_get(nc_open(ncfname), 'time')
timestr <- ncvar_get(nc_open(ncfname), 'timestr')
latitude <- ncvar_get(nc_open(ncfname), 'latitude')
longitude <- ncvar_get(nc_open(ncfname), 'longitude')
ncells <- ncvar_get(nc_open(ncfname), 'ncells')

data <- cbind.data.frame(frame,idx,hdim_1,hdim_2,num,threshold_value,feature,time,timestr,latitude,longitude,ncells)

timestep = '2021-06-22 13:13:36'

data_short <- data[data$timestr == timestep,]

points <- data_short[,c(10:11)]
colnames(points) <- c('y','x')
coordinates(points) <- ~x+y

####
fname <- '/Volumes/Elements/data/msevi_rss/tobac_tb/2021/06/OLR_20210622.nc'
nc_open(fname)

Sys.setenv(TZ='UTC')

OLR <- ncvar_get(nc_open(fname), 'olr')
lat <- ncvar_get(nc_open(fname), 'lat')
lon <- ncvar_get(nc_open(fname), 'lon')
time <- ncvar_get(nc_open(fname), 'time')

xx = which(as.POSIXct(time,origin = '1970-01-01') == timestep)

OLR_ts <- OLR[,,xx]
OLR_ts <- OLR_ts[,c(ncol(OLR_ts):1)]
OLR_ts <- t(OLR_ts)

r_OLR <- raster(OLR_ts)
projection(r_OLR) = CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
extent(r_OLR) = c(min(lon), max(lon),min(lat),max(lat))

countries <- map("world", plot=FALSE) 
countries <- map2SpatialLines(countries, proj4string = CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"))

mycol2 <- colorRampPalette(c('white','black'))(101)

time_POS <- as.POSIXct(time[26],origin = '1970-01-01')

print(levelplot(r_OLR, margin = FALSE, at = seq(210,310,1), main = list(paste0(time_POS, ' UTC'), hjust = +0.35), col.regions = mycol2, colorkey = list(title = 'K', space = 'bottom', width = 0.95), xlab = list(label = "Longitude", vjust = -.2), ylab = list(label = "Latitude", vjust = +1))
      +layer(sp.lines(countries))+layer(sp.points(points)))

## SEGMENTATION
ncfname <- '/Volumes/Elements/data/tobac/Save/2021/06/21/Mask_Segmentation_TB.nc'
nc_open(ncfname)

seg <- ncvar_get(nc_open(ncfname), 'segmentation_mask')
lat <- ncvar_get(nc_open(ncfname), 'lat')
lon <- ncvar_get(nc_open(ncfname), 'lon')
time <- ncvar_get(nc_open(ncfname), 'time')

seg_short <- seg[,,25]
seg_short <- seg_short[,c(ncol(seg_short):1)]
seg_short <- t(seg_short)

r_seg <- raster(seg_short)
projection(r_seg) = CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
extent(r_seg) = c(min(lon), max(lon),min(lat),max(lat))

countries <- map("world", plot=FALSE) 
countries <- map2SpatialLines(countries, proj4string = CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"))

print(levelplot(r_seg, margin = FALSE, contour = FALSE, at=seq(from=100,to=160,length=60))+layer(sp.lines(countries))+layer(sp.points(points)))

#plot(x = data_short$longitude, y = data_short$latitude)
#contourplot(r_seg, margin = FALSE, labels = FALSE)+layer(sp.points(points))

