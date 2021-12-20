library(ncdf4)

## FEATURES
ncfname <- '~/Documents/Uni_Leipzig/tobac/tobac-tutorials/themes/tobac_v1/OLR_tracking_satellite_NB/Save/2021/07/Features_OLR_20210728.nc'
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

data_short <- data[data$timestr == '2021-07-28 09:04:00',]

points <- data_short[,c(10:11)]
colnames(points) <- c('y','x')
coordinates(points) <- ~x+y



## SEGMENTATION
ncfname <- '~/Documents/Uni_Leipzig/tobac/tobac-tutorials/themes/tobac_v1/OLR_tracking_satellite_NB/Save/2021/07/Mask_Segmentation_OLR_20210728.nc'
nc_open(ncfname)

seg <- ncvar_get(nc_open(ncfname), 'segmentation_mask')
lat <- ncvar_get(nc_open(ncfname), 'lat')
lon <- ncvar_get(nc_open(ncfname), 'lon')
time <- ncvar_get(nc_open(ncfname), 'time')

seg_short <- seg[,,1]
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

