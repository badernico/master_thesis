library(ncdf4)
library(rasterVis)
library(raster)
library(sp)
library(maps)
library(maptools)
library(lubridate)
library(gifski)

fname <- '/Volumes/Elements/data/msevi_rss/tobac_olr/2021/07/OLR_20210704.nc'
nc_open(fname)

Sys.setenv(TZ='UTC')

OLR <- ncvar_get(nc_open(fname), 'olr')
lat <- ncvar_get(nc_open(fname), 'lat')
lon <- ncvar_get(nc_open(fname), 'lon')
time <- ncvar_get(nc_open(fname), 'time')

i = which(time == as.numeric(as.POSIXct('2021-07-04 13:49:52')))

timestep <- time[i]
time_POS <- as.POSIXct(timestep, origin = '1970-01-01')

OLR_ts <- OLR[,,i]
OLR_ts <- OLR_ts[,c(ncol(OLR_ts):1)]
OLR_ts <- t(OLR_ts)

r_OLR <- raster(OLR_ts)
projection(r_OLR) = CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
extent(r_OLR) = c(min(lon), max(lon),min(lat),max(lat))

countries <- map("world", plot=FALSE) 
countries <- map2SpatialLines(countries, proj4string = CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"))

#mycol <- colorRampPalette(c('#000099', '#0080FF','#00FFFF','#B2FF66','#FFFF33','#FF9933','#FF3333','#990000'))(351)
mycol2 <- colorRampPalette(c('white','black'))(700)

png(paste0('~/Documents/Uni_Leipzig/master_thesis/seminar/P6/pics/TB_',time_POS,'.png'), width = 1800, height = 2400, res = 400)
print(levelplot(r_OLR, margin = FALSE, at = seq(min(OLR_ts),max(OLR_ts),length = 700), main = list(paste0(time_POS, ' UTC'), hjust = +0.35), col.regions = c(mycol2), colorkey = list(title = 'K', space = 'bottom', width = 0.95), xlab = list(label = "Longitude", vjust = -.2), ylab = list(label = "Latitude", vjust = +1))
      +latticeExtra::layer(sp.lines(countries))+latticeExtra::layer(sp.points(coo)))
dev.off()

png(paste0('~/Documents/Uni_Leipzig/master_thesis/seminar/P6/pics/OLR_',time_POS,'.png'), width = 1800, height = 2400, res = 400)
print(levelplot(r_OLR, margin = FALSE, at = seq(min(OLR_ts),max(OLR_ts),length = 700), main = list(paste0(time_POS, ' UTC'), hjust = +0.35), col.regions = c(mycol2), colorkey = list(title = expression('Wm'^'-2'), space = 'bottom', width = 0.95), xlab = list(label = "Longitude", vjust = -.2), ylab = list(label = "Latitude", vjust = +1))
      +latticeExtra::layer(sp.lines(countries)))
dev.off()


# OLR: OLR_real
# TB: OLR_ts

# OLR <- OLR_ts
# TB <- OLR_ts


# convert to normalized values
OLR_norm <- (OLR-min(OLR))/(max(OLR)-min(OLR))
TB_norm <- (TB-min(TB))/(max(TB)-min(TB))

png(paste0('~/Documents/Uni_Leipzig/master_thesis/seminar/P6/pics/Hist_',time_POS,'.png'), width = 1800, height = 2400, res = 400)
hist(TB_norm, col = 'skyblue',border=F, xlab = 'Normalized', main = paste0(time_POS, ' UTC'))
hist(OLR_norm, add = T,col=scales::alpha('red',.5),border=F)
legend('topleft',inset=0.05,legend = c('OLR',expression("T"["10.8"])), col = c('red','skyblue'), lty = c(1,1), box.lwd = 0, lwd = 5:5)
dev.off()

##################
# Difference of TB and OLR
OLR_ts <- TB_norm-OLR_norm

r_OLR <- raster(OLR_ts)
projection(r_OLR) = CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
extent(r_OLR) = c(min(lon), max(lon),min(lat),max(lat))

countries <- map("world", plot=FALSE) 
countries <- map2SpatialLines(countries, proj4string = CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"))

mycol <- colorRampPalette(c('#000099', '#0080FF','#00FFFF','#B2FF66','#FFFF33','#FF9933','#FF3333','#990000'))(351)
#mycol2 <- colorRampPalette(c('white','black'))(700)

png(paste0('~/Documents/Uni_Leipzig/master_thesis/seminar/P6/pics/Difference_',time_POS,'.png'), width = 1800, height = 2400, res = 400)
print(levelplot(r_OLR, margin = FALSE, at = seq(min(OLR_ts),max(OLR_ts),length = 351), main = list(paste0(time_POS, ' UTC'), hjust = +0.35), col.regions = c(mycol), colorkey = list(title = "DIFF  ", space = 'bottom', width = 0.95), xlab = list(label = "Longitude", vjust = -.2), ylab = list(label = "Latitude", vjust = +1))
      +latticeExtra::layer(sp.lines(countries)))
dev.off()


