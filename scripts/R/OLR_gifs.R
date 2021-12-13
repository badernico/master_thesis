library(ncdf4)
library(rasterVis)
library(raster)
library(sp)
library(maps)
library(maptools)
library(lubridate)

fname <- '/Volumes/Elements/data/msevi_rss/olr/2021/07/OLR_20210715.nc'
nc_open(fname)

Sys.setenv(TZ='UTC')

OLR <- ncvar_get(nc_open(fname), 'olr')
lat <- ncvar_get(nc_open(fname), 'lat')
lon <- ncvar_get(nc_open(fname), 'lon')
time <- ncvar_get(nc_open(fname), 'time')

for (i in c(1:length(time))) {
  timestep <- time[i]
  time_POS <- as.POSIXct(timestep, origin = '1970-01-01')
  
  OLR_ts <- OLR[,,i]
  OLR_ts <- t(OLR_ts)
  
  r_OLR <- raster(OLR_ts)
  projection(r_OLR) = CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
  extent(r_OLR) = c(min(lon), max(lon),min(lat),max(lat))
  
  countries <- map("world", plot=FALSE) 
  countries <- map2SpatialLines(countries, proj4string = CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"))
  
  png(paste0('Desktop/pics/',time_POS,'.png'), width = 1500, height = 1200, res = 250)
  print(levelplot(r_OLR, margin = FALSE, at=seq(from=85,to=205,length=30), main = paste0(time_POS, ' UTC'), colorkey = list(title = 'W m-2'))+layer(sp.lines(countries)))
  dev.off()
}

timestep <- time[1]
time_POS <- as.POSIXct(timestep, origin = '1970-01-01')

library(gifski)
png_files <- list.files(paste0("Desktop/pics/"), pattern = ".png", full.names = TRUE)
gifski(png_files, gif_file = paste0("/Volumes/Elements/gifs/",'OLR_',year(time_POS),'_',month(time_POS),'_',day(time_POS),'.gif'), width = 1500, height = 1200, delay = 0.5)






