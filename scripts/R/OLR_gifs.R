library(ncdf4)
library(rasterVis)
library(raster)
library(sp)
library(maps)
library(maptools)
library(lubridate)
library(gifski)

fname <- '/Volumes/Elements/data/msevi_rss/tobac_tb/2021/06/OLR_20210622.nc'
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
  OLR_ts <- OLR_ts[,c(ncol(OLR_ts):1)]
  OLR_ts <- t(OLR_ts)
  
  r_OLR <- raster(OLR_ts)
  projection(r_OLR) = CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
  extent(r_OLR) = c(min(lon), max(lon),min(lat),max(lat))
  
  countries <- map("world", plot=FALSE) 
  countries <- map2SpatialLines(countries, proj4string = CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"))
  
  mycol <- colorRampPalette(c('#000099', '#0080FF','#00FFFF','#B2FF66','#FFFF33','#FF9933','#FF3333','#990000'))(301)
  mycol2 <- colorRampPalette(c('white','black'))(700)
  
  png(paste0('~/Desktop/pics/',time_POS,'.png'), width = 1800, height = 2400, res = 400)
  #print(levelplot(r_OLR, margin = FALSE, at=seq(from=195,to=305,length=60), main = paste0(time_POS, ' UTC'), colorkey = list(title = 'K'))+latticeExtra::layer(sp.lines(countries)))#+layer(sp.points(points)))
  print(levelplot(r_OLR, margin = FALSE, at = c(seq(210,240,0.1),seq(240.1,310,0.1)), main = list(paste0(time_POS, ' UTC'), hjust = +0.35), col.regions = c(rev(mycol),mycol2), colorkey = list(title = 'K', space = 'bottom', width = 0.95), xlab = list(label = "Longitude", vjust = -.2), ylab = list(label = "Latitude", vjust = +1))
        +latticeExtra::layer(sp.lines(countries)))
  dev.off()
  print(paste0(round((i/length(time))*100,2),' %'))
}

length(seq(240.1,310,0.1))

png_files <- list.files(paste0("~/Desktop/pics/"), pattern = ".png", full.names = TRUE)
time_POS <- as.POSIXct(time[1], origin = '1970-01-01')
gifski(png_files, gif_file = paste0("/Volumes/Elements/gifs/",'OLR_',year(time_POS),'_',month(time_POS),'_',day(time_POS),'.gif'), width = 1800, height = 2400, delay = 0.5)

