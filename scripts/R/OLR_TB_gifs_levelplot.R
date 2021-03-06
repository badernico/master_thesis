library(ncdf4)
library(rasterVis)
library(raster)
library(sp)
library(maps)
library(maptools)
library(lubridate)
library(gifski)

#### read tobac output
ncfname <- '/Volumes/Elements/data/tobac/Save/2021/06/05/Track.nc'
#nc_open(ncfname)

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
cell <- ncvar_get(nc_open(ncfname), 'cell')
time_cell <- ncvar_get(nc_open(ncfname), 'time_cell')

data <- cbind.data.frame(frame,idx,hdim_1,hdim_2,num,threshold_value,feature,time,timestr,latitude,longitude,cell,time_cell)

#cells <- read.csv('~/Documents/Uni_Leipzig/master_thesis/analysis/tobac_tracking_statistics/2021/06/05/cells_filtered.csv')

#data = data[data$cell %in% cells$cellID,]

####

fname <- '/Volumes/Elements/data/msevi_rss/tobac_tb/2021/06/OLR_20210608.nc'
nc_open(fname)

Sys.setenv(TZ='UTC')

OLR <- ncvar_get(nc_open(fname), 'olr')
lat <- ncvar_get(nc_open(fname), 'lat')
lon <- ncvar_get(nc_open(fname), 'lon')
time <- ncvar_get(nc_open(fname), 'time')

i = 36

for (i in c(1:length(time))) {
  timestep <- time[i]
  time_POS <- as.POSIXct(timestep, origin = '1970-01-01')
  
  ####
  data_short <- data[data$timestr == time_POS,]
  
  data_short <- data_short[data_short$cell == "1136",]
  
  if (nrow(data_short) == 0){
    points <- data_short[,c(10:11)]
    points[1,] = rep(0,2)
  }else{
    points <- data_short[,c(10:11)]
  }
  colnames(points) <- c('lat','lon')
  coordinates(points) <- ~ lon + lat
  ####
  
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
        +latticeExtra::layer(sp.lines(countries))+latticeExtra::layer(sp.points(points, col = 'black',fill= 'violet', pch = 21)))
  dev.off()
  print(paste0(round((i/length(time))*100,2),' %'))
}


png_files <- list.files(paste0("~/Desktop/pics/"), pattern = ".png", full.names = TRUE)
time_POS <- as.POSIXct(time[1], origin = '1970-01-01')
gifski(png_files, gif_file = paste0("/Volumes/Elements/gifs/",'OLR_',year(time_POS),'_',month(time_POS),'_',day(time_POS),'.gif'), width = 1800, height = 2400, delay = 0.5)

test = data[data$timestr == "2021-06-05 09:18:56",]
test = data[data$cell == "1136",]

plot(x = test$longitude, test$latitude)

