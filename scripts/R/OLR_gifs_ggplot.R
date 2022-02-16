library(ncdf4)
library(rasterVis)
library(raster)
library(sp)
library(maps)
library(maptools)
library(lubridate)
library(gifski)
library("rnaturalearth")
library("rnaturalearthdata")

Sys.setenv(TZ='UTC')

#### read tobac output
ncfname <- '~/Documents/Uni_Leipzig/tobac/tobac-tutorials/themes/tobac_v1/tobac_auto_tracking/Save/Track_20210622.nc'
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
####

fname <- '/Volumes/Elements/data/msevi_rss/tobac_tb/2021/06/OLR_20210622.nc'
#nc_open(fname)

OLR <- ncvar_get(nc_open(fname), 'olr')
lat <- ncvar_get(nc_open(fname), 'lat')
lon <- ncvar_get(nc_open(fname), 'lon')
time <- ncvar_get(nc_open(fname), 'time')
i = 60
for (i in c(1:length(time))) {
  timestep <- time[i]
  time_POS <- as.POSIXct(timestep, origin = '1970-01-01')
  
  ####
  data_short <- data[data$timestr == time_POS,]
  points <- data_short[,c(10:11)]
   colnames(points) <- c('lat','lon')
  
  OLR_ts <- OLR[,,i]
  OLR_ts <- OLR_ts[,c(ncol(OLR_ts):1)]
  OLR_ts <- t(OLR_ts)
  
  rownames(OLR_ts) <- sort(lat,decreasing = TRUE)
  colnames(OLR_ts) <- lon
  
  longdata <- melt(OLR_ts)
  colnames(longdata) <- c('lat','lon','Tb')
  
  longdata <- longdata[which(longdata$lat >= 47.5 & longdata$lat <= 50.5 & longdata$lon >= 8.5 & longdata$lon <= 12.5),]
  points <- points[which(points$lat >= 47.5 & points$lat <= 50.5 & points$lon >= 8.5 & points$lon <= 12.5),]
  
  mycol <- colorRampPalette(c('#000099', '#0080FF','#00FFFF','#B2FF66','#FFFF33','#FF9933','#FF3333','#990000'))(301)
  mycol2 <- colorRampPalette(c('white','black'))(700)
  mycols <- c(rev(mycol),mycol2)
  
  world <- ne_countries(scale = "medium", returnclass = "sf")
  
  png(paste0('~/Desktop/pics/',time_POS,'.png'), width = 1800, height = 2400, res = 400)
  print(ggplot() +
          geom_point(data=longdata,aes(x = lon, y = lat, colour = Tb), size = 1)+
          geom_sf(data = world, alpha = 0.2, col ='black') +
          coord_sf(xlim = c(min(longdata$lon)-0.5, max(longdata$lon)+0.5), ylim = c(min(longdata$lat)-0.5, max(longdata$lat)+0.5), expand = FALSE)+
          geom_point(data = points, aes(x = lon, y = lat), col = 'red',shape = 8, size = 1)+
          scale_colour_gradientn(name = expression("T"[B]), values=seq(0,1,0.01),colours = mycols,
                                 space = "Lab" , guide = "colourbar", na.value = "grey50",
                                 aesthetics = "colour", limits = c(210,310)) +
          labs(x="Longitude [º]", y="Latitude [º]"))
  dev.off()
  print(paste0(round((i/length(time))*100,2),' %'))
}


png_files <- list.files(paste0("~/Desktop/pics/"), pattern = ".png", full.names = TRUE)
time_POS <- as.POSIXct(time[1], origin = '1970-01-01')
gifski(png_files, gif_file = paste0("/Volumes/Elements/gifs/",'OLR_',year(time_POS),'_',month(time_POS),'_',day(time_POS),'.gif'), width = 1800, height = 2400, delay = 0.5)



as.POSIXct(time[i],origin = '1970-01-01')





