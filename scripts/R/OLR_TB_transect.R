# This script is for determining the threshold values for tobac
# For this, lets have a look at the time series of a thunderstorm development
# at a specific point.
library(ncdf4)
library(raster)
library(dplyr)
library(ggmap)
library(caTools)

Sys.setenv(TZ='UTC')

lat_seq <- seq(48.523, 48.523, ((48.523)-48.523)/1000)
lon_seq <- seq(8.107, 15.731, ((15.731)-8.107)/1000)

coo <- data.frame('x' = lon_seq, 'y' = lat_seq)
coordinates(coo) <- ~x+y

# netCDF file of the DOI
fname <- '/Volumes/Elements/data/msevi_rss/tobac_tb/2021/06/OLR_20210621.nc'
nc_open(fname)

OLR <- ncvar_get(nc_open(fname), 'olr')
lat <- ncvar_get(nc_open(fname), 'lat')
lon <- ncvar_get(nc_open(fname), 'lon')
time <- ncvar_get(nc_open(fname), 'time')

# extract the OLR data of the POI
i = 50

timestep <- time[i]
time_POS <- as.POSIXct(timestep, origin = '1970-01-01')

OLR_ts <- OLR[,,i]
OLR_ts <- OLR_ts[,c(ncol(OLR_ts):1)]
OLR_ts <- t(OLR_ts)

#OLR_ts <- (OLR_ts-min(OLR_ts))/(max(OLR_ts)-min(OLR_ts))

#OLR_ts <- TB-OLR

r_OLR <- raster(OLR_ts)
projection(r_OLR) = CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
extent(r_OLR) = c(min(lon), max(lon),min(lat),max(lat))

data_TB <- extract(r_OLR, coo)
#data_OLR <- extract(r_OLR, coo)

data_TB <- data.frame('x' = c(1:1001),'TB' = data_TB)

write.csv(data_TB, file = 'Desktop/data_TB.csv', row.names = FALSE)

plot(runmean(data_TB,5), type = 'l', ylim = rev(range(data_TB)),
     xlab = "", ylab = "",
     xaxt = 'n', yaxt = 'n',
     lwd = 1.5)
axis(1, at = c(0,length(data_TB)),labels = c('A','B'), font = 2)
title(ylab=expression("T"["10.8"]~" [K]"), line=2.5, cex.lab=1)
title(xlab="Transect", line=1, cex.lab=1.2, font = 2)
axis(2, at = c(280,260,250,240,230,220), labels = c(280,260,250,240,230,220))
abline(h = c(220,230,240,250), lty = 2, col = 'grey50')
abline(h = 220, lty = 2, col = 'darkred')
abline(h = 230, lty = 2, col = 'red')
abline(h = 240, lty = 2, col = 'green')
abline(h = 250, lty = 2, col = 'forestgreen')


# differenct between Brightness Temperature and OLR
data_diff <- data_TB-data_OLR
data_diff_norm <- (data_diff-min(data_diff))/(max(data_diff)-min(data_diff))


png(paste0('~/Documents/Uni_Leipzig/master_thesis/seminar/P6/pics/Transect_',time_POS,'.png'), width = 2400, height = 1800, res = 400)
plot(runmean(data_TB,k=5, align = 'center'), type = 'l', ylim = c(0,1), col = 'skyblue',lwd=1.5,
     xlab = 'Transect', ylab = 'Amplitude (normalized)', xaxt = 'n', main = paste0(time_POS, ' UTC'))
axis(1, at = c(0,length(data_diff)),labels = c('A','B'), font = 2)
lines(runmean(data_OLR,k=5, align = 'center'), col = 'red',lwd=1.5)
lines(runmean(data_diff_norm, k = 10, align = 'center'), lty = 2, col = 'black')
legend('bottom',legend = c('OLR',expression("T"["10.8"]),'Difference'), col = c('red','skyblue','black'), lty = c(1,1,2), lwd = c(1.5,1.5,1), horiz = TRUE, cex =0.6)
dev.off()

library(ggplot2)

col_temp <- c("#FF00FFFF", "#0000FFFF", "#00FFFFFF", "#00FF00FF",# "#FFFF00FF", 
              "#FF0000FF", "#6B0000")

png('Documents/Uni_Leipzig/master_thesis/thesis/P5/plots/transect_olr_20210620T005528Z.png', width = 2000, height = 2000)
print(get_map(c(left = min(data$lon)-0.5, bottom = min(data$lat)-0.5, right = max(data$lon)+0.5, top = max(data$lat)+0.5)) %>% 
        ggmap()+
        geom_point(data = data, aes(x = data$lon, y = data$lat, color = data$olr), size = 15) +
        scale_colour_gradientn(colors = col_temp, na.value = NA, name = 'OLR [W m-2]')+
        xlab("Longitude [°]") +
        ylab("Latitude [°]") +
        theme(legend.title = element_text(size = 40),
              legend.text = element_text(size = 40),
              legend.key.size = unit(6, "cm"),
              legend.key.width = unit(1.5, "cm"),
              axis.text = element_text(size = 40),
              axis.title = element_text(size = 40),
              plot.title = element_text(size = 40, hjust = 0.5)) 
)
dev.off()


