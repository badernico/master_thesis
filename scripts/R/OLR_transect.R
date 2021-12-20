# This script is for determining the threshold values for tobac
# For this, lets have a look at the time series of a thunderstorm development
# at a specific point.
library(ncdf4)
library(raster)
library(dplyr)
library(ggmap)

Sys.setenv(TZ='UTC')

lat_seq <- seq(49.38911, 53.25186, ((53.25186)-49.38911)/100)
lon_seq <- seq(6.84775, 15.99609, ((15.99609)-6.84775)/100)

# netCDF file of the DOI
fname <- '/Volumes/Elements/data/msevi_rss/olr/2021/06/OLR_20210620.nc'
nc_open(fname)

OLR <- ncvar_get(nc_open(fname), 'olr')
lat <- ncvar_get(nc_open(fname), 'lat')
lon <- ncvar_get(nc_open(fname), 'lon')
time <- ncvar_get(nc_open(fname), 'time')

# extract the OLR data of the POI
i = 83
data <- data.frame()
for (i in c(1:length(time))) {
  timestep <- time[i]
  time_POS <- as.POSIXct(timestep, origin = '1970-01-01')
  
  OLR_ts <- OLR[,,i]
  OLR_ts <- t(OLR_ts)
  
  r_OLR <- raster(OLR_ts)
  projection(r_OLR) = CRS("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
  extent(r_OLR) = c(min(lon), max(lon),min(lat),max(lat))
  
  temp_point <- extract(r_OLR, coo)
  
  temp_row <- data.frame('lat' = lat_seq, 'lon' = lon_seq, 'olr' = temp_point)
  data <- rbind.data.frame(data, temp_row)
  print(as.POSIXct(timestep, origin = '1970-01-01'))
}

plot(x = data$time, y = data$olr, type = 'l')

plot(data$olr)
as.POSIXct(time, origin = '1970-01-01')

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


