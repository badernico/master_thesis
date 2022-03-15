library(dplyr)
library(ggplot2)
library(ggmap)
library(lubridate)
library(gifski)
Sys.setenv(TZ='UTC')

setwd('~/Documents/Uni Leipzig/master/')

gif <- 'YES' #NO

# june files
file_path <- list.files('processing/data/radar/VAIS-LI-202106/',full.names = TRUE)
file_path_short <- list.files('processing/data/radar/VAIS-LI-202106/',full.names = FALSE)
file_path_short <- gsub(file_path_short, pattern = '.txt', replacement = '')

for (i in seq(1,length(file_path),1)) {
  light <- read.table(file_path[i])
  colnames(light) <- c('year','month','day','hour','min','sec','lat','lon','amplitude')
  
  assign(file_path_short[i], light)
  
  #lat_max<-max(light$lat)
  lat_max <- 55.46662
  #lat_min<-min(light$lat)
  lat_min <- 46.60125
  #lon_max<-max(light$lon)
  lon_max <- 15.24322
  #lon_min<-min(light$lon)
  lon_min <- 4.89480
  
  date_mid <- paste0(substr(file_path_short[i],9,12),'-',substr(file_path_short[i],14,15),'-',substr(file_path_short[i],17,18))
  
  col_temp <- c("#FF00FFFF", "#0000FFFF", "#00FFFFFF", "#00FF00FF", "#FFFF00FF", "#FF0000FF", "#6B0000")
  
  #png(paste0("/home/meteoblue/cityclimatemodel_sebastian_branch/output/Munich_model/temp_",date_mid1,".png"), width = 2000, height = 2000, units = "px")
  png(paste0("processing/plots/LI_",date_mid,'.png'), width = 2000, height = 2000, units = "px")
  print(
    get_map(c(left = lon_min +0.01, bottom = lat_min + 0.01, right = lon_max - 0.01, top = lat_max - 0.01)) %>% 
      ggmap()  + 
      
      geom_point(data = light, aes(x = light$lon, y = light$lat, color = light$amplitude), size = 3, alpha = 0.7) +
      
      scale_colour_gradientn(name = "Amplitude", colours = col_temp, values=seq(0,1,0.01),
                             space = "Lab" , guide = "colourbar", # , na.value = "grey50"
                             aesthetics = "colour", limits = c(min(light$amplitude), max(light$amplitude))) +
      #aesthetics = "colour", limits = c(-5,45)) +
      # ggplot2::scale_colour_stepsn(name = "Temperature [°C]", colours = col_temp,n.breaks=50,nice.breaks=TRUE,trans="identity",
      #                       space = "Lab" , guide = "coloursteps", # , na.value = "grey50"
      #                       #aesthetics = "colour", limits = c(min(AirTemperatureInterpolation$temperature), max(AirTemperatureInterpolation$temperature))) +
      #                      aesthetics = "colour", limits = c(-5,45)) +
      #ggplot2::scale_colour_stepsn(name = "Temperature [°C]", n.breaks = 50, trans="log2",colours = col_temp,limits=c(-5,45)) +                            
      
      #ggplot2::scale_fill_stepsn(name = "Temperature [°C]",colors = col_temp, breaks = breaks) +                       
      
      #ggtitle(paste0("Urban heat island")) +
      ggtitle(paste0(" Lightning ", date_mid)) +
      
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
}




if (gif == 'YES') {
  
}

light <- get(file_path_short[20])
light$timestamp <- paste0(light$year,'-',light$month,'-',light$day,' ', light$hour,':',light$min,':',light$sec)
light$timestamp <- as.POSIXct(light$timestamp)
light$timestamp <- round_date(light$timestamp, '15 minutes')

time_seq <- as.data.frame(seq(as.POSIXct(paste0(substr(as.character(light$timestamp),1,10)[1],' 00:00:00')), as.POSIXct(paste0(substr(as.character(light$timestamp),1,10)[1],' 23:45:00')),'15 mins'))
colnames(time_seq) <- 'timestamp'
light <- merge.data.frame(time_seq, light, by = 'timestamp', all.x = TRUE)

for (i in c(1:nrow(time_seq))) {

  light_temp <- light[which(light$timestamp == time_seq$timestamp[i]),]
  
  #lat_max<-max(light$lat)
  lat_max <- 55.46662
  #lat_min<-min(light$lat)
  lat_min <- 46.60125
  #lon_max<-max(light$lon)
  lon_max <- 15.24322
  #lon_min<-min(light$lon)
  lon_min <- 4.89480
  
  date_mid <- as.character(light_temp$timestamp)
  
  col_temp <- c("#FF00FFFF", "#0000FFFF", "#00FFFFFF", "#00FF00FF", "#FFFF00FF", "#FF0000FF", "#6B0000")
  
  #png(paste0("/home/meteoblue/cityclimatemodel_sebastian_branch/output/Munich_model/temp_",date_mid1,".png"), width = 2000, height = 2000, units = "px")
  png(paste0("processing/plots/gif_test/LI_",date_mid,'.png'), width = 1000, height = 1000, units = "px")
  print(
    get_map(c(left = lon_min +0.01, bottom = lat_min + 0.01, right = lon_max - 0.01, top = lat_max - 0.01)) %>% 
      ggmap()  + 
      
      geom_point(data = light_temp, aes(x = light_temp$lon, y = light_temp$lat, color = light_temp$amplitude), size = 3, alpha = 0.7) +
      
      scale_colour_gradientn(name = "Amplitude", colours = col_temp, values=seq(0,1,0.01),
                             space = "Lab" , guide = "colourbar", # , na.value = "grey50"
                             aesthetics = "colour", limits = c(min(light$amplitude, na.rm = TRUE), max(light$amplitude, na.rm = TRUE))) +
      #aesthetics = "colour", limits = c(-5,45)) +
      # ggplot2::scale_colour_stepsn(name = "Temperature [°C]", colours = col_temp,n.breaks=50,nice.breaks=TRUE,trans="identity",
      #                       space = "Lab" , guide = "coloursteps", # , na.value = "grey50"
      #                       #aesthetics = "colour", limits = c(min(AirTemperatureInterpolation$temperature), max(AirTemperatureInterpolation$temperature))) +
      #                      aesthetics = "colour", limits = c(-5,45)) +
      #ggplot2::scale_colour_stepsn(name = "Temperature [°C]", n.breaks = 50, trans="log2",colours = col_temp,limits=c(-5,45)) +                            
      
      #ggplot2::scale_fill_stepsn(name = "Temperature [°C]",colors = col_temp, breaks = breaks) +                       
      
      #ggtitle(paste0("Urban heat island")) +
      ggtitle(paste0(" Lightning ", date_mid)) +
      
      xlab("Longitude [°]") +
      ylab("Latitude [°]") +
      theme(legend.title = element_text(size = 20),
            legend.text = element_text(size = 20),
            legend.key.size = unit(6, "cm"),
            legend.key.width = unit(1.5, "cm"),
            axis.text = element_text(size = 20),
            axis.title = element_text(size = 20),
            plot.title = element_text(size = 20, hjust = 0.5)) 
  )
  dev.off()
}


png_files <- png_files[c(58:84)]
png_files <- list.files(paste0("processing/plots/gif_test/"), pattern = ".png", full.names = TRUE)
gifski(png_files, gif_file = paste0("processing/plots/gif_test/LI_2021-06-23.gif"), width = 1000, height = 1000, delay = 0.5)






