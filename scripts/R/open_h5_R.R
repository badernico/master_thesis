library(ncdf4)
library(raster)
library(rasterVis)

ncfname <- '/Volumes/Elements/data/msevi_rss/example_20130618/raw_hdf/msg2-sevi-20130618t1000z-l15hdf-rss-eu.c2.h5'
nc_open(ncfname)

IR <- ncvar_get(nc_open(ncfname), 'l15_images/image_ir_108')
IR <- t(IR)
WV <- ncvar_get(nc_open(ncfname), 'l15_images/image_wv_062')
WV <- t(WV)

IR <- ncvar_get(nc_open(ncfname), 'geometry/satellite_zenith')

r_IR <- raster(IR)
levelplot(r_IR, margin = FALSE)

r_WV <- raster(WV)
levelplot(r_WV, margin = FALSE)

# convert mW m-2 sr-1 cm-1 to W m-2 sr-1 cm-1
IR <- (IR/1000)*nu_c_ir
WV <- (WV/1000)*nu_c_wv

IR <- a_ir + b_ir*IR
WV <- a_wv + b_wv*WV

r_IR <- raster(IR)
levelplot(r_IR, margin = FALSE)

r_WV <- raster(WV)
levelplot(r_WV, margin = FALSE)


OLR <- 11.44*IR+9.04*WV+((9.11*WV)/IR)-(86.36/IR)-0.14*(WV^2)+111.12
OLR <- abs(OLR)

r_OLR <- raster(OLR)
levelplot(r_OLR, margin = FALSE)


# testing dataset tobac

ncfname <- 'Documents/Uni_Leipzig/tobac/tobac-tutorials/themes/tobac_v1/climate-processes-tobac_example_data-b3e69ee/data/Example_input_OLR_satellite.nc'
nc_open(ncfname)

OLR2 <- ncvar_get(nc_open(ncfname), 'olr')

#lon <- ncvar_get(nc_open(ncfname), 'lon')
#lat <- ncvar_get(nc_open(ncfname), 'lat')
#time <- ncvar_get(nc_open(ncfname), 'time')

OLR2 <- t(OLR2[,,1])
r_OLR2 <- raster(OLR2)

levelplot(r_OLR2, margin = FALSE)

hist(OLR)


####
h = 6.626*10^-34
c = 299792458
k = 1.381*10^-23

C1 <- 2*h*c^2
C2 <- (h*c)/k

a_ir <- 0.9983
b_ir <- 0.64
nu_c_ir <- 931.7

a_wv <- 0.9963
b_wv <- 2.185
nu_c_wv <- 1600.548

T_b_ir <- ((C2*nu_c_ir)/(a_ir*log((C1*nu_c_ir^3)/(IR+1))))-(b_ir/a_ir)


plot(T_b_ir)
