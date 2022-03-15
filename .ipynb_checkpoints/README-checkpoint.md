# Master thesis
## Convective strength and thunderstorm potential depending on conditions prior to convective initiation (CI)

This is my repository for my Master thesis at the Leibniz Institute for Tropospheric Research.

### Installation
To work within this project i've worked within a conda environment.

- Install the conda environment `conda create -n satenv`
- Activate the conda environment `conda activate satenv`
- Install the required modules `pip install -r requirements.txt`

### Convert satellite data to different netCDF4 files

- Download your MSG SEVIRI (Rapid Scan) data in Native format (.nat) from the Eumetsat [Earth Observation Portal](https://eoportal.eumetsat.int/cas/login?service=https%3A%2F%2Feoportal.eumetsat.int%2FuserMgmt%2Fcallback%3Fclient_name%3DCasClient)
- Convert the native format to netCDF format in units *counts* with the script *msevi_nat_to_netcdf.py*
- Convert the native format to netCDF format in units *Brighntess Temperature* with the script *msevi_nat_to_tb.py*

Afterwards the 5-min netCDF files can be saved as a one-day netCDF file (as tobac input):
- Convert to a one-day netCDF file with *Brightness Temperature* with the script *msevi_tobac_tb.py*
- Convert to a one-day netCDF file with *OLR* with the script *msevi_tobac_OLR.py*

