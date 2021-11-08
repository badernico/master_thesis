
import sys
import numpy as np
import pandas as pd
import xarray as xr

import glob
import matplotlib.pyplot as plt
plt.switch_backend("Agg")

import glob


def array_to_256(array):
    return (array*255.999).astype("uint8")

def day_natural_composite(vis006_data,vis008_data,nir016_data,factor=1,gamma=1):
    blue = array_to_256(np.clip(vis006_data/factor,0,1)**(1./gamma))
    green = array_to_256(np.clip(vis008_data/factor,0,1)**(1./gamma))
    red = array_to_256(np.clip(nir016_data/factor,0,1)**(1./gamma))
    
    return np.dstack([red,green,blue]).astype("uint8")



talos_home = "/vols/talos/home/stephan"
data_path = "{talos_home}/data".format(talos_home=talos_home)
track_data_path = "{dp}/radar_track/trackdata".format(dp=data_path)

out_dir = "{}/proj/2019-01_trackingstudie/pics/nc_rgbs/".format(talos_home)

if __name__ == "__main__":
    track_paths = glob.glob("{tdp}/*.nc".format(tdp=track_data_path))

    for tr in track_paths:
        track_data = xr.open_dataset(tr)

        tr_id = "{}_{}".format(tr.split("/")[-1].split(".")[0].split("_")[-2],
                               tr.split("/")[-1].split(".")[0].split("_")[-2])


        nc_rgbs = np.array([day_natural_composite(track_data.vis006.data[i],
                                                  track_data.vis008.data[i],
                                                  track_data.ir016.data[i],0.9,1.8) 
                            for i in xrange(len(track_example.ir108.data))])

        fig,ax = plt.subplots(4,4,figsize=(16,16))
        axs=ax.ravel()
        for i, n in enumerate(nc_rgbs):
            axs[i].imshow(n)
            axs[i].set_title(pd.to_datetime(track_example.time.data[i]))
        fig.title('NC-RGB, Track {}'.format(tr_id))

        plt.savefig("{}/{}.png".format(out_dir,tr_id))
        plt.close()

