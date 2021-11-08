#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 16:19:46 2018

@author: lenk
"""
import pandas as pd

folder = "/vols/satellite/home/lenk/proj/2017-12_radartracks/"

if __name__ == "__main__":
    case_days = pd.read_table("{}falltage_2013_manuell.txt".format(folder),header=None)
    
    tlist = []
    for i,day in case_days.iterrows():
        day_str = str(day[0])
        start = "{}-{}-{}t00:00".format(day_str[0:4],day_str[4:6],day_str[6:8])
        end = "{}-{}-{}t23:55".format(day_str[0:4],day_str[4:6],day_str[6:8])
        
        time_range = pd.date_range(start=start, end = end,freq="5min")
        time_range = time_range.format(formatter=lambda x: x.strftime('%Y%m%dT%H%MZ'))
    
        tlist.extend(time_range)
    
    time = pd.DataFrame(tlist)
    time.to_csv("{}times_cases_2013.list".format(folder),index=False,header=None)
