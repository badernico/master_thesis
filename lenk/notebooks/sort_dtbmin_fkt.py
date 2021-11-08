import numpy as np
from ausschnitt_fkt import ausschnitt

def sort_dtbmin(trackdaten,dt):
    dtb = []
    
    for i in range(1,len(trackdaten['ir108'])):
        dtb.append(np.median(ausschnitt(trackdaten['ir108'][i-1],3)-np.median(ausschnitt(trackdaten['ir108'][i],3))))
       
    # Index des ersten Zeitschrittes wo Differenz maximal wird, also die Ab-
    # kuehlung am groeszten ist
    try:
        kuehl_idx = np.where(dtb == np.max(dtb))
        
        zeit_nach = np.arange(0,len(trackdaten['ir108'])-kuehl_idx[0])
        zeit_vor = np.arange(-kuehl_idx[0],0)
        
        rel_zeit = np.concatenate((zeit_vor,zeit_nach))
        
        return rel_zeit * dt
    except:
        return
