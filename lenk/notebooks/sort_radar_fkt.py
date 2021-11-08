import numpy as np
from ausschnitt_fkt import ausschnitt
def sort_radar(trackdaten,dt):
    rx_max = []
    
    for i in range(0,len(trackdaten['radar'])):
        radarmax = np.max(ausschnitt(trackdaten['radar'][i],9))
        rx_max.append(radarmax)
    
    # Index des ersten Zeitschrittes mit mehr als 35 dBZ finden 
    try:
        sw_idx = np.min(np.where(np.array(rx_max)>35))
        
        zeit_nach = np.arange(0,len(trackdaten['radar'])-sw_idx)
        zeit_vor = np.arange(-sw_idx,0)
        
        rel_zeit = np.concatenate((zeit_vor,zeit_nach))
        
        return rel_zeit * dt
    except:
        return
