import numpy as np
from ausschnitt_fkt import ausschnitt
def feldstatistik(felddaten,nachbarschaft):
    feld_min = []
    feld_q25 = []            
    feld_med = []
    feld_mit = []
    feld_q75 = []      
    feld_max = []
    
    for f in felddaten:
        feld_ausschnitt = ausschnitt(f,nachbarschaft)
        feld_min.append(np.min(feld_ausschnitt))
        feld_q25.append(np.percentile(feld_ausschnitt,25))
        feld_mit.append(np.median(feld_ausschnitt))
        feld_med.append(np.percentile(feld_ausschnitt,50))
        feld_q75.append(np.percentile(feld_ausschnitt,75))
        feld_max.append(np.max(feld_ausschnitt))
        
    ausgabe = {'min':feld_min,
               'q25':feld_q25,
               'med':feld_med,
               'mit':feld_mit,
               'q75':feld_q75,
               'max':feld_max}
               
    return ausgabe  
