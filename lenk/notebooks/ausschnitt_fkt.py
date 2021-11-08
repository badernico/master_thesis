import numpy as np

def ausschnitt(daten,nachbarschaft):
    datenvektor = daten.flatten()
    bereich = datenvektor[len(datenvektor)/2-(nachbarschaft**2/2):len(datenvektor)/2+(nachbarschaft**2/2)+1]
    bereich = np.reshape(bereich,(nachbarschaft,nachbarschaft))
    return bereich
