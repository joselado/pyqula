import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
# suppress the numpy warning on arrays


# library to target ingap states


def energy_ingap_state(EO,ne=51,**kwargs):
    """Return the energy of an in-gap state given an embedding object"""
    ev = EO.H.get_gap(mode="valence")
    ec = EO.H.get_gap(mode="conduction")
    es = np.linspace(-ev*0.95,ec*0.95,ne) # number of energies
    delta = (ev+ec)/ne # delta
    (es,ds) = EO.multidos(es=es,delta=delta,**kwargs) # return DOS
    fd = interp1d(np.array(es),-np.array(ds),bounds_error=False,
                         fill_value=(ds[0],ds[-1]),
                        kind="quadratic")
    res = minimize(fd,np.array([0.0]),bounds=np.array([[min(es),max(es)]]),
                          method="Powell")
    return res.x[0]

