import numpy as np
from scipy.integrate import dblquad 

def keldysh(v0=1.0,alpha=1.0,rcut=10.):
    """Keldysh potential from Phys. Rev. B 84, 085406"""

    def vf(r):
        r0 = alpha
        out = np.log(r/(r+r0)) + (0.5772 - np.log(2))*np.exp(-r/r0)
        return -v0*out/r0
    rs = np.linspace(1.,rcut,100)
    vr = [vf(ir) for ir in rs]
    
    from scipy.interpolate import interp1d
    
    fint = interp1d(rs,vr,fill_value=0.0,bounds_error=False)
#    return fint
    def fout(r1,r2=np.array([0.,0.,0.])):
        dr = r1-r2
        dr2 = dr.dot(dr)
        return fint(np.sqrt(dr2))
    return fout # return function
