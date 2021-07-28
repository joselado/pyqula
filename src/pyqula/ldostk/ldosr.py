import numpy as np
from .. import sculpt
from ..dos import calculate_dos


# specialized routines to compute the LDOS in continuum space

def ldosr_generator(h,rs=0.2,es=np.linspace(-1.,1.,100),
        delta=3e-2,nn=20,sector="electron",**kwargs):
    """Return a function that computes the LDOS as a function of the
    position"""
    (evals,vs) = h.get_eigenvectors(**kwargs,kpoints=False)
    ds = np.array([(np.conjugate(v)*v).real for v in vs]) # densities
    del vs # remove vectors
    g = h.geometry
    def fun(r):
        """Function to compute the DOS at position r"""
        r = np.array(r)
        inds = sculpt.get_closest(g,n=nn,r0=r) # indexes
        drs = [g.r[i]-r for i in inds] # distances
        dr = np.array([np.sqrt(ir.dot(ir)) for ir in drs]) # distances
        ws = np.exp(-dr/rs) # relative weight
        ws = ws/np.sum(ws) # normalize
        yout = 0. # initialize
        for i in range(len(inds)):
            ii = inds[i] # get this index
            if h.check_mode("spinless"):
              yi = calculate_dos(evals,es,delta,w=ds[:,ii])
              yout = yi*ws[i] # multiply by the weight
            elif h.check_mode("spinful"):
              yout = yout + calculate_dos(evals,es,delta,w=ds[:,2*ii])*ws[i]
              yout = yout+ calculate_dos(evals,es,delta,w=ds[:,2*ii+1])*ws[i]
            elif h.check_mode("spinful_nambu"):
              yout = yout + calculate_dos(evals,es,delta,w=ds[:,4*ii])*ws[i]
              yout = yout+ calculate_dos(evals,es,delta,w=ds[:,4*ii+1])*ws[i]
              if sector=="all":
                yout = yout + calculate_dos(evals,es,delta,w=ds[:,4*ii+2])*ws[i]
                yout = yout+ calculate_dos(evals,es,delta,w=ds[:,4*ii+3])*ws[i]
            else: raise
        return (es,yout)
    return fun







