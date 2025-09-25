import numpy as np
from . import algebra
from . import parallel




def get_fermi_energy(es,filling,fermi_shift=0.0,
        e_reg = 1e-5 # energy regularization for fully filled/empty
        ):
  """Return the Fermi energy"""
  ne = len(es) ; ifermi = int(round(ne*filling)) # index for fermi
  sorte = np.sort(es) # sorted eigenvalues
  if ifermi>=ne: return sorte[-1] + fermi_shift + e_reg
  elif ifermi==0: return sorte[0] + fermi_shift - e_reg
  else:
      fermi = (sorte[ifermi-1] + sorte[ifermi])/2.+fermi_shift # fermi energy
      return fermi




def eigenvalues(h0,nk=10,notime=True):
    """Return all the eigenvalues of a Hamiltonian"""
    from . import klist
    h = h0.copy() # copy hamiltonian
    h = h.get_dense()
    ks = klist.kmesh(h.dimensionality,nk=nk) # get grid
    hkgen = h.get_hk_gen() # get generator
    if parallel.cores==1:
      es = [] # empty list
      if not notime: est = timing.Testimator(maxite=len(ks))
      for k in ks: # loop
          if not notime: est.iterate()
          es += algebra.eigvalsh(hkgen(k)).tolist() # add
    else:
        f = lambda k: algebra.eigvalsh(hkgen(k)) # add
        es = parallel.pcall(f,ks) # call in parallel
        es = np.array(es)
        es = es.reshape(es.shape[0]*es.shape[1])
    return es # return all the eigenvalues


def set_filling(h,average=True,**kwargs):
    """Function to set the filling"""
    if average:
        return set_average_filling(h,**kwargs)
    else:
        return set_individual_filling(h,**kwargs)


def set_average_filling(h,filling=0.5,nk=10,extrae=0.,
    mode="ED",**kwargs):
    """
    Set the filling of a Hamiltonian
    - nk = 10, number of kpoints in each direction
    - filling = 0.5, filling of the lattice
    - extrae = 0.0, number of extra electrons
    """
    if h.has_eh: # quick workaround
        ef = h.get_fermi4filling(filling,nk=nk) # fermi energy
        h.add_onsite(-ef)
        return
    fill = filling + extrae/h.intra.shape[0] # filling
    n = h.intra.shape[0]
    use_kpm = False
    if n>algebra.maxsize: # use the KPM method
        mode="KPM"
        print("Using KPM in set_filling")
    if mode=="KPM": # use KPM
        es,ds = h.get_dos(energies=np.linspace(-5.0,5.0,1000),
                mode="KPM",nk=nk,**kwargs)
        from scipy.integrate import cumtrapz
        di = cumtrapz(ds,es)
        ei = (es[0:len(es)-1] + es[1:len(es)])/2.
        di /= di[len(di)-1] # normalize
        from scipy.interpolate import interp1d
        f = interp1d(di,ei) # interpolating function
        efermi = f(fill) # get the fermi energy
    elif mode=="ED": # dense Hamiltonian, use ED
        es = eigenvalues(h,nk=nk,notime=True)
        efermi = get_fermi_energy(es,fill)
    else: raise
    h.shift_fermi(-efermi) # shift the fermi energy



def set_individual_filling(h,filling=0.5,**kwargs):
    """Set the fillings of all the sites"""
    def fmin(ons):
        """Function to solve"""
        hi = h.copy()
        hi.add_onsite(ons) # add these onsites
        out = hi.get_vev(delta=1e-2,**kwargs) # output fillings
        return out - filling
    x0 = np.zeros(len(h.geometry.r) ) # initial guess
    from scipy.optimize import fsolve
    x = fsolve(fmin,x0,xtol=1e-5,factor=1.)
    h.add_onsite(x)
    return h









