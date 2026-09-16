import numpy as np
from . import algebra
from . import parallel




def check_filling(filling):
  """Complain if the filling is not a fraction of the occupied states.

  The convention throughout pyqula is that filling is the fraction of
  *all* the states of the Hamiltonian that are occupied, so it must lie
  in [0,1] (half filling is 0.5, both for spinful and spinless
  Hamiltonians). Values outside that range used to be accepted silently:
  a negative filling wrapped around through negative indexing and
  returned the Fermi energy of filling 1+f. A per-site array of fillings
  is checked element by element, with the same convention per site."""
  if filling is None: return # nothing to check
  f = np.asarray(np.real(filling),dtype=float)
  if not np.all(np.isfinite(f)) or np.any(f<0.0) or np.any(f>1.0):
      raise ValueError("filling must be a fraction of the total number of "
        +"states, i.e. in [0,1] (half filling is 0.5), got "+str(filling)
        +". If you meant electrons per site, divide by the number of "
        +"states per site.")


def get_fermi_energy(es,filling,fermi_shift=0.0,
        e_reg = 1e-5 # energy regularization for fully filled/empty
        ):
  """Return the Fermi energy"""
  check_filling(filling) # complain about a meaningless filling
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
    from .htk.eigenvectors import peigvalsh, hk_matrix_batch
    h = h0.copy() # copy hamiltonian
    h = h.get_dense()
    ks = klist.kmesh(h.dimensionality,nk=nk) # get grid
    hkgen = h.get_hk_gen() # get generator
    mats = hk_matrix_batch(hkgen,ks) # H(k) batch, densified
    es = peigvalsh(mats) # batched numba eigh
    es = es.reshape(es.shape[0]*es.shape[1])
    return es # return all the eigenvalues


def set_filling(h,filling=0.5,average=True,**kwargs):
    """Function to set the filling.

    A scalar filling is enforced on average (average=True, one shift of
    the Fermi energy) or on every site (average=False, one onsite energy
    per site). An array of fillings, one per site, can only be enforced
    site by site, so it always takes the second route."""
    if np.ndim(filling)>0: # per-site fillings
        n = len(h.geometry.r) # number of sites
        if len(filling)!=n:
            raise ValueError("a per-site filling needs one value per site, "
                    +"got "+str(len(filling))+" values for "+str(n)+" sites")
        return set_individual_filling(h,filling=np.asarray(filling),
                **kwargs)
    if average:
        return set_average_filling(h,filling=filling,**kwargs)
    else:
        return set_individual_filling(h,filling=filling,**kwargs)


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
    else:
        raise ValueError("unknown mode; set_average_filling accepts 'KPM' and "
                "'ED'")
    h.shift_fermi(-efermi) # shift the fermi energy



def set_individual_filling(h,filling=0.5,**kwargs):
    """Set the fillings of all the sites.

    `filling` keeps the convention of check_filling -- the fraction of all
    the states that are occupied, in [0,1] -- while get_vev returns an
    occupancy per site, which runs to 2 for a spinful Hamiltonian. The two
    used to be compared directly, so the solver aimed at half the
    occupancy it should have on every spinful system. `filling` is either
    a scalar, the same on every site, or an array with one value per
    site."""
    check_filling(filling) # complain about a meaningless filling
    # states per site, counting only the electron sector: get_vev already
    # restricts a Nambu Hamiltonian to it
    nper = 2 if h.has_spin else 1
    target = filling*nper # occupancy per site the solver aims at
    def fmin(ons):
        """Function to solve"""
        hi = h.copy()
        hi.add_onsite(ons) # add these onsites
        out = hi.get_vev(delta=1e-2,**kwargs) # output fillings
        return out - target
    x0 = np.zeros(len(h.geometry.r) ) # initial guess
    from scipy.optimize import fsolve
    x = fsolve(fmin,x0,xtol=1e-5,factor=1.)
    h.add_onsite(x)
    return h









