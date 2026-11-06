from __future__ import print_function
from . import filewrite
import scipy.sparse.linalg as slg
import scipy.linalg as lg
from scipy.sparse import csc_matrix as csc
from scipy.sparse import csc_matrix 
from scipy.sparse import bmat
import os
from numba import jit
import numpy as np
from . import klist
from . import operators
from . import timing
from . import parallel
from . import algebra
from .increase_hilbert import full2profile as spatial_dos
from . import filesystem as fs
from .algebra import dagger

def ldos0d(h,e=0.0,delta=0.01,write=None):
  """Calculates the local density of states of a Hamiltonian and
     writes it in file"""
  write = filewrite.resolve(write,True) # the call, else the global switch
  if h.dimensionality==0:  # only for 0d
    iden = np.identity(h.intra.shape[0],dtype=np.complex128) # create identity
    g = algebra.inv( (e+1j*delta)*iden -h.intra ) # calculate green function
  else:
    raise NotImplementedError("ldos0d is only for 0d Hamiltonians")
  d = [ -(g[i,i]).imag/np.pi for i in range(len(g))] # get imaginary part
  d = spatial_dos(h,d) # convert to spatial resolved DOS
  g = h.geometry  # store geometry
  if write: write_ldos(g.x,g.y,d,z=g.z) # write in file
  return d



def dos_site_kpm(h,energies=np.linspace(-1.,1.,1000),
        delta=0.01,scale = 10.0,i=0,nk=5,sector=None):
    """Local DOS at site i from a Chebyshev expansion seeded with the site
    vector, summed over the spin and Nambu components of the site and
    averaged over an nk k-mesh.

    delta: energy resolution, the half width at half maximum of the peak a
    single level gives, as the Lorentzian of the exact-diagonalization modes
    (see kpmtk.kernels.jackson_npol). It sets the number of polynomials,
    about 1.85 scale/delta, and the width narrows as sqrt(1-(E/scale)^2)
    away from E=0. The Jackson kernel makes each peak nearly a Gaussian,
    about 1.5 times taller than the Lorentzian of the same half width.
    scale: the whole spectrum must lie inside [-scale,scale]
    sector: None for the whole Nambu site, "electron" for its electron part"""
    h = h.copy()
    h.turn_sparse()
    from .kpmtk.kernels import jackson_npol
    npol = jackson_npol(scale,delta) # number of polynomials
    from . import kpm
    def getm(m,j): # for the matrix
      es,ds = kpm.ldos(m,i=j,scale=scale,npol=npol,ne=npol*5)
      return es,ds # return
    def get(j): # for the kpoint
        ks = klist.kmesh(h.dimensionality,nk=nk)
        hk = h.get_hk_gen() # get generator
        out = [getm(hk(k),j) for k in ks] # compute all
        ds = np.mean([o[1] for o in out],axis=0)
        es = out[0][0]
        return es,ds
#        if h.dimensonality!=0: raise # not implemented
#        return getm(h.intra)
    from scipy.interpolate import interp1d
    if h.has_spin and h.has_eh: # spin with electron-hole
        if sector is None:
            (es,ds1) = get(4*i)
            (es,ds2) = get(4*i+1)
            (es,ds3) = get(4*i+2)
            (es,ds4) = get(4*i+3)
            ds = ds1+ds2+ds3+ds4
        elif sector=="electron":
            (es,ds1) = get(4*i)
            (es,ds2) = get(4*i+1)
            ds = ds1+ds2
        else:
          raise ValueError("unknown sector for a Nambu Hamiltonian; the "
                  "accepted ones are None and 'electron'")
    elif h.has_spin and not h.has_eh: # spinful
        (es,ds1) = get(2*i)
        (es,ds2) = get(2*i+1)
        ds = ds1+ds2
    elif not h.has_spin and not h.has_eh: # spinless
        (es,ds) = get(i)
    f = interp1d(es,ds.real,bounds_error=False,fill_value=0.0)
    return energies,f(energies)





def dos_site(h,i=0,mode="ED",energies=np.linspace(-1.,1.,500),**kwargs):
    """DOS in a particular site for different energies"""
    if mode=="ED":
      if h.dimensionality!=0: # only for 0d
        raise ValueError("the exact-diagonalization site DOS is only "
                "implemented for 0d Hamiltonians")
      out = []
      for e in energies:
          d = ldos0d(h,e=e,write=False,**kwargs)
          out.append(d[i]) # store
      return (energies,np.array(out)) # return result
    elif mode=="KPM":
        return dos_site_kpm(h,energies=energies,i=i,**kwargs)






def ldos0d_wf(h,e=0.0,delta=0.01,num_wf = 10,robust=False,tol=0,write=None):
  """Calculates the local density of states of a hamiltonian and
     writes it in file, using arpack"""
  write = filewrite.resolve(write,True) # the call, else the global switch
  if h.dimensionality==0:  # only for 0d
    intra = csc_matrix(h.intra) # matrix
  else:
    raise NotImplementedError("ldos0d_wf is only for 0d Hamiltonians")
  if robust: # go to the imaginary axis for stability
    eig,eigvec = slg.eigs(intra,k=int(num_wf),which="LM",
                        sigma=e+1j*delta,tol=tol) 
    eig = eig.real # real part only
  else: # Hermitic Hamiltonian, orthonormal in degenerate levels
    eig,eigvec = algebra.arpack_eigh(intra,k=int(num_wf),which="LM",
                                         sigma=e,tol=tol)
  d = np.array([0.0 for i in range(intra.shape[0])]) # initialize
  for (v,ie) in zip(eigvec.transpose(),eig): # loop over wavefunctions
    v2 = (np.conjugate(v)*v).real # square of wavefunction
    fac = delta/((e-ie)**2 + delta**2) # factor to create a delta
    d += fac*v2 # add contribution
#  d /= num_wf # normalize
  d /= np.pi # normalize
  d = spatial_dos(h,d) # resum if necessary
  g = h.geometry  # store geometry
  if write: write_ldos(g.x,g.y,d,z=g.z) # write in file
  return g.x,g.y,d




def ldos_arpack(intra,num_wf=10,robust=False,tol=0,e=0.0,delta=0.01):
  """Use arpack to calculate hte local density of states at a certain energy"""
  if robust: # go to the imaginary axis for stability
    eig,eigvec = slg.eigs(intra,k=int(num_wf),which="LM",
                        sigma=e+1j*delta,tol=tol) 
    eig = eig.real # real part only
  else: # Hermitic Hamiltonian, orthonormal in degenerate levels
    eig,eigvec = algebra.arpack_eigh(intra,k=int(num_wf),which="LM",
                                         sigma=e,tol=tol)
  d = np.array([0.0 for i in range(intra.shape[0])]) # initialize
  for (v,ie) in zip(eigvec.transpose(),eig): # loop over wavefunctions
    v2 = (np.conjugate(v)*v).real # square of wavefunction
    fac = delta/((e-ie)**2 + delta**2) # factor to create a delta
    d += fac*v2 # add contribution
#  d /= num_wf # normalize
  d /= np.pi # normalize
  return d




from .ldostk.ldoswaves import ldos_diagonalization
from .ldostk.ldoswaves import ldos_waves




def ldosmap(h,energies=np.linspace(-1.0,1.0,40),delta=None,
        nk=40,operator=None,integration="grid",tolerance=1e-6,**kwargs):
  """Write a map of the ldos using full diagonalization

  integration: one of get_integration_names(). "grid" averages the
  Lorentzian-broadened eigenstates over nk random k-points, with delta
  the energy spacing by default. "triangle" and "qtci" use instead the
  uniform nk^d mesh and the simplex integration of get_ldos, with
  delta=0 (the exact limit) by default."""
  if integration!="grid":
    if kwargs:
      raise TypeError("ldosmap with integration='"+str(integration)+"' got "
              "the unexpected arguments "+", ".join(sorted(kwargs)))
    ds,mesh = simplex_ldos(h,energies,integration=integration,nk=nk,
            delta=0. if delta is None else delta,operator=operator,
            tolerance=tolerance)
    return energies,ds
  if h.dimensionality == 0: nk = 1
  if delta is None:
    delta = (np.max(energies)-np.min(energies))/len(energies) # delta
  hkgen = h.get_hk_gen() # get generator
  dstot = np.zeros((len(energies),h.intra.shape[0])) # initialize
  operator = h.get_operator(operator)
  def getd(k): # get LDOS
    hk = hkgen(k) # get Hamiltonian
    # LDOS for this kpoint
    ds = ldos_waves(hk,k=k,es=energies,delta=delta,operator=operator,**kwargs) 
    return ds
  ks = [np.random.random(3) for ik in range(nk)] # kpoints
  if kwargs.get("num_bands") is None and not kwargs.get("non_hermitian",False):
      # batched, numba-parallel path -- no interprocess dispatch
      from .htk.eigenvectors import peigh_bloch
      from .ldostk.ldoswaves import ldos_waves_from_eigsystem
      delta_discard = kwargs.get("delta_discard")
      ds = []
      batch_size = 64
      for i0 in range(0,len(ks),batch_size): # loop over batches of kpoints
          kbatch = ks[i0:i0+batch_size]
          es_batch,vs_batch = peigh_bloch(hkgen,kbatch) # diagonalize the batch in parallel
          for ii,k in enumerate(kbatch):
              eigvec = vs_batch[ii].T # rows are eigenvectors
              ds.append(ldos_waves_from_eigsystem(es_batch[ii],eigvec,energies,
                      delta,operator=operator,k=k,delta_discard=delta_discard))
  else:
      ds = parallel.pcall(getd,ks) # get densities
  dstot = np.mean(ds,axis=0) # average over first axis
  print("LDOS finished")
  dstot = [spatial_dos(h,d) for d in dstot] # convert to spatial resolved DOS
  return energies,np.array(dstot)



def spatial_energy_profile(h,write=None,**kwargs):
  """Computes the DOS for each site of an slab, only for 2d"""
  write = filewrite.resolve(write,True) # the call, else the global switch
  if h.dimensionality==0:
      pos = h.geometry.r[:,0]
  elif h.dimensionality==1: pos = h.geometry.r[:,1]
  elif h.dimensionality==2: pos = h.geometry.r[:,2]
  else:
    raise ValueError("the spatial energy profile is only implemented for "
            "Hamiltonians up to 2d")
  es,ds = ldosmap(h,**kwargs)
  if len(ds[0])!=len(pos): 
    raise ValueError("the LDOS map has "+str(len(ds[0]))+" entries and the "
            "list of positions has "+str(len(pos)))
  if write:
    f = open("DOSMAP.OUT","w")
    f.write("# energy, index, DOS, position\n")
    for ie in range(len(es)):
      for ip in range(len(pos)):
        f.write(str(es[ie])+"  ")
        f.write(str(ip)+"  ")
        f.write(str(ds[ie,ip])+"   ")
        f.write(str(pos[ip])+"\n")
    f.close()
  return es,np.transpose(ds) # retunr LDOS 



slabldos = spatial_energy_profile # redefine





def ldos1d(h,e=0.0,delta=0.001,nrep=3,write=None):
  """ Calculate DOS for a 1d system"""
  write = filewrite.resolve(write,True) # the call, else the global switch
  from . import green
  if h.dimensionality!=1: # only for 1d
    raise ValueError("ldos1d is only for 1d Hamiltonians")
  if h.is_multicell: h = h.get_no_multicell() # the decimation needs h.inter
  gb,gs = green.green_renormalization(h.intra,h.inter,energy=e,delta=delta)
  d = [ -(gb[i,i]).imag for i in range(len(gb))] # get imaginary part
  d = spatial_dos(h,d) # convert to spatial resolved DOS
  g = h.geometry  # store geometry
  x,y = g.x,g.y # get the coordinates
  go = h.geometry.copy() # copy geometry
  go = go.supercell(nrep) # create supercell
  if write: write_ldos(go.x,go.y,d.tolist()*nrep) # write in file
  return d


def ldos_projector(h,e=0.0,**kwargs):
    """Return an operator to project onto that region"""
    # nrep=1 is essential: the profile returned by get_ldos_tb is written
    # on an nrep-replicated supercell grid (nrep defaults to 5), and the
    # operator has to live in the Hilbert space of the unit cell instead
    (x,y,d) = ldos(h,e=e,mode="arpack",silent=True,write=False,nrep=1,
            **kwargs)
    inds = np.array(range(len(d))) # indexes
    n = len(d)
    d = d/np.sum(d) # normalize
    m = csc_matrix((d,(inds,inds)),shape=(n,n),dtype=np.complex128) # matrix
    m = h.spinless2full(m) # to full matrix
    return operators.Operator(m) # convert to operator

def ldos_density(h,**kwargs):
    """Return a normalized profile with the DOS"""
    # nrep=1 for the same reason as in ldos_projector: this is one value
    # per site of the unit cell, not of an nrep-replicated supercell
    (x,y,d) = ldos(h,mode="arpack",silent=True,write=False,nrep=1,**kwargs)
    d = d/np.sum(d) # normalize
    return d


def ldos_potential(h,**kwargs):
    """Return a function that evaluates an LDOS profile"""
    raise NotImplementedError("ldos_potential is not implemented; use "
            "ldos_density for a normalized LDOS profile, or "
            "ldos_projector for an operator projecting onto it")


def get_ldos(h,**kwargs):
    """Master method for LDOS"""
    from .utilities import rename_kwarg
    # `energy` is how the Green's function, embedding and transport
    # routines spell it, so accept it here as well instead of dropping it
    kwargs = rename_kwarg(kwargs,"energy","e")
    if not h.non_hermitian: # Hermitian case
        return get_ldos_general(h,**kwargs)
    else:
        from .nonhermitiantk.nhmethods import get_ldos as get_ldos_NH
        return get_ldos_NH(h,**kwargs)


def get_ldos_general(h,projection="TB",**kwargs):
    """ Calculate LDOS with different methods"""
    if projection=="TB": return get_ldos_tb(h,**kwargs)
    elif projection=="TBRS": return get_ldos_tb(h,interpolate=True,**kwargs)
    elif projection=="atomic": 
        from .ldostk import atomicmultildos
        return atomicmultildos.get_ldos(h,**kwargs)
    else:
      raise ValueError("unknown projection; get_ldos accepts 'TB', 'TBRS' and "
              "'atomic'")


def green2ldos(g,op=None):
    """Local density of states read off a Green's function, optionally
    resolved with an operator A as -Im(diag(G A + A G))/(2 pi).

    Only the trace of diag(G A) does not depend on the ordering: site by
    site, diag(G A) and diag(A G) are complex conjugates of one another,
    so each carries a spurious piece that the other cancels -- large
    enough to give, for instance, a nonzero sy-resolved map on a real
    Hamiltonian, which has no sy at all. The Hermitian combination keeps
    only the Lorentzian-weighted Re<psi|A|psi>_i, which is the local
    spectral weight, and still sums over sites to the operator-resolved
    DOS that green.green_operator returns. Without an operator this is
    the ordinary -Im(diag(G))/pi.

    This is the same which-matrix-goes-first question that spectrum.ev
    and vev.get_dm_vev had to settle (fbee7c9, 7087ee6)."""
    g = np.array(g)
    if op is None: return -np.diag(g).imag/np.pi
    return -np.diag(g@op + op@g).imag/(2.*np.pi)


def _ldos_triangle(h,energies,nk=None,delta=0.,operator=None,
        tolerance=None):
    from .dostk.simplexdos import ldos_triangle_counted
    out = ldos_triangle_counted(h,energies,nk=100 if nk is None else nk,
            delta=delta,operator=operator)
    return out[1],out[3]


def _ldos_qtci(h,energies,nk=None,delta=0.,operator=None,tolerance=1e-6):
    from .dostk.simplexdos import ldos_qtci_counted
    out = ldos_qtci_counted(h,energies,nk=128 if nk is None else nk,
            delta=delta,operator=operator,tolerance=tolerance)
    return out[1],out[3]


_integrations = {"grid":None, # the Lorentzian sum over a k-mesh of each mode
        "triangle":_ldos_triangle,
        "qtci":_ldos_qtci,
        }


def get_integration_names():
    """Accepted values of integration= in get_ldos, multi_ldos and
    ldosmap"""
    return list(_integrations)


def check_integration(integration):
    if integration not in _integrations:
        raise ValueError("unknown integration '"+str(integration)+"' for "
                "the LDOS; it must be one of "+str(get_integration_names()))


def simplex_ldos(h,energies,integration="triangle",nk=None,delta=0.,
        operator=None,tolerance=1e-6):
    """The LDOS of every site at every energy from a simplex integration
    (not "grid"), as a (len(energies),nsites) array, and the band mesh
    that gives the DOS with no new diagonalization"""
    check_integration(integration)
    if integration=="grid":
        raise ValueError("simplex_ldos takes a simplex integration, one of "
                +str([n for n in get_integration_names() if n!="grid"]))
    if operator is not None:
        operator = h.get_operator(operator)
        if operator.get_matrix(required=False) is None:
            raise NotImplementedError("integration='"+str(integration)+"' "
                    "weights the bands at the vertices of the mesh with the "
                    "matrix of the operator, so it cannot take a "
                    "k-dependent operator; use integration='grid'")
    return _integrations[integration](h,np.asarray(energies,dtype=float),
            nk=nk,delta=delta,operator=operator,tolerance=tolerance)


def get_ldos_tb(h,e=0.0,delta=None,nrep=5,nk=None,ks=None,mode="arpack",
             random=False,silent=True,interpolate=False,
             operator=None,return_rd = False,
             write=None,integration="grid",tolerance=1e-6,**kwargs):
    """ Calculate LDOS in a tight binding basis

    mode: "arpack" (or "diagonalization") broadens the eigenstates of H(k)
        with a Lorentzian of half width delta (0.001 by default), "green"
        reads the Green's function of a 2d Hamiltonian, and "KPM" expands
        the LDOS of every orbital in Chebyshev polynomials with the Jackson
        kernel, in a time linear in the number of sites (see
        kpmtk/ldosmap.py), which is the mode for a large sparse
        Hamiltonian. There delta is the half width at half maximum of the
        peak a level gives, or npol=, the number of polynomials, sets it
        (200 when neither is given), and kpm_radius=, scale= and kpm_prec=
        are those of the KPM mean field: the map is exact for
        kpm_radius=None, and a smaller radius in hops keeps a finite
        cluster around each site.

    operator: None, or an operator spec (a name, a matrix, an Operator).
        The LDOS is then resolved with that operator instead of being the
        plain charge one. The two modes weight it differently, and both
        integrate over sites to the same operator-resolved DOS:
        mode="arpack" uses <psi|A|psi> times the local density |psi(i)|^2,
        while mode="green" uses the local matrix element
        Re<psi|A|psi>_i -- the genuinely local quantity, which is what
        differs for states that are not eigenstates of A. A k-dependent
        operator is only available in mode="arpack", since mode="green"
        has already integrated over the Brillouin zone.

    integration: how mode="arpack" (or "diagonalization") integrates over
        k, one of get_integration_names(). "grid" averages over the
        uniform nk mesh (nk=10 by default), and the mesh must resolve
        delta. "triangle" interpolates the bands and the weight of each
        band on each site linearly inside each segment, triangle or
        tetrahedron of the uniform nk^d mesh (nk=100 by default) and
        integrates the resolvent exactly, so the mesh only has to resolve
        the bands, and delta=0 (the default for this integration) is the
        exact limit; the site weights sum to the triangle DOS of
        h.get_dos(integration="triangle"). "qtci" is the same in 2D on a
        2^R mesh whose fields come from a quantics tensor cross
        interpolation with the given tolerance (nk=128 by default). Both
        take an operator with a matrix, and neither takes ks or random.
    """
    write = filewrite.resolve(write,True) # the call, else the global switch
    check_integration(integration)
    from .utilities import check_delta
    if integration=="grid":
        if delta is None and mode!="KPM": delta = 0.001 # Lorentzian
        if delta is not None: check_delta(delta)
        if ks is not None and mode=="green":
            raise ValueError("an explicit k-point list (ks) is incompatible "
              +"with mode='green', which integrates over the Brillouin zone")
    else: # the simplex integrations of the diagonalization mode
        if mode not in ("arpack","diagonalization"):
            raise ValueError("integration='"+str(integration)+"' applies "
                    "to mode='arpack' and mode='diagonalization', got "
                    "mode="+repr(mode))
        if kwargs and not interpolate: # the interpolation takes the rest
            raise TypeError("get_ldos with integration='"+str(integration)
                    +"' got the unexpected arguments "
                    +", ".join(sorted(kwargs)))
        if ks is not None or random:
            raise ValueError("ks and random only apply to integration='grid'"
                    ", got integration='"+str(integration)+"'")
        if delta is None: delta = 0. # the exact limit
    if operator is not None: operator = h.get_operator(operator)
    if integration!="grid": # one value per site already, spin summed
      ds,mesh = simplex_ldos(h,[e],integration=integration,nk=nk,
              delta=delta,operator=operator,tolerance=tolerance)
      d = ds[0]
    elif mode=="green":
      from . import green
      if h.dimensionality!=2: # only for 2d
        raise ValueError("the Green's function LDOS is only implemented for "
                "2d Hamiltonians")
      h = h.copy()
      h = h.get_dense()
      op = None # no operator
      if operator is not None:
          op = operator.get_matrix(required=False) # matrix of the operator
          if op is None: # momentum dependent operator
              raise NotImplementedError("mode='green' integrates over the "
                +"Brillouin zone before the operator is applied, so it "
                +"cannot take a k-dependent operator; use mode='arpack'")
          op = algebra.todense(op)
      if nk is not None:
        print("LDOS using normal integration with nkpoints",nk)
        gb,gs = green.bloch_selfenergy(h,energy=e,delta=delta,mode="full",nk=nk)
      else:
        print("LDOS using renormalization adaptative Green function")
        gb,gs = green.bloch_selfenergy(h,energy=e,delta=delta,mode="adaptive")
      d = green2ldos(gb,op=op) # local density of states
    elif mode=="arpack" or mode=="diagonalization": # arpack diagonalization
      from . import klist
      if nk is None: nk = 10
      hkgen = h.get_hk_gen() # get generator
      ds = [] # empty list
      if ks is None:
        ks = klist.kmesh(h.dimensionality,nk=nk)
        if random: ks = [np.random.random(3) for k in ks] # random mesh
      ts = timing.Testimator(title="LDOS",maxite=len(ks),silent=silent)
      for k in ks: # loop over kpoints
        ts.iterate()
        hk = hkgen(k) # get Hamiltonian
        ds += [ldos_diagonalization(hk,e=e,delta=delta,operator=operator,
                                    k=k,**kwargs)]
      d = np.mean(ds,axis=0) # average
    elif mode=="KPM": # Chebyshev expansion, linear in the number of sites
      from . import klist
      from .kpmtk.ldosmap import ldos_map
      if ks is None:
        ks = klist.kmesh(h.dimensionality,nk=10 if nk is None else nk)
        if random: ks = [np.random.random(3) for k in ks] # random mesh
      kpm = {key:kwargs.pop(key) for key in ["npol","scale","kpm_radius",
          "kpm_prec"] if key in kwargs} # the options of the expansion
      if kwargs and not interpolate: # the interpolation takes the rest
        raise TypeError("get_ldos with mode='KPM' got the unexpected "
                "arguments "+", ".join(sorted(kwargs))+"; it takes delta or "
                "npol, kpm_radius, scale and kpm_prec")
      d = ldos_map(h,[e],ks,delta=delta,operator=operator,**kpm)[0][0]
    else: # not recognized
      raise ValueError("unknown mode %r; the LDOS accepts 'green', 'arpack', "
              "'diagonalization' and 'KPM'" % (mode,))
    # write result
    if integration=="grid": d = spatial_dos(h,d) # spatially resolved DOS
    g = h.geometry  # store geometry
    x,y,z,r = g.x,g.y,g.z,g.r # get the coordinates
    if nrep>1:
        go = h.geometry.copy() # copy geometry
        go = go.get_supercell(nrep) # create supercell
        from .geometry import replicate_array
        do = replicate_array(g,d,nrep=nrep) # replicate
#        do = d.tolist()*(nrep**g.dimensionality) # replicate
        xo = go.x
        yo = go.y
        zo = go.z
        ro = go.r
    else: xo,yo,zo,ro,do = x,y,z,r,d
    if interpolate:
        from .interpolation import atomic_interpolation
        xo,yo,do = atomic_interpolation(xo,yo,do,**kwargs)
        if return_rd:
          raise NotImplementedError("return_rd is not implemented together "
                  "with the atomic interpolation")
    if write: 
#        if return_rd: raise # not implemented
        write_ldos(xo,yo,do) # write in file
    xo = np.array(xo)
    yo = np.array(yo)
    zo = np.array(zo)
    ro = np.array(ro)
    do = np.array(do)
    if return_rd: # return positions and DOS
        return (ro,do) # return LDOS
    else: # return x,y and DOS
        return (xo,yo,do) # return LDOS


ldos = get_ldos # for backcompatibility

def multi_ldos(h,projection="TB",**kwargs):
    """Compute the LDOS at different energies, and save everything in a file"""
    from .utilities import rename_kwarg
    # multi_ldos_tb used to spell the operator `op`, which is how the
    # examples call it; the rest of the library spells it `operator`
    kwargs = rename_kwarg(kwargs,"op","operator")
    if projection=="TB": return multi_ldos_tb(h,**kwargs)
    elif projection=="atomic": 
        if kwargs.pop("operator",None) is not None:
            raise NotImplementedError("the atomic projection of the LDOS "
                    "does not accept an operator; use projection='TB'")
        from .ldostk import atomicmultildos
        return atomicmultildos.multi_ldos(h,**kwargs)
    else:
      raise ValueError("unknown projection; multi_ldos accepts 'TB' and "
              "'atomic'")


def multi_ldos_tb(h,energies=np.linspace(-1.0,1.0,100),delta=None,
        nrep=3,nk=100,num_bands=20,
        random=False,operator=None,write=None,mode="diagonalization",
        integration="grid",tolerance=1e-6,**kwargs):
  """Calculate many LDOS, by diagonalizing the Hamiltonian or from a
  Chebyshev expansion

  mode: "diagonalization" broadens the eigenstates with a Lorentzian of
      half width delta (0.01 by default), and "KPM" expands the LDOS of
      every orbital in Chebyshev polynomials, all the energies from one
      recursion, in a time linear in the number of sites, with the
      options of get_ldos in that mode (delta or npol, kpm_radius, scale,
      kpm_prec).
  operator: None, or an operator spec (a name, a matrix, an Operator).
      With mode="diagonalization" each eigenstate is weighted with its
      expectation value <psi|A|psi>, which is the same convention get_ldos
      uses in mode="arpack" (see ldostk.ldoswaves), so the map written for
      a given energy is what get_ldos returns at that energy; with
      mode="KPM" the map is the local matrix element Re(A rho)_ii, as in
      get_ldos with mode="green" or "KPM".
  integration: for mode="diagonalization", one of get_integration_names().
      "triangle" and "qtci" are the simplex integrations of get_ldos on the
      uniform nk^d mesh, with delta=0 (the exact limit) by default: one
      diagonalization of the mesh gives the map at every energy, and the
      DOS written beside the maps comes from the same band energies.
      num_bands does not apply to them.
  """
  write = filewrite.resolve(write,True) # the call, else the global switch
  check_integration(integration)
  if operator is not None: operator = h.get_operator(operator) # resolve name
  if integration!="grid":
    if mode!="diagonalization":
      raise ValueError("integration='"+str(integration)+"' applies to "
              "mode='diagonalization', got mode="+repr(mode))
    if kwargs:
      raise TypeError("multi_ldos with integration='"+str(integration)+"' "
              "got the unexpected arguments "+", ".join(sorted(kwargs)))
    if random:
      raise ValueError("random only applies to integration='grid', got "
              "integration='"+str(integration)+"'")
    if delta is None: delta = 0. # the exact limit
    ds,mesh = simplex_ldos(h,energies,integration=integration,nk=nk,
            delta=delta,operator=operator,tolerance=tolerance)
    outs = list(ds) # one map per energy, one value per site
    from .dostk.simplexdos import dos_from_band_mesh
    es2 = np.linspace(min(energies),max(energies),len(energies)*10)
    ys = dos_from_band_mesh(mesh,es2,delta) # the DOS of the same mesh
  elif mode=="KPM":
    (outs,es2,ys) = _multi_ldos_kpm(h,energies,delta=delta,nk=nk,
            random=random,operator=operator,**kwargs)
  elif mode=="diagonalization":
    if kwargs:
      raise TypeError("multi_ldos with mode='diagonalization' got the "
              "unexpected arguments "+", ".join(sorted(kwargs)))
    if delta is None: delta = 0.01 # Lorentzian broadening
    (outs,es2,ys) = _multi_ldos_diagonalization(h,energies,delta=delta,
            nk=nk,num_bands=num_bands,random=random,operator=operator)
  else:
    raise ValueError("unknown mode %r; multi_ldos accepts 'diagonalization' "
            "and 'KPM'" % (mode,))
  out = (h.geometry.x,h.geometry.y,np.array(energies),np.array(outs))
  if not write: return out
  _write_multi_ldos(h,energies,outs,es2,ys,nrep)
  return out


def _multi_ldos_kpm(h,energies,delta=None,nk=100,random=False,
        operator=None,**kwargs):
  """The maps of multi_ldos_tb from the Chebyshev expansion
  (kpmtk/ldosmap.py), one per energy, and the DOS they add up to, on a
  grid ten times finer"""
  from .kpmtk.ldosmap import ldos_map, kpm_ldos_basis
  unknown = set(kwargs) - {"npol","scale","kpm_radius","kpm_prec"}
  if unknown:
    raise TypeError("multi_ldos with mode='KPM' got the unexpected "
            "arguments "+", ".join(sorted(unknown))+"; it takes delta or "
            "npol, kpm_radius, scale and kpm_prec")
  ks = klist.kmesh(h.dimensionality,nk=nk) # get grid
  if random: ks = [np.random.random(3) for k in ks] # random vectors
  (ds,mus,scale) = ldos_map(h,energies,ks,delta=delta,operator=operator,
          **kwargs)
  outs = [spatial_dos(h,d) for d in ds] # resum if necessary
  es2 = np.linspace(min(energies),max(energies),len(energies)*10)
  # the trace of the maps, the moments averaged over the orbitals
  ys = (mus*h.intra.shape[0])@kpm_ldos_basis(es2,scale,len(mus))
  return outs,es2,ys


def _multi_ldos_diagonalization(h,energies,delta=0.01,nk=100,num_bands=20,
        random=False,operator=None):
  """The maps of multi_ldos_tb from the eigenstates, one per energy, and
  the DOS they add up to, on a grid ten times finer"""
  print("Calculating eigenvectors in LDOS")
  ps = [] # weights
  evals,ws = [],[] # empty list
  ks = klist.kmesh(h.dimensionality,nk=nk) # get grid
  if random:
      ks = [np.random.random(3) for k in ks] # random vectors
      print("RANDOM vectors in LDOS")
  hk = h.get_hk_gen() # get generator
  def get_weight(v,k):
      """Weight of this eigenstate, its expectation value of the operator"""
      if operator is None: return 1.0 # no operator, plain charge LDOS
      return np.real(operator.braket(v,k=k)) # expectation value
  if h.is_sparse: # sparse Hamiltonian
    from .bandstructure import smalleig
    print("SPARSE Matrix")
    for k in ks: # loop
      print("Diagonalizing in LDOS, SPARSE mode")
      e,w = smalleig(hk(k),numw=num_bands,evecs=True,e0=np.mean(energies))
      evals += [ie for ie in e]
      ws += [iw for iw in w]
      ps += [get_weight(iw,k) for iw in w] # weights
  else:
    print("Diagonalizing in LDOS, DENSE mode")
    # the k-points are diagonalized in batches with the same numba-parallel
    # routine dos.py and ldosmap use, instead of one LAPACK call per k-point
    from .htk.eigenvectors import peigh_bloch
    batch_size = 64
    for i0 in range(0,len(ks),batch_size): # loop over batches of kpoints
      kb = ks[i0:i0+batch_size] # kpoints in this batch
      es,vs = peigh_bloch(hk,kb) # diagonalize
      for (ii,k) in enumerate(kb): # loop over kpoints of the batch
        w = vs[ii].transpose() # rows are the eigenvectors
        evals += [ie for ie in es[ii]]
        ws += [iw for iw in w]
        ps += [get_weight(iw,k) for iw in w] # weights
  ds = np.array([(np.conjugate(v)*v).real for v in ws]) # calculate densities
  del ws # remove the wavefunctions
  # The accumulation over eigenstates is bilinear, so the whole double loop
  # over energies and eigenstates is a single matrix product: the weight
  # matrix W[ie,n] = delta/((e-E_n)^2+delta^2)*p_n times the densities.
  # It is done in blocks of energies so that W stays small.
  evals = np.array(evals) # eigenvalues of the whole mesh
  ps = np.array(ps) # weights of the whole mesh
  nkp = len(ks) # number of kpoints in the mesh
  outs = [] # LDOS at each energy
  eblock = max(1,int(1e6//max(len(evals),1))) # energies per block
  for i0 in range(0,len(energies),eblock): # loop over blocks of energies
      eb = np.array(energies[i0:i0+eblock]) # energies of this block
      w = delta/((eb[:,None]-evals[None,:])**2 + delta**2)*ps[None,:]
      # 1/pi is the normalization of the Lorentzian and 1/nkp the average
      # over the Brillouin zone, the same two factors get_ldos applies
      out = (w@ds)/(np.pi*nkp) # LDOS of this block of energies
      outs += [spatial_dos(h,o) for o in out] # resum if necessary
  # the same DOS the maps integrate to
  from .dos import calculate_dos
  es2 = np.linspace(min(energies),max(energies),len(energies)*10)
  # same normalization as the maps above, and as dos.dos_kmesh
  ys = calculate_dos(evals,es2,delta,w=None)/(np.pi*nkp) # compute DOS
  return outs,es2,ys


def _write_multi_ldos(h,energies,outs,es2,ys,nrep):
  """Write the maps of multi_ldos_tb in MULTILDOS, one file per energy,
  their list, DOSMAP.OUT and the DOS"""
  fs.rmdir("MULTILDOS") # remove folder
  fs.mkdir("MULTILDOS") # create folder
  go = h.geometry.copy() # copy geometry
  go = go.supercell(nrep) # create supercell
  fo = open("MULTILDOS/MULTILDOS.TXT","w") # files with the names
  ie = 0
  for e in energies: # loop over energies
    print("MULTILDOS for energy",e)
    oute = outs[ie] ; ie += 1 # get and increase
    name0 = "LDOS_"+str(e)+"_.OUT" # name of the output
    name = "MULTILDOS/" + name0
    from .geometry import replicate_array
    write_ldos(go.x,go.y,replicate_array(h.geometry,oute,nrep=nrep),
                  output_file=name) # write in file
    fo.write(name0+"\n") # name of the file
    fo.flush() # flush
  fo.close() # close file
  fmap = open("DOSMAP.OUT","w")
  for ii in range(len(h.geometry.x)):
      for ie in range(len(energies)):
          fmap.write(str(ii)+"  ")
          fmap.write(str(energies[ie])+"  ")
          fmap.write(str(outs[ie][ii])+"\n")
  fmap.close()
  from .dos import write_dos
  write_dos(es2,ys,output_file="MULTILDOS/DOS.OUT")





def write_ldos(x,y,dos,output_file="LDOS.OUT",z=None):
  """ Write LDOS in a file"""
  fd = open(output_file,"w")   # open file
  fd.write("# x,  y, local density of states\n")
  ii = 0
  for (ix,iy,idos) in zip(x,y,dos): # write everything
    fd.write(str(ix) +"   "+ str(iy) + "   "+ str(idos))
    if z is not None: fd.write("   "+str(z[ii]))
    fd.write("\n")
    ii += 1
  fd.close() # close file




def ldos_finite(h,e=0.0,n=10,nwf=4,delta=0.0001,write=None):
  """Calculate the density of states for a finite system"""
  write = filewrite.resolve(write,True) # the call, else the global switch
  if h.dimensionality!=1: # if it is not one dimensional
    raise ValueError("ldos_finite is only for 1d Hamiltonians")
  intra = csc(h.intra) # convert to sparse
  inter = csc(h.inter) # convert to sparse
  interH = dagger(inter) # hermitian
  m = [[None for i in range(n)] for j in range(n)] # full matrix
  for i in range(n): # add intracell
    m[i][i] = intra
  for i in range(n-1): # add intercell
    m[i][i+1] = inter
    m[i+1][i] = interH
  m = bmat(m) # convert to matrix
  (ene,wfs) = algebra.arpack_eigh(m,k=nwf,which="LM",sigma=0.0) # diagonalize
  wfs = wfs.transpose() # transpose wavefunctions
  dos = (wfs[0].real)*0.0 # calculate dos
  for (ie,f) in zip(ene,wfs): # loop over waves
    c = 1./(1.+((ie-e)/delta)**2) # calculate coefficient
    dos += np.abs(f)*c # add contribution
  odos = spatial_dos(h,dos) # get the spatial distribution
  go = h.geometry.supercell(n) # get the supercell
  if write: write_ldos(go.x,go.y,odos) # write in a file
  return dos # return the dos





def ldos_defect(h,v,e=0.0,delta=0.001,n=1):
  """Calculates the LDOS of a cell with a defect, writting the n
  neighring cells"""
  raise NotImplementedError("ldos_defect is not implemented; use the "
          "Embedding class for a defect in an infinite system")
  from . import green
  # number of repetitions
  rep = 2*n +1
  # calculate pristine green function
  g,selfe = green.supercell_selfenergy(h,e=e,delta=delta,nk=100,nsuper=rep)
  # now calculate defective green function 
  ez = e + 1j*delta # complex energy
  emat = np.array(np.identity(len(g)))*ez  # E +i\delta
  from . import supercell
  pintra = supercell.intra_super2d(h,n=rep) # pristine
  vintra = supercell.intra_super2d(h,n=rep,central=v) # defective
  selfe = emat - pintra - algebra.inv(g) # dyson euqation, get selfenergy
  gv = algebra.inv(emat - vintra -selfe)   # Green function of a vacancy, with selfener
  return



from .ldostk.ldosr import ldosr_generator


