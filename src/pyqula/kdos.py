from __future__ import print_function,division
import numpy as np
from . import green
from . import dos
from . import klist
from . import kpm
from . import timing
from . import multicell
from . import kpm
from . import sculpt
from . import parallel
from . import algebra
from . import operators
from .algebra import dagger

def write_kdos(k=0.,es=[],ds=[],new=True):
  """ Write KDOS in a file"""
  if new: f = open("KDOS.OUT","w") # open new file
  else: f = open("KDOS.OUT","a") # continue writting
  for (e,d) in zip(es,ds): # loop over e and dos
    f.write(str(k)+"     ")
    f.write(str(e)+"     ")
    f.write(str(d)+"\n")
  f.close()





def kdos1d_sites(h,sites=[0],scale=10.,nk=100,npol=100,kshift=0.,
                  ewindow=None,info=False):
  """ Calculate kresolved density of states of
  a 1d system for a certain orbitals"""
  if h.dimensionality!=1: # only for 1d
    raise ValueError("kdos1d_sites is only implemented for 1d Hamiltonians")
  ks = np.linspace(0.,1.,nk) # number of kpoints
  h.turn_sparse() # turn the hamiltonian sparse
  hkgen = h.get_hk_gen() # get generator
  if ewindow is None:  xs = np.linspace(-0.9,0.9,nk) # x points
  else:  xs = np.linspace(-ewindow/scale,ewindow/scale,nk) # x points
  write_kdos() # initialize file
  for k in ks: # loop over kpoints
    mus = np.array([0.0j for i in range(2*npol)]) # initialize polynomials
    hk = hkgen(k+kshift) # hamiltonian
    for isite in sites:
      mus += kpm.moments_local_dos(hk/scale,i=isite,n=npol)
    kpm.check_scale(mus,scale,bound=len(sites)) # a sum of unit-vector moments
    ys = kpm.generate_profile(mus,xs) # generate the profile
    write_kdos(k,xs*scale,ys/scale,new=False) # per unit energy, appended
    if info: print("Done",k)

#
#def surface(h,energies=None,klist=None,delta=0.01):
#  """Return bulk and surface DOS"""
#  bout = [] # empty list, bulk
#  sout = [] # empty list, surface
#  for k in klist:
#    for energy in energies:
#      gs,sf = green.green_kchain(h,k=k,energy=energy,delta=delta,only_bulk=False) 
#      bout.append(gs.trace()[0,0].imag) # bulk
#      sout.append(sf.trace()[0,0].imag) # surface
#  bout = np.array(bout).reshape((len(energies),len(klist))) # convert to array
#  sout = np.array(sout).reshape((len(energies),len(klist))) # convert to array
#  return (bout.transpose(),sout.transpose())
#
#
#






def write_surface(h,energies=np.linspace(-.5,.5,300),
        klist=None,delta=None,operator=None,hs=None,**kwargs):
  if delta is None: delta = (np.max(energies)-np.min(energies))/len(energies)
  if h.dimensionality==1:
    write_surface_1d(h,energies=energies,delta=delta,
                         operator=operator)
  elif h.dimensionality==2:
    write_surface_2d(h,energies=energies,klist=klist,delta=delta,
                         operator=operator,hs=hs,**kwargs)
  elif h.dimensionality==3:
    write_surface_3d(h,energies=energies,klist=klist,delta=delta)
  else:
    raise ValueError("write_surface needs a Hamiltonian of dimensionality 1, "
            "2 or 3")



def get_surface_operator(h,operator):
  """Return the matrix a surface/bulk DOS is projected onto.

  This used to be inlined in write_surface_1d/2d as

      if operator is None: op = np.identity(...)
      elif callable(operator): op = callable(op)
      else: op = operator

  whose middle branch referenced an unbound `op` -- and an Operator, which
  is what h.get_operator("sz") returns, is callable, so every named
  operator raised UnboundLocalError before any physics happened."""
  if operator is None: return np.identity(h.intra.shape[0],dtype=np.complex128)
  op = h.get_operator(operator) # resolve names, matrices and Operators alike
  m = op.get_matrix(required=False) # the matrix it acts with
  if m is None:
      raise NotImplementedError("the surface spectral function applies the "
              "operator as a single matrix, the same at every kpoint, so it "
              "cannot take an operator that is defined only by its action on "
              "a wavefunction (a k-dependent one such as \"unfold\", for "
              "instance); use h.get_kdos_bands(mode=\"ED\") instead")
  from scipy.sparse import issparse
  if issparse(m): m = m.todense()
  return np.array(m)


def write_surface_1d(h,energies=None,delta=None,
        operator=None):
  if energies is None: energies = np.linspace(-.5,.5,200)
  if delta is None: delta = (max(energies)-min(energies))/len(energies)
  h = h.get_no_multicell()
  op = get_surface_operator(h,operator) # projection matrix, once
  fo  = open("SURFACE_DOS.OUT","w") # open file
  for energy in energies:
      gs,sf = green.green_renormalization(h.intra,h.inter,
              energy=energy,delta=delta) # surface green function 
      # gs and sf are plain ndarrays, so `*` here was an elementwise
      # product: the trace picked up only sum_i g[i,i]*op[i,i] and any
      # off-diagonal operator (sx, sy, a current) came out identically zero
      db = -algebra.trace(gs@op).imag # bulk
      ds = -algebra.trace(sf@op).imag # surface
      fo.write(str(energy)+"   "+str(ds)+"   "+str(db)+"\n")
      fo.flush()
  fo.close()





def write_surface_2d(h,energies=None,klist=None,delta=0.01,
                         operator=None,hs=None,nk=50):
  bout = [] # empty list, bulk
  sout = [] # empty list, surface
  if klist is None: 
      klist = [[i,0.,0.] for i in np.linspace(-.5,.5,nk)]
  if energies is None: energies = np.linspace(-.5,.5,50)
  op = get_surface_operator(h,operator) # projection matrix, once
  fo  = open("KDOS.OUT","w") # open file
  for k in klist:
    print("Doing k-point",k)
    for energy in energies:
      gs,sf = green.green_kchain(h,k=k,energy=energy,delta=delta,
                       only_bulk=False,hs=hs) # surface green function 
      # see write_surface_1d: `*` was an elementwise product here too
      db = -algebra.trace(gs@op).imag # bulk
      ds = -algebra.trace(sf@op).imag # surface
      fo.write(str(k[0])+"   "+str(energy)+"   "+str(ds)+"   "+str(db)+"\n")
      fo.flush()
  fo.close()


def write_surface_3d(h,energies=None,klist=None,delta=0.01):
  raise NotImplementedError("write_surface_3d is not implemented")
  if h.dimensionality != 3: # only for 3d
    raise ValueError("write_surface_3d is only for 3d Hamiltonians")
  ho = h.copy() # copy Hamiltonian
  ho = ho.turn_multicell() # multicell Hamiltonian
  bout = [] # empty list, bulk
  sout = [] # empty list, surface
  if klist is None:
    raise ValueError("write_surface_3d needs an explicit k-path, pass it as "
            "klist")
  if energies is None: energies = np.linspace(-.5,.5,50)
  fo  = open("KDOS.OUT","w") # open file
  for k in klist:
    for energy in energies:
      gs,sf = green.green_kchain(h,k=k,energy=energy,delta=delta,only_bulk=False) 
      db = -algebra.trace(gs).imag # bulk
      ds = -algebra.trace(sf).imag # surface
      fo.write(str(k)+"   "+str(energy)+"   "+str(ds)+"   "+str(db)+"\n")



def kdos_bands(h,use_kpm=False,kpath=None,scale=10.0,frand=None,
                 P = None,
                 ewindow=4.0,delta=0.01,ntries=10,nk=100,
                 operator=None,energies=np.linspace(-3.0,3.0,200),
                 mode="ED",biorthogonal=False,**kwargs):
    """Calculate the KDOS bands using the KPM.

    Every mode returns the trace -Im Tr[O G(k,E)]/pi, the spectral weight
    summed over the orbitals (weighted by the operator O, if any), so that
    they can be compared: "ED" from the eigenstates, "green" from the
    Green's function, and "KPM" from a Chebyshev expansion, stochastic
    for a matrix O and exact for an O given by a factor, as the unfolding
    one is.

    frand is the KPM random-vector generator (the one kpm.pdos and
    kpm.tdos take): it is what makes the KDOS a projected one, by drawing
    the random vectors from a subspace instead of the whole Hilbert
    space. It used to be accepted here and never forwarded, so the output
    was the unprojected one and was byte-identical with and without it.
    With frand or P the KPM result is an average over the vectors drawn,
    not a trace.

    biorthogonal: which spectral function of a non-Hermitian Hamiltonian,
    where the two definitions in use differ. False (the default, mode="ED"
    only) weighs each eigenstate by its right eigenvector,
    <R_n|O|R_n>/<R_n|R_n>, with a Lorentzian of width delta at Re E_n.
    True is the Green's function one, -Im Tr[O G]/pi with
    G = (w + i delta - H)^-1, which is sum_n <L_n|O|R_n>/<L_n|R_n> over
    poles at the complex E_n, so that a state with Im E_n < 0 is broadened
    by its lifetime (Kozii and Fu, arXiv:1708.05841, Eq. 24); it is what
    mode="green" computes, and mode="ED" gives the same numbers from the
    left and right eigenvectors. A Hermitian Hamiltonian has one spectral
    function and ignores the keyword."""
    if h.non_hermitian: # two spectral functions, and no Chebyshev one
        if mode=="KPM" or use_kpm:
            raise NotImplementedError("the KPM kdos expands in Chebyshev "
                    "polynomials of a Hermitian matrix and cannot take a "
                    "non-Hermitian Hamiltonian; use mode='ED' or "
                    "mode='green'")
        if mode=="green" and not biorthogonal:
            raise ValueError("the Green's function of a non-Hermitian "
                    "Hamiltonian gives the biorthogonal spectral function, "
                    "sum_n <L_n|O|R_n> over its complex poles; pass "
                    "biorthogonal=True to ask for it, or use mode='ED' for "
                    "the one weighted by the right eigenvectors")
    if use_kpm: mode ="KPM" # conventional method
    if frand is not None and mode!="KPM": # nothing to do with it here
        raise ValueError("frand is the KPM random-vector generator, and is "
                "only used by mode='KPM'; pass use_kpm=True or mode='KPM' "
                "to have it honoured (this call asked for mode='"+str(mode)
                +"', which diagonalizes instead of sampling)")
    # normalize the kpath up front: the ED branch below needs the actual
    # kpoints (it indexes get_bands's output by k-index), and get_kpath
    # also expands a list of high-symmetry-point labels into vectors
    kpath = h.geometry.get_kpath(kpath,nk=nk) # generate kpath
    # resolve names ("unfold", "sz", ...) into an Operator once, for every
    # mode: only the ED branch used to do it, by way of get_bands, so a
    # string reached green.GtimesO and operators.Operator unresolved and
    # died inside them. Resolving an Operator again is a no-op.
    operator = h.get_operator(operator)
    if mode=="ED":
        # batched path: diagonalize the whole kpath at once via get_bands
        # (already numba-parallel, see bandstructure.get_bands_nd) instead
        # of dispatching one h.get_dos call per kpoint through an outer
        # process pool -- that wrapped many cheap single-kpoint
        # diagonalizations in pcall, exactly the overhead-dominates-work
        # failure mode the rest of this codebase's pcall->prange migration
        # was built to avoid.
        from .dostk.eigtodos import calculate_dos
        if biorthogonal and h.non_hermitian: # -Im Tr[O G]/pi, pole by pole
            return kdos_biorthogonal(h,kpath,operator,energies,delta,**kwargs)
        bout = h.get_bands(kpath=kpath,operator=operator,write=False,**kwargs)
        kidx = np.real(bout[0]).astype(int) # k-index per row
        es_col = bout[1] # energy per row
        w_col = bout[2] if len(bout)>2 else None # operator weight per row, if any
        out = [] # (energies,dos) pair per kpoint, matching the old pfun contract
        for ik in range(len(kpath)):
            mask = kidx==ik
            w_k = w_col[mask] if w_col is not None else None
            ys = calculate_dos(es_col[mask],energies,delta,w=w_k)
            ys *= 1./np.pi # normalization of the Lorentzian, as in dos.dos_kmesh
            out.append((energies,ys))
    elif mode=="green":
      f = h.get_gk_gen(delta=delta) # Green generator
      def pfun(k): # do it for this k-point
          def gfun(e):
              m = f(k=k,e=e) # Green's function
              m = green.GtimesO(m,operator,k=k)
              return -algebra.trace(m).imag/np.pi # return DOS, same normalization as mode="ED"
          return energies,np.array([gfun(e) for e in energies])
      out = parallel.pcall(pfun,kpath) # compute all
    elif mode=="KPM": # KPM method
      # an operator given as O(k) = U(k) U(k)^dagger, as the unfolding one
      # is, has its weight taken exactly from the columns of U, one
      # Chebyshev expansion each, with no random vectors (see
      # kpm.factored_dos)
      factor = getattr(operator,"factor",None)
      if factor is not None:
          if P is not None or frand is not None:
              raise ValueError("the KPM kdos takes the weight of this "
                      "operator exactly, from its factor, with no random "
                      "vectors, so it cannot also take P or frand, which "
                      "choose the random vectors")
      elif operator is not None:
          # a matrix is the only form kpm.pdos takes; asking for one raises
          # rather than returning None, which used to leave operator=None
          # here and quietly produce an unweighted KDOS
          m = operator.get_matrix(required=False)
          if m is None:
              raise NotImplementedError("the KPM kdos samples the operator "
                      "as a matrix, so it cannot take an operator that is "
                      "defined only by its action on a wavefunction and "
                      "has no factor; use mode=\"ED\" instead")
          operator = m
      h = h.copy()
      h.turn_sparse()
      hkgen = h.get_hk_gen() # get generator
      npol = 3*int(scale/delta) # number of polynomials
      def pfun(k): # do it for this k-point
        hk = hkgen(k) # get Hamiltonian
        if factor is not None: # exact, from the factor
            return kpm.factored_dos(hk,factor(k),scale=scale,npol=npol,
                    ne=npol*4,ewindow=ewindow,x=energies,**kwargs)
        (x,y) = kpm.pdos(hk,scale=scale,npol=npol,ne=npol*4,P=P,
                     operator=operator,frand=frand,
                     ewindow=ewindow,ntries=ntries,x=energies,
                     **kwargs) # compute
        # pdos averages over unit random vectors, which is the trace over
        # the N orbitals divided by N, while mode="ED" and mode="green"
        # return the trace itself; this used to come out N times smaller
        # than them. With P or frand the vectors are drawn from a subspace
        # of the caller's choosing and the result stays an average over it
        if P is None and frand is None: y = y*hk.shape[0]
        return (x,y)
      out = parallel.pcall(pfun,kpath) # compute all
    return write_kdos_bands(kpath,out)


def write_kdos_bands(kpath,out):
    """Write the (energies,kdos) pair of every kpoint to KDOS_BANDS.OUT,
    and return its three columns: position along the path, energy, kdos"""
    ik = 0
    fo = open("KDOS_BANDS.OUT","w") # open file
    for k in kpath: # loop over kpoints
      (x,y) = out[ik] # get this one
      for (ix,iy) in zip(x,y): # loop
        fo.write(str(ik/len(kpath))+"   ")
        fo.write(str(ix)+"   ")
        fo.write(str(iy)+"\n")
      fo.flush()
      ik += 1
    fo.close()
    return np.genfromtxt("KDOS_BANDS.OUT").T


def kdos_biorthogonal(h,kpath,operator,energies,delta,**kwargs):
    """The biorthogonal spectral function of a non-Hermitian Hamiltonian,
    -Im Tr[O G]/pi with G = (w + i delta - H)^-1, from its eigenstates:
    sum_n w_n/(w + i delta - E_n) with w_n = <L_n|O|R_n>/<L_n|R_n> (one
    without an operator) and E_n complex, which is mode="green" pole by
    pole (see kdos_bands)"""
    if kwargs.get("eigmode","complex")!="complex":
        raise ValueError("the biorthogonal spectral function needs the "
                "complex eigenvalues, whose imaginary part broadens each "
                "state, so it takes no eigmode='"+str(kwargs["eigmode"])+"'")
    bout = h.get_bands(kpath=kpath,operator=operator,write=False,
            biorthogonal=True,**kwargs)
    kidx = np.real(bout[0]).astype(int) # k-index per row
    energies = np.array(energies,dtype=float)
    out = [] # (energies,kdos) per kpoint
    for ik in range(len(kpath)):
        mask = kidx==ik
        es = bout[1][mask] # complex eigenvalues
        ws = bout[2][mask] if len(bout)>2 else np.ones(len(es)) # weights
        g = ws[None,:]/(energies[:,None] + 1j*delta - es[None,:]) # poles
        out.append((energies,-np.sum(g,axis=1).imag/np.pi))
    return write_kdos_bands(kpath,out)









def write_surface_kpm(h,ne=400,klist=None,scale=4.,npol=200,w=20,ntries=20):
  """Write the surface DOS using the KPM"""
  if klist is None: klist = np.linspace(-.5,.5,50)
  fo  = open("KDOS.OUT","w") # open file
  for k in klist:
    print("Doing kpoint",k)
    if h.dimensionality==2: 
      (intra,inter) = h.kchain(k) # k hamiltonian
      (es,ds,dsb) = kpm.edge_dos(intra,inter,scale=scale,w=w,npol=npol,
                            ne=ne,bulk=True)
    # if the Hamiltonian is 1d from the beginning
    elif h.dimensionality==1: 
      intra,inter = h.intra,h.inter # 1d hamiltonian
      dd = h.intra.shape[0] # dimension
      inde = np.zeros(dd) # array with zeros
      indb = np.zeros(dd) # array with zeros
      for i in range(dd//10): # one tenth
        inde[i] = 1. # use this one
        indb[4*dd//10 + i] = 1. # use this one
      def gedge(): return (np.random.random(len(inde))-0.5)*inde
      def gbulk(): return (np.random.random(len(indb))-0.5)*(indb)
      # hamiltonian
      h0 = intra + inter*np.exp(1j*np.pi*2.*k) + dagger(inter*np.exp(1j*np.pi*2.*k))
      xs = np.linspace(-0.9,0.9,4*npol) # x points
      es = xs*scale
      # calculate the bulk
      mus = kpm.random_trace(h0/scale,ntries=ntries,n=npol,fun=gbulk)
      kpm.check_scale(mus,scale)
      dsb = kpm.generate_profile(mus,xs)/scale # per unit energy
      # calculate the edge
      mus = kpm.random_trace(h0/scale,ntries=ntries,n=npol,fun=gedge)
      kpm.check_scale(mus,scale)
      ds = kpm.generate_profile(mus,xs)/scale # per unit energy
    else:
      raise ValueError("write_surface_kpm needs a 1d or 2d Hamiltonian")
    for (e,d1,d2) in zip(es,ds,dsb):
      fo.write(str(k)+"   "+str(e)+"   "+str(d1)+"    "+str(d2)+"\n")
  fo.close()




def interface(h1,h2,energies=np.linspace(-1.,1.,100),operator=None,
                    write=True,
                    delta=None,kpath=None,dh1=None,dh2=None,nk=50):
    """Get the surface DOS of an interface"""
    from scipy.sparse import csc_matrix,bmat
    if delta is None:
        delta = 1*(max(energies) - min(energies))/len(energies)
    if kpath is None: 
      if h1.dimensionality==1:
        kpath = [[0.,0.,0.]]
      elif h1.dimensionality==3:
        g2d = h1.geometry.copy() # copy Hamiltonian
        g2d = sculpt.set_xy_plane(g2d)
        kpath = klist.default(g2d,nk=nk)
      elif h1.dimensionality==2:
        kpath = [[k,0.,0.] for k in np.linspace(0.,1.,nk)]
      else:
        raise ValueError("the interface k-path is only defined for "
                "Hamiltonians of dimensionality 2 or 3")
  #  tr = timing.Testimator("KDOS") # generate object
  #  tr.remaining(ik,len(kpath)) # generate object
    ik = 0
    h1 = h1.get_multicell() # multicell Hamiltonian
    h2 = h2.get_multicell() # multicell Hamiltonian
    def computek(ik):
      k = kpath[ik] # get this one
  #    for energy in energies:
  #  (b1,s1,b2,s2,b12) = green.interface(h1,h2,k=k,energy=energy,delta=delta)
  #      out = green.interface(h1,h2,k=k,energy=energy,delta=delta)
      outs = green.interface_multienergy(h1,h2,k=k,energies=energies,
              delta=delta,dh1=dh1,dh2=dh2)
      outstr = ""
      for (energy,out) in zip(energies,outs):
        if operator is None: 
          op = np.identity(h1.intra.shape[0]*2) # normal cell
          ops = np.identity(h1.intra.shape[0]) # supercell 
  #      elif callable(operator): op = callable(op)
        else:
          op = operator # normal cell 
          ops = bmat([[csc_matrix(operator),None],[None,csc_matrix(operator)]])
        # write everything
        outstr += str(ik)+"   "+str(energy)+"   "
        for g in out: # loop
          if g.shape[0]==op.shape[0]: d = -algebra.trace(g@op).imag # bulk
          else: d = -algebra.trace(g@ops).imag # interface
          outstr += str(d)+"   "
        outstr += "\n"
      return outstr
    out = parallel.pcall(computek,range(len(kpath))) # compute all
    if write:
        fo = open("KDOS_INTERFACE.OUT","w")
        fo.write("# k, E, Bulk1, Surf1, Bulk2, Surf2, interface\n")
        for o in out: fo.write(o)
        fo.close()
    return np.genfromtxt("KDOS_INTERFACE.OUT") # return data







def surface_kdos(h1,energies=np.linspace(-1.,1.,100),operator=None,
                    delta=0.01,kpath=None,hs=None,nsuper=None,
                    info = False,
                    write=True,nk=None,**kwargs):
    """Get the surface DOS of an interface"""
    if nk is None: nk = len(energies)
    h1 = h1.get_supercell(nsuper)
    from scipy.sparse import csc_matrix,bmat
    if kpath is None: 
        if h1.dimensionality==3:
          g2d = h1.geometry.copy() # copy Hamiltonian
          g2d = sculpt.set_xy_plane(g2d)
          kpath = klist.default(g2d,nk=nk)
        elif h1.dimensionality==2:
          kpath = [[k,0.,0.] for k in np.linspace(0.,1.,nk)]
        elif h1.dimensionality==1: kpath = [[0.,0.,0.0]] # one dummy point
        else:
          raise ValueError("the surface k-path is only defined for "
                  "Hamiltonians of dimensionality 1, 2 or 3")
    if write: fo = open("KDOS.OUT","w")
    if write: fo.write("# k, E, Surface, Bulk\n")
    if info: tr = timing.Testimator("KDOS") # generate object
    ik = 0
    h1 = h1.get_multicell() # multicell Hamiltonian
    kout = [] # storage
    eout = [] # storage
    dsout = [] # storage
    dbout = [] # storage
    for k in kpath:
      if info: tr.remaining(ik,len(kpath)) # generate object
      ik += 1
      outs = green.surface_multienergy(h1,k=k,energies=energies,
                           delta=delta,hs=hs,**kwargs)
      for (energy,out) in zip(energies,outs):
        # write everything
        kout.append(k[0]) # add to the output
        eout.append(energy) # add to the output
        if write:
          if h1.dimensionality==1: fo.write(str(energy)+"   ")
          else: fo.write(str(ik)+"   "+str(energy)+"   ")
        do = []
        for g in out: # loop
          if operator is None: d = -algebra.trace(g).imag # only the trace 
          elif type(operator)==operators.Operator:
              d = -(operators.Operator(g)*operator).trace().imag
          elif callable(operator): d = operator(g,k=k) # call the operator
          else:  d = -algebra.trace(g@operator).imag # assume it is a matrix
          if write: fo.write(str(d)+"   ") # write in a file
          do.append(d) # store
        dsout.append(do[0]) # add to the output
        dbout.append(do[1]) # add to the output
        if write:
          fo.write("\n") # next line
          fo.flush() # flush
    if write: fo.close()
    return np.array(kout),np.array(eout),np.array(dsout),np.array(dbout)



surface = surface_kdos
