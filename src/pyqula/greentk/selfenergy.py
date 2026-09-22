
import numpy as np
from .. import algebra
from .. import integration
from .rg import green_renormalization
from .kchain import green_kchain



def bloch_selfenergy(h,nk=100,energy = 0.0, delta = 1e-2,
                         mode="adaptive", # algorithm for integration
                         gtype="bulk", # bulk or surface
                         error=1e-3,numba=None):
  """ Calculates the selfenergy of a cell defect,
      input is a hamiltonian class"""
  if mode=="adaptative": mode = "adaptive"
  from ..htk.kchain import detect_longest_hopping
  if detect_longest_hopping(h)==1:
      def gr(h):
        """ Calculates G by renormalization"""
        # get_no_multicell() always redoes a full conversion + a
        # multicell<->non-multicell round-trip consistency check (two
        # deepcopies) even when h is already non-multicell -- only
        # turn_no_multicell's *first* step short-circuits for that case,
        # not the method's own consistency check. Skipping the call
        # entirely when nothing needs converting is exact (h already is
        # its own non-multicell form) and removes a call measured to cost
        # as much as the entire self-energy solve it was embedded in, in
        # hot loops like the LocalProbe Keldysh sideband sweep that call
        # this once per energy with the same, unchanging h.
        if h.is_multicell: h = h.get_no_multicell()
        ons,hop = h.intra,h.inter
        gf,sf = green_renormalization(ons,hop,energy=energy,nite=None,
                                error=error,info=False,delta=delta,
                                numba=numba)
        return gf,sf
  elif detect_longest_hopping(h)==2:
      from ..htk.kchain import kchain_NNN # extract up to NNN
      def gr(h):
          (ons,t1,t2) = kchain_NNN(h) # return the three matrices
          from ..greentk.dyson import dysonNNN
          gf,sf = dysonNNN(ons,t1,t2,energy=energy,delta=delta,
                  error=error)
          return gf,sf
  else:
      from ..htk.kchain import kchain_LR # extract all
      def gr(h):
          hops = kchain_LR(h) # return all matrices
          from ..greentk.dyson import dysonLR
          gf,sf = dysonLR(hops,energy=energy,delta=delta,
                  error=error)
          return gf,sf
#  else: # too long range hoppings for RG, use full integration
#      mode = "full_adaptive" 
#      print("Changed to full adaptive mode in selfenergy")
  # get_dense() already returns an independent copy (it deepcopies
  # internally before modifying), so an extra h.copy() here would just
  # deepcopy the whole Hamiltonian a second time for nothing -- this
  # function is called once per energy in hot loops like the LocalProbe
  # Keldysh sideband sweep (keldyshtk/current.py), where that redundant
  # deepcopy dominated the profile.
  h = h.get_dense() # dense Hamiltonian
  # `h.get_hk_gen()` goes through `get_multicell()`, a full deepcopy of the
  # Hamiltonian (geometry included), and only the two modes that integrate
  # the Bloch Hamiltonian over the Brillouin zone explicitly ("full" and
  # "full_adaptive") ever call the generator -- "adaptive" and
  # "renormalization" go through the decimation instead and never look at
  # it. Building it eagerly therefore paid that deepcopy once per energy
  # for nothing on exactly the path the LocalProbe Keldysh sideband sweep
  # (keldyshtk/current.py) spends its time on, tens of thousands of times
  # per dI/dV point. Built on first use instead, which keeps it out of
  # every mode that does not ask for it without any mode having to know.
  _hk_gen = [] # holds the generator once something actually calls it
  def hk_gen(k):
      if len(_hk_gen)==0: _hk_gen.append(h.get_hk_gen())
      return _hk_gen[0](k)
  # sanity check for surface mode
  if gtype=="surface": mode = "adaptive" # only the adaptive mode
  #######################################
  d = h.dimensionality # dimensionality of the system
  g = h.intra *0.0j # initialize green function
  e = np.array(np.identity(g.shape[0]))*(energy + delta*1j) # complex energy
  if mode=="full":  # full integration in the BZ
    if d==1: # one dimensional
      ks = [[k,0.,0.] for k in np.linspace(0.,1.,nk,endpoint=False)]
    elif d==2: # two dimensional
      ks = []
      kk = np.linspace(0.,1.,nk,endpoint=False)  # interval 0,1
      for ikx in kk:
        for iky in kk:
          ks.append([ikx,iky,0.])
      ks = np.array(ks)  # all the kpoints
    else: # raise error
      raise NotImplementedError("the full-integration selfenergy is only "
              "implemented for 1d and 2d Hamiltonians")
    for k in ks:  # loop in BZ
      g += algebra.inv(e - hk_gen(k))  # add green function  
    g = g/len(ks)  # normalize
  #####################################################
  #####################################################
  elif mode=="renormalization":
    if d==1: # full renormalization
      g,s = gr(h)  # perform renormalization
    elif d==2: # two dimensional, loop over k's
      ks = [[k,0.,0.] for k in np.linspace(0.,1.,nk,endpoint=False)]
#      from ..multicell import rotate90
#      h90 = rotate90(h) # rotated Hamiltonian
      for k in ks:  # loop over k in y direction
 # add contribution to green function
        g += green_kchain(h,k=k,energy=energy,delta=delta,
                error=error,only_bulk=True)
#        g += green_kchain(h90,k=k,energy=energy,delta=delta,error=error)
      g = g/len(ks)
    else:
      raise NotImplementedError("the renormalization selfenergy is only "
              "implemented for 1d and 2d Hamiltonians")
  #####################################################
  #####################################################
  elif mode=="adaptive":
    if d==1: # full renormalization
      g,s = gr(h)  # perform renormalization
      if gtype=="surface": g = s.copy() # take the surface one
      elif gtype=="bulk": pass # do nothing
      else:
        raise ValueError("unknown gtype; the accepted ones are 'surface' and "
                "'bulk'")
    elif d==2: # two dimensional, loop over k's
      ks = [[k,0.,0.] for k in np.linspace(0.,1.,nk,endpoint=False)]
      if gtype=="surface": ig = 1 # take the surface one
      elif gtype=="bulk": ig = 0 # take the bulk one
      else:
        raise ValueError("unknown gtype; the accepted ones are 'surface' and "
                "'bulk'")
      def fint(k):
        """ Function to integrate """
        return green_kchain(h,k=[k,0.,0.],energy=energy,
                delta=delta,error=error,only_bulk=False)[ig]
      # eps is error, might work....
      g = integration.integrate_matrix(fint,xlim=[0.,1.],eps=error)
        # chain in the y direction
    else:
      raise NotImplementedError("the adaptive selfenergy is only implemented "
              "for 1d and 2d Hamiltonians")
  elif mode=="full_adaptive":
    fint = lambda k: algebra.inv(e - hk_gen(k))  # green's function
    if d==1: # adaptive 1D
        # the 1D integrator hands the integrand a scalar, the Bloch
        # generator wants a k-vector
        g = integration.integrate_matrix(lambda k: fint([k]),xlim=[0.,1.],
              eps=error)
    elif d==2: # adaptive 2D
        g = integration.integrate_matrix_2D(fint,xlim=[0.,1.],ylim=[0.,1.],
              eps=error)
    else:
      raise NotImplementedError("the fully adaptive selfenergy is only "
              "implemented for 1d and 2d Hamiltonians")
  # now calculate selfenergy
  selfenergy = e - h.intra - algebra.inv(g)
  return g,selfenergy




def bloch_selfenergy_batch(h,energies,delta=1e-2,mode="adaptive",
                           gtype="bulk",error=1e-3,**kwargs):
    """`bloch_selfenergy` at a whole set of energies at once, returned as
    two (len(energies),n,n) arrays -- the Green's function and the
    selfenergy, in the same order the scalar function returns them.

    Only one shape actually batches: a 1d Hamiltonian with first-neighbour
    hoppings solved by decimation (`mode="adaptive"`), where every energy
    runs the same Sancho-Rubio iteration on the same (intra,inter) pair and
    only the complex energy differs. Those go through
    `greentk.rg.green_renormalization_jit_batch`, one numba prange-parallel
    call for the whole set, instead of one Python-level call per energy --
    which is what a LocalProbe's sample selfenergy costs in the Floquet
    sideband sweep of `keldyshtk.current.dc_current`, tens of thousands of
    times per dI/dV point.

    Everything else falls back to a plain loop over `bloch_selfenergy`, so
    this is always safe to call: the same numbers either way, and a
    speedup only where there is one to be had."""
    from ..htk.kchain import detect_longest_hopping
    energies = np.asarray(energies,dtype=np.float64)
    if not (h.dimensionality==1 and mode=="adaptive"
                and detect_longest_hopping(h)==1):
        out = [bloch_selfenergy(h,energy=e,delta=delta,mode=mode,
                                gtype=gtype,error=error,**kwargs)
               for e in energies]
        return (np.array([o[0] for o in out]),np.array([o[1] for o in out]))
    if gtype!="surface" and gtype!="bulk":
        raise ValueError("unknown gtype; the accepted ones are 'surface' and "
                "'bulk'")
    h = h.get_dense() # dense Hamiltonian, as bloch_selfenergy does
    if h.is_multicell: h = h.get_no_multicell()
    from .rg import green_renormalization_jit_batch
    g_bulk,g_surf = green_renormalization_jit_batch(h.intra,h.inter,energies,
                                                     delta=delta,error=error)
    g = g_surf if gtype=="surface" else g_bulk
    intra = algebra.todense(h.intra)
    iden = np.identity(intra.shape[0],dtype=np.complex128)
    # one complex energy per entry of the batch, broadcast over the block
    e = (energies+1j*delta)[:,None,None]*iden[None,:,:]
    selfenergy = e - intra[None,:,:] - np.linalg.inv(g)
    return g,selfenergy
