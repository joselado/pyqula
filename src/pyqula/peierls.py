import numpy as np
from scipy.sparse import coo_matrix,csc_matrix,issparse


def add_phase(m1,r1,r2,phasefun,has_spin=False):
  if m1.shape[0] != len(r1):
    raise ValueError("the matrix and the list of positions must have the same "
            "size, one row per site")
  m = coo_matrix(m1) # convert to sparse matrix
  row,col = m.row,m.col
  data = m.data +0j
  for k in range(len(m.data)): # loop over non vanishing elements
    i = m.row[k]
    j = m.col[k]
    if has_spin: i,j = i//2,j//2 # if spinful
    # peierls phase
    p = phasefun(r1[i],r2[j]) # function yielding the phase
    data[k] *= p # add phase
  out = csc_matrix((data,(row,col)),shape=m1.shape) # convert to csc
  if not issparse(m1): out = out.todense() # dense matrix
  return out






def add_peierls(h,mag_field=0.0,new=False):
  """ Adds Peierls phase to the Hamiltonian"""
  if h.has_eh:
    raise NotImplementedError("the Peierls phase is not implemented for "
            "Hamiltonians with the electron-hole (Nambu) degree of freedom")
  x = h.geometry.x    # x coordinate 
  y = h.geometry.y    # x coordinate 
  a1 = h.geometry.a1    # distance to neighboring cell
  celldis = np.sqrt(a1.dot(a1))
  from numpy import array
  norb = h.intra.shape[0]  # number of orbitals
  if h.is_multicell:
    raise NotImplementedError("the Peierls phase is not implemented for "
            "multicell Hamiltonians")
  if new:
    print("New method to calculate peierls")
    g = h.geometry # geometry
    has_spin = h.has_spin # has spin degree of freedom
    def phasefun(ri,rj): 
      if callable(mag_field): return mag_field(ri[0],ri[1],rj[0],rj[1])
      else:
        raise TypeError("this Peierls branch needs mag_field to be a callable "
                "of the two positions")
    h.intra = add_phase(h.intra,g.r,g.r,phasefun,has_spin) 
    if h.dimensionality==2:
      h.tx = add_phase(h.tx,g.r,g.replicas(d=[1.,0.,0.]),phasefun,has_spin) 
      h.ty = add_phase(h.ty,g.r,g.replicas(d=[0.,1.,0.]),phasefun,has_spin) 
      h.txy = add_phase(h.txy,g.r,g.replicas(d=[1.,1.,0.]),phasefun,has_spin) 
      h.txmy = add_phase(h.txmy,g.r,g.replicas(d=[1.,-1.,0.]),phasefun,has_spin) 
      return
  else: # old method


    if h.is_sparse: # sparse hamiltonian
      if True: # zero dimensional
        m = coo_matrix(h.intra) # convert to sparse matrix
        row,col = m.row,m.col
        data = m.data +0j
        for k in range(len(m.data)): # loop over non vanishing elements
          i = m.row[k]
          j = m.col[k]
          if h.has_spin: i,j = i/2,j/2 # raise if spinful
          p = peierls(x[i],y[i],x[j],y[j],mag_field) # peierls phase
          data[k] *= p # add phase
        h.intra = csc_matrix((data,(row,col)),shape=(norb,norb)) # convert to csc
      if h.dimensionality==1: # one dimensional
        # check that celldis is right
        if np.abs(celldis - h.geometry.a1[0])>0.001:
          raise ValueError("the Peierls gauge assumes the 1d lattice vector "
                  "points along x, and this one does not")
        def phaseize(inter,numn=1):
          m = coo_matrix(inter) # convert to sparse matrix
          row,col = m.row,m.col
          data = m.data +0j
          for k in range(len(m.data)): # loop over non vanishing elements
            i = m.row[k]
            j = m.col[k]
            if h.has_spin: i,j = i//2,j//2 # raise if spinful
            # peierls phase
            p = peierls(x[i],y[i],x[j]+numn*celldis,y[j],mag_field) 
            data[k] *= p # add phase
          return csc_matrix((data,(row,col)),shape=(norb,norb)) # convert to csc
        # for normal hamiltonians
        if not h.is_multicell: h.inter = phaseize(h.inter,numn=1) 
        # for multicell hamiltonians
        if h.is_multicell: 
          hopping = [] # empty list
          for i in range(len(h.hopping)):
            h.hopping[i].m = phaseize(h.hopping[i].m,numn=h.hopping[i].dir[0]) 
    else: # not sparse
      def gaugeize(m,d=0.0):
        """Add gauge phase to a matrix"""
        for i in range(len(x)):
          for j in range(len(x)):
            p = peierls(x[i],y[i],x[j]+d,y[j],mag_field) # peierls phase
            if h.has_spin:
              m[2*i,2*j] *= p
              m[2*i,2*j+1] *= p
              m[2*i+1,2*j] *= p
              m[2*i+1,2*j+1] *= p
            else:
              m[i,j] *= p
      gaugeize(h.intra,d=0.0)  # gaugeize intraterm
      if h.dimensionality==0: pass # if zero dimensional
      elif h.dimensionality==1: # if one dimensional
        if h.is_multicell:
          raise NotImplementedError("the Peierls phase is not implemented for "
                  "multicell Hamiltonians")
        gaugeize(h.inter,d=celldis) # gaugeize interterm
      elif h.dimensionality==2: # if bigger dimensional
        print("WARNING, is your gauge periodic?")
        gaugeize(h.tx,d=h.geometry.a1[0]) # gaugeize interterm
      else:
        raise NotImplementedError("the Peierls phase is only implemented up "
                "to 2d")




def peierls(x1,y1,x2,y2,mag_field):
  """ Returns the complex phase with magnetic field """
  if is_number(mag_field): b = mag_field 
  elif callable(mag_field): 
      b = mag_field(np.array([x1,y1,0.0]),np.array([x2,y2,0.0]))
  else:
    raise TypeError("mag_field must be a number or a callable of the two "
            "positions")
  phase = b*(x1-x2)*(y1+y2)/2.0
  return np.exp(1j*phase*2*np.pi)


from .algebra import isnumber as is_number



def add_inplane_bfield(h,**kwargs):   add_bfield(h,mode="inplane",**kwargs)
def add_offplane_bfield(h,**kwargs):   add_bfield(h,mode="offplane",**kwargs)

def add_peierls(h,mag_field,**kwargs):
    add_offplane_bfield(h,b=mag_field*2,**kwargs)





def add_bfield(h,b=0.0,phi=0.0,mode="inplane",gauge="Landau"):
    """Add an in-plane magnetic field"""
    if h.dimensionality>2:
      raise NotImplementedError("an in-plane magnetic field is only "
              "implemented for Hamiltonians up to 2d")
    # number of orbitals per site: this used to assume 2 for a spinful
    # Hamiltonian, which is wrong for a Nambu one (4 per site, or 2 for a
    # spinless Nambu) and indexed off the end of the position list
    nsites = len(h.geometry.r) # number of sites
    ndim = h.intra.shape[0] # dimension of the Hamiltonian
    if ndim%nsites!=0:
        raise ValueError("the Hamiltonian dimension "+str(ndim)+" is not a "
          +"multiple of the number of sites "+str(nsites))
    nper = ndim//nsites # orbitals per site
    gi = lambda i: i//nper # index of the site of this orbital
    if h.has_eh: # Nambu Hamiltonian
        # the hole block is -H*(-A), so it must pick up the conjugate
        # Peierls phase; the anomalous block has no single well defined
        # one (a Cooper pair carries charge 2e, and a field there means
        # vortices), so refuse rather than guess
        if not h.get_anomalous_hamiltonian().is_zero():
            raise NotImplementedError("an orbital magnetic field cannot be "
              +"added to a Hamiltonian that already carries a pairing "
              +"amplitude: the anomalous term has no single Peierls phase "
              +"(a Cooper pair has charge 2e, and an orbital field in a "
              +"superconductor means vortices). Add the field to the "
              +"normal-state Hamiltonian first, then turn_nambu()/"
              +"add_swave(), so that the hole block is built from the "
              +"field-carrying electron block")
        ehsign = lambda i: 1.0 if (i%nper)<nper//2 else -1.0 # electron/hole
    else:
        ehsign = lambda i: 1.0 # no electron-hole degree of freedom
    if not h.is_multicell: 
        h.turn_multicell() # turn to multicell form
    cphi = np.cos(phi*np.pi)
    sphi = np.sin(phi*np.pi)
    g = h.geometry # get geometry
    r = g.r # get positions
    def get_phase(r,dr):
        if mode=="inplane": return 2*r[2]*(dr[0]*sphi - dr[1]*cphi)
        elif mode=="offplane": 
            if gauge=="Landau": return r[1]*dr[0]/2.
            elif gauge=="symmetric": return r[1]*dr[0]/4.- r[0]*dr[1]/4.
            else: raise ValueError("unknown gauge '"+str(gauge)
                    +"', expected 'Landau' or 'symmetric'")
        else: raise ValueError("unknown mode '"+str(mode)
                +"', expected 'inplane' or 'offplane'")
    def add_phase(m,r1,r2): # add the phase
        mo = coo_matrix(m) # convert to coo matrix
        data = mo.data +0.0j
        k = 0
        for (i,j,d) in zip(mo.row,mo.col,mo.data): # loop
            r = (r1[gi(i)] + r2[gi(j)])/2.
            dr = r1[gi(i)] - r2[gi(j)]
            p = get_phase(r,dr)
            # the hole block carries the opposite charge, hence the
            # conjugate phase; ehsign is 1 everywhere without Nambu
            data[k] *= np.exp(1j*ehsign(i)*b*p*2.*np.pi)
            k += 1
        out = csc_matrix((data,(mo.row,mo.col)),shape=mo.shape) 
        return out.todense() # return matrix
    # now add the phase to the Hamiltonian
    h.intra = add_phase(h.intra,r,r) # add the phase
    for i in range(len(h.hopping)):
        t = h.hopping[i]
        h.hopping[i].m = add_phase(t.m,r,g.replicas(d=t.dir))
    return h




