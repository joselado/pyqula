from . import specialhopping
from . import specialgeometry
import numpy as np
from . import geometry


def twisted_multilayer_graphene(n=7,ti=0.12,lambi=8.0,lamb=12.0,is_sparse=True,
        g=None,g0=None,has_spin=False,dl=3.0,**kwargs):
    """
    Return the Hamiltonian of twisted bilayer graphene
    """
    if g is None: g = specialgeometry.twisted_bilayer(n,dz=dl,g=g0,**kwargs)
    mgenerator = specialhopping.twisted_matrix(ti=ti,
            lambi=lambi,lamb=lamb,dl=dl)
    h = g.get_hamiltonian(is_sparse=is_sparse,has_spin=has_spin,
            is_multicell=True,mgenerator=mgenerator)
    return h

# some aliases
twisted_bilayer = twisted_multilayer_graphene
tbg = twisted_bilayer
twisted_bilayer_graphene = tbg

def tdbg(n=4,**kwargs): 
    """Twisted double bilayer Hamiltonian"""
    g = specialgeometry.tdbg(n) # TBDG geometry
    return twisted_multilayer_graphene(g=g,**kwargs) # return Hamiltonian


def multilayer_graphene(l=[0],real=False,**kwargs):
  """Return the hamiltonian of multilayer graphene"""
  g = specialgeometry.multilayer_graphene(l=l)
  g.center()
  if real: # real tight binding hopping
      mgenerator = specialhopping.twisted_matrix(**kwargs)
      h = g.get_hamiltonian(has_spin=False,
          mgenerator=mgenerator)
  else:
      h = g.get_hamiltonian(has_spin=False,
          tij=specialhopping.multilayer(**kwargs))
  return h


def flux2d(g,n=1,m=1):
    """Return a Hamiltonian with a certain commensurate flux per unit cell"""
    from . import sculpt
    from . import supercell
    g = supercell.target_angle(g,0.5) # target a orthogonal cell
    g = sculpt.rotate_a2b(g,g.a1,np.array([1.,0.,0.])) # set in the x direction
    g = g.supercell([1,n,1])
    h = g.get_hamiltonian(has_spin=False)
    r = np.array([0.,1.,0.])
    dr = g.a2.dot(r) # distance
    h.add_peierls(m*2.*np.pi/dr)
    return h



def valence_TMDC(g=None,soc=0.0,**kwargs):
    """Return the Hamiltonian for the valence band of a TMDC"""
    if g is None:
        g = geometry.triangular_lattice()
    ft = specialhopping.phase_C3(g,phi=soc,**kwargs)
    h = g.get_hamiltonian(tij=ft,is_multicell=True,has_spin=False)
    h.turn_spinful(enforce_tr=True)
    return h # return the Hamiltonian


def SOC_TMDC(g=None,soc=0.0,**kwargs):
    """Return the Ising SOC for a triangular lattice"""
    if g is None:
        g = geometry.triangular_lattice()
    ft0 = specialhopping.phase_C3(g,phi=.5,**kwargs)
    ft = lambda r1,r2: 1j*(ft0(r1,r2).imag) # only imaginary part
    h = g.get_hamiltonian(tij=ft,is_multicell=True,has_spin=False)
    h.turn_spinful(enforce_tr=True)
    return h # return the Hamiltonian


from .specialhamiltoniantk.tmdc import doped_MoS2
from .specialhamiltoniantk.heavyfermion import H2HFH




def NbSe2(**kwargs):
    return TMDC_MX2(**kwargs,
            tij=[0.02126916,  0.08462639,  0.00504899, -0.00906057, -0.00572983])

def TaS2(**kwargs):
    return TMDC_MX2(**kwargs,
             tij=[ 0.05525963, 0.08784492, -0.00185766, -0.01204535 ,0.00220995])


def TaS2_SOC(**kwargs):
    return TMDC_MX2(**kwargs,soc=-0.5451483039554,
             tij=[0.05901736,0.10243251,-0.00172311,-0.00993678,0.00459015])



def TMDC_MX2(soc=0.0,cdw=0.0,g=None,tij=[1.0],
              drcdw = np.array([0.,0.,0.]), # shift in the CDW profile
              normalize=True):
    """Return the Hamiltonian of NbSe2"""
    if g is None: 
        g = geometry.triangular_lattice()  # triangular lattice
        if cdw!=0.0: g = g.get_supercell(3,store_primal=True)
    tij = np.array(tij)
    if normalize: tij = tij/np.max(np.abs(tij)) # normalize
    h = g.get_hamiltonian(is_multicell=True,has_spin=False,tij=tij)
    ## Now add the SOC if necessary
    if soc!=0.0: # add the SOC
        h.turn_spinful() # turn spinful
        d = g.neighbor_distances()[1]
        d = 1.0
        hsoc = tij[0]*soc*SOC_TMDC(g=h.geometry,d=d) # hamiltonian with SOC
        h = h + hsoc # add the two Hamiltonians
    if cdw!=0.0: # add the CDW
        g0 = geometry.triangular_lattice()  # triangular lattice
        g0 = g0.supercell(3) # create a supercell
        from . import potentials
        f0 = potentials.commensurate_potential(g0,n=6,angle=np.pi/6.,
                k=1./np.sqrt(3)*2.)
        f0 = f0.normalize()*cdw
        f0 = f0.set_average(0.)
        rc = np.mean(h.geometry.get_closest_position([.1,.1,0.],n=3),axis=0)
        rc = rc - np.array(drcdw) # add the shift
        f = lambda r: f0(r-rc)
        h.geometry.write_profile(f)
        h.add_onsite(f)
    h.set_filling(.5)
#    h = h.supercell(4)
#    m = np.array(h.intra.todense()).reshape(h.intra.shape[0]**2)
    return h
    



def triangular_pi_flux(g=None,**kwargs):
    """Return a pi-flux Hamiltonian"""
    if g is None: # no geometry given
        g = geometry.triangular_lattice() # geometry of a triangular lattice
    g = g.supercell((2,1)) # create a supercell with 2 atoms
    ft = specialhopping.phase_C3_matrix(g,phi=.5)
    h = g.get_hamiltonian(mgenerator=ft,is_multicell=True,**kwargs) # return the Hamiltonian
    h.add_peierls(2*np.pi*1./np.sqrt(3.),gauge="Landau") # field
    from . import gauge
#    h = gauge.hamiltonian_gauge_transformation(h,[0.,0.25])
#    return h
    if h.has_time_reversal_symmetry(): pass
    else: 
        print(np.round(h.intra,2))
        for t in h.hopping:
          print(np.round(t.m,2))
        print("Something wrong happened in pi-flux")
        raise
    exit()
    return h



def excitonic_bilayer(gap=0.0,g=None,**kwargs):
    """Return the Hamiltonian for a bilayer system for an excitonic system"""
    from .geometry import get_geometry
    g0 = get_geometry(g) # get the geometry
    h0 = g0.get_hamiltonian(has_spin=False) # spinless
    hv = h0.copy() # valence
    hc = h0.copy() # conduction
    fermiv = hv.get_fermi4filling(1.0,nk=40) # fermi for valence
    fermic = hc.get_fermi4filling(0.0,nk=40) # fermi for conduction
    gv = g0.copy() # valence geometry
    gc = g0.copy() # conduction geometry
    gv = gv + np.array([0.,0.,-1]) # valence
    gc = gc + np.array([0.,0.,1]) # conduction
    g = gv + gc # add geometries
    def tij(r1,r2):
        dr = r1 - r2
        dr2 = dr.dot(dr)
        if 0.99<dr2<1.01: return np.sign(r1[2])
        return 0.0
    h = g.get_hamiltonian(tij=tij,has_spin=False) # no spin
    def fons(r):
        if r[2]>0.0: return -fermiv - gap/2.
        else: return fermiv + gap/2
    h.add_onsite(fons)
    return h



def FeSe_GXY(nem=0.,**kwargs):
    """Return the Hamiltonian of FeSe, a bandstructure
    displaying two pockets"""
    g = geometry.square_lattice() # cubic lattice
#    h = g.get_hamiltonian(tij=[1.,3.],**kwargs) 
    h = g.get_hamiltonian(tij=[1.,-0.4,0.4],**kwargs) 
#    h.add_onsite(-5.)
    h.add_onsite(-2.5)
    v = np.array([1.,0.,0.]) # nematic vector
    v = v/np.sqrt(v.dot(v))
#    h.remove_spin()
    def fnem(dr):
        if dr.dot(dr)>1e-2: dr = dr/np.sqrt(dr.dot(dr)) # unitary vector
        else: return 1.0 # same site
        o = 1. - nem*(dr.dot(v)**2-0.5) # return 
        return o
    if np.abs(nem)>0.:
      h.add_strain(fnem, mode="directional")
    h.turn_dense()
    return h


FeSe = FeSe_GXY


def FeSe_GM(nem=0.,**kwargs):
    """Return the Hamiltonian of FeSe, a bandstructure
    displaying two pockets, one at G and one at M"""
    g = geometry.square_lattice() # cubic lattice
    h = g.get_hamiltonian(tij=[1.,3.],**kwargs) 
    h.add_onsite(-2.5)
    if nem==0.: return h
    v = np.array([1.,0.,0.]) # nematic vector
    v = v/np.sqrt(v.dot(v))
#    h.remove_spin()
    def fnem(dr):
        if dr.dot(dr)>1e-2: dr = dr/np.sqrt(dr.dot(dr)) # unitary vector
        else: return 1.0 # same site
        o = 1. - nem*(dr.dot(v)**2-0.5) # return 
        return o
    if np.abs(nem)>0.:
      h.add_strain(fnem, mode="directional")
    h.turn_dense()
    return h

