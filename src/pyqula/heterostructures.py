from __future__ import print_function
import numpy as np
import pylab as py
from copy import deepcopy as dc
import scipy.linalg as lg
from scipy.sparse import bmat,coo_matrix,csc_matrix
from . import green
from .algebra import dagger

dag = dagger

class Heterostructure():
    """Class for a HTstructure"""
    def copy(self):
      """Copy the HTstructure"""
      from copy import deepcopy
      return deepcopy(self)
    def __init__(self,h=None):  # initialization using a hamiltonian
      self.file_right_green = "green_right.dat"  # in/out file for right green
      self.file_left_green = "green_left.dat"   # in/out file for left green
      self.file_heff = "heff.dat"   # out for the effective hamiltonian
      self.scale_rc = 1.0 # scale coupling to the right lead
      self.scale_lc = 1.0 # scale coupling to the left lead
      self.is_sparse = False
      self.dimensionality = 1 # default is one dimensional
      self.transparency = 1.0 # reference transparency (for kappa)
      self.delta = 1e-4
      self.extra_delta_central = 0. # additional delta in the central region
      self.extra_delta_right = 0. # additional delta in the right region
      self.extra_delta_left = 0. # additional delta in the left region
      self.interpolated_selfenergy = False
      self.use_minimal_selfenergy = False # use a model selfenergy
      self.minimal_selfenergy_gamma = 1.0 # the Gamma of the leads
      self.block_diagonal = False
      if h is not None:
        self.heff = None  # effective hamiltonian
        self.right_intra = h.intra  # intraterm in the right lead
        self.right_inter = h.inter  # interterm in the right lead (to the right)
        self.right_green = None  # right green function
        self.left_intra = h.intra  # intraterm in the left lead
        self.left_inter = dagger(h.inter)  # interterm in the left lead (to the left)
        self.left_green = None  # left green function
        self.central_intra = h.intra  # intraterm in the center
        self.right_coupling = h.inter # coupling from the center to the right lead
        self.left_coupling = dagger(h.inter) # coupling from the center to the left lead
        # geometry of the central part
        gc = dc(h.geometry)
        self.central_geometry = gc # geometry of the central part
        # additional degrees of freedom
        self.has_spin = h.has_spin   # spin degree of freedom
        self.has_eh = h.has_eh   # electron hole pairs
    def surface_dos(self,**kwargs):
        from .transporttk import sdos
        return sdos.surface_dos(self,**kwargs)
    def get_kdos(self,**kwargs):
        from .transporttk import kdos
        return kdos.kdos(self,**kwargs)
    def get_ldos(self,**kwargs):
        from .transporttk import ldos
        return ldos.ldos(self,**kwargs)
    def get_dos(self,**kwargs):
        from .transporttk import dos
        return dos.get_dos(self,**kwargs)
    def get_coupled_central_dos(self,**kwargs):
        return device_dos(self,mode="central",**kwargs)
    def get_coupled_left_dos(self,**kwargs):
        return device_dos(self,mode="left",**kwargs)
    def get_coupled_right_dos(self,**kwargs):
        return device_dos(self,mode="right",**kwargs)
    def landauer(self,energy=[0.],delta=0.0001,do_leads=True,left_channel=None,
                  right_channel=None):
      """ Return the Landauer transmission"""
      if self.has_eh: raise # invalid if there is electorn-hole
      return landauer(self,energy=energy,delta=delta,do_leads=do_leads,
                      left_channel=left_channel,right_channel=right_channel)
    def write_green(self):
        """Writes the green functions in a file"""
        from .green import write_matrix
        write_matrix(self.file_right_green,self.right_green)
        write_matrix(self.file_left_green,self.left_green)
    def read_green(self):
      """Reads the green functions from a file"""
      from .green import read_matrix
      self.right_green = read_matrix(self.file_right_green)
      self.left_green = read_matrix(self.file_left_green)
    def write_heff(self):
      """ Writes effective hamiltonian in a file"""
      from .green import write_sparse
      write_sparse(self.file_heff,self.heff)
    def eigenvalues(self,numeig = 10,effective=False,full=False):
      """ Calculates eigenvalues of the central part """
      return eigenvalues(self,numeig=numeig,effective=effective,
                          full=full)
    def replace_center(self,ht_replacement):
        """ Replaces the central part by the second argument"""
        self.central_intra = ht_replacement.central_intra  # make the change
    def calculate_surface_green(self,energy=0.0,delta=0.0001,error=0.0000001):
        """Calculate the surface Green function"""
        delta = self.delta
        # Right
        intra = self.right_intra
        inter = self.right_inter
        gbulk,g = green.green_renormalization(intra,inter,error=error,
                                               energy=energy,delta=delta)
        self.right_green = g # save green function
        # Left
        intra = self.left_intra
        inter = self.left_inter
        gbulk,g = green.green_renormalization(intra,inter,error=error,
                                               energy=energy,delta=delta)
        self.left_green = g # save green function
        self.energy_green = energy # energy of the Green function
    def copy_surface_green(self,ht):
        """Copy the surface Green fucntions"""
        self.energy_green = ht.energy_green
        self.left_green = ht.left_green
        self.right_green = ht.right_green
    def get_selfenergy(self,energy,**kwargs):
       """Return selfenergy of iesim lead"""
       from .transporttk.selfenergy import get_selfenergy
       return get_selfenergy(self,energy,**kwargs)
    def get_reflection_normal_lead(self,s):
       from .transporttk.builder import get_reflection_normal_lead
       return get_reflection_normal_lead(self,s)
    def get_central_gmatrix(self,**kwargs):
       """Return the inverse central Green's function"""
       from .transporttk.smatrix import get_central_gmatrix
       return get_central_gmatrix(self,**kwargs) 
    def set_coupling(self,c): 
       """Coupling for kappa functionality"""
       self.scale_lc = np.sqrt(c)
       self.scale_rc = np.sqrt(c)
    def setup_selfenergy_interpolation(self,es=np.linspace(-4.0,4.0,100),
             delta=0.0001,pristine=False):
        """Create the functions that interpolate the selfenergy"""
        from .interpolation import intermatrix
        self.interpolated_selfenergy = False # set as False
        fsl = lambda e: self.get_selfenergy(e,delta=delta,lead=0,pristine=pristine)
        fsr = lambda e: self.get_selfenergy(e,delta=delta,lead=1,pristine=pristine)
        fun_sr = intermatrix(fsr,xs=es) # get the function
        fun_sl = intermatrix(fsl,xs=es) # get the function
        self.selfgen = [fun_sl,fun_sr] # store functions
        self.interpolated_selfenergy = True # set as true
    def didv(self,**kwargs):
        from .transporttk.didv import generic_didv
        return generic_didv(self,**kwargs)
    def kdidv(self,**kwargs):
        from .transporttk.didv import didv_kmap
        return didv_kmap(self,**kwargs)
    def block2full(self,sparse=False):
        """Put in full form"""
        return block2full(self,sparse=sparse)
    def get_kappa(self,**kwargs):
        from .transporttk.kappa import get_kappa_ratio
        return get_kappa_ratio(self,**kwargs)
  




def create_leads_and_central(h_right,h_left,h_central,num_central=1,
      interpolation="None",block_diagonal=True):
  """ Creates an HTjunction by giving the hamiltonian
     of the leads and the center, with a certain number of cells
     in the center  """
  # check the hamiltonians
  if num_central==1: block_diagonal = False
#  h_right.check()
#  h_left.check()
#  h_central.check()
  ht = HTstructure(h_central) # create HTstructure
  ht.has_eh = h_right.has_eh # if it has electron-hole
  ht.get_eh_sector = h_right.get_eh_sector # if it has electron-hole
  # assign matrices of the leads
  ht.right_intra = h_right.intra.copy()  
  ht.right_inter = h_right.inter.copy()  
  ht.left_intra = h_left.intra.copy()  
  ht.left_inter = dagger(h_left.inter).copy() 
  # create matrix of the central part and couplings to the leads
  from scipy.sparse import csc_matrix,bmat
  z = csc_matrix(h_central.intra*0.0j) # zero matrix
  hc = [[None for i in range(num_central)] for j in range(num_central)]
  tcr = [[None] for i in range(num_central)]
  tcl = [[None] for i in range(num_central)]  
  ##############################################################
  # create the centrla hamiltonian according to different schemes
  ##############################################################
  if not block_diagonal: 
    ec = csc_matrix(h_central.intra) 
    er = csc_matrix(h_right.intra) 
    el = csc_matrix(h_left.intra) 
    tr = csc_matrix(h_right.inter) 
    tl = csc_matrix(h_right.inter) 
    tc = csc_matrix(h_central.inter) 
  if block_diagonal: 
    ec = h_central.intra.copy()  
    er = h_right.intra.copy()  
    el = h_left.intra.copy()  
    tr = h_right.inter.copy()  
    tl = h_right.inter.copy() 
    tc = h_central.inter.copy() 
  # central part is pure central input hamilotnian
  if interpolation=="None": # without central interpolation
    for i in range(num_central):  # intra term of the central blocks
      hc[i][i] = ec
    for i in range(num_central-1): # interterm of the central blocks
      hc[i][i+1] = tc
      hc[i+1][i] = tc.H

  # central part is a step of the right and left hamiltonians
  elif interpolation=="step": # with step interpolation
    # intraterm
    for i in range(num_central):  # intra term of the central blocks
      if i<num_central/2:
          hc[i][i] = el
      elif i>num_central/2:
          hc[i][i] = er
      elif i==num_central/2:
          hc[i][i] = ec
      else:
        raise
    # interterm
    for i in range(num_central-1): # interterm of the central blocks
      if i<num_central/2:
        hc[i][i+1] = tl
        hc[i+1][i] = tl.H
      elif i>num_central/2:
        hc[i][i+1] = tr
        hc[i+1][i] = tr.H
      elif i==num_central/2:
        hc[i][i+1] = tc
        hc[i+1][i] = tc.H
      else:
        raise


  # central part is a linear interpolation of right and left
  elif interpolation=="linear": 
    for i in range(num_central):  # intra term of the central blocks
      r = float(i+1)/float(num_central+1)   # 
      hc[i][i] = er*r + el*(1.-r)  # mean interpolation
    for i in range(num_central-1): # interterm of the central blocks
      r = float(i+1)/float(num_central+1)   # 
      hc[i][i+1] = tr*r + tl*(1.-r)
      hc[i+1][i] = (tr*r + tl*(1.-r)).H


  else:
    raise

  for i in range(num_central):  # intra term of the central blocks
    tcr[i][0] = csc_matrix(z) 
    tcl[i][0] = csc_matrix(z) 

  tcr[-1][0] = csc_matrix(h_right.inter) # hopping to the right lead
  tcl[0][0] = csc_matrix(dagger(h_left.inter)) # hopping to the left lead
  # create dense matrices
  if not block_diagonal:
    hc = bmat(hc).todense()
    tcr = bmat(tcr).todense() 
    tcl = bmat(tcl).todense()
    ht.block_diagonal = False
  # do not create dense matrices
  if block_diagonal:
    hc = hc   # this is a list !!!!!
    tcr = h_right.inter.copy()    # this is a matrix
    tcl = h_left.inter.H.copy()   # this is a matrix
    ht.block_diagonal = True
  # assign to the HTstructure
  ht.right_coupling = tcr
  ht.left_coupling = tcl
  ht.central_intra = hc
  # and modify the geometry
  ht.central_geometry.supercell(num_central) 
  return ht

HTstructure = Heterostructure

from .transporttk import dos
device_dos = dos.device_dos


def plot_central_dos(ht,energies=[0.0],num_rep=100,
                      mixing=0.7,eps=0.0001,green_guess=None,max_error=0.0001):
   """ Plots the density of states by using a 
    green function approach"""
   # get the dos
   dos = central_dos(ht,energies=energies,
             num_rep=num_rep,mixing=mixing,
             eps=eps,green_guess=green_guess,max_error=max_error)
   # plot the figure
   fig = py.figure() # create figure
   fig.set_facecolor("white") # face in white
   sp = fig.add_subplot(111) # create subplot
   sp.set_xlabel("Energy",size=20)
   sp.set_ylabel("Central DOS",size=20)
   sp.plot(energies,dos) # create the plot
   return fig




from .transporttk.landauer import landauer





def block2full(ht,sparse=False):
  """Convert a HTstructure with block diagonal Hamiltonian
  into the full form"""
  if not ht.block_diagonal: return ht # stop
  ho = ht.copy()
  ho.block_diagonal = False # set in false from now on
  nb = len(ht.central_intra) # number of blocks
  if nb>=2:
      lc = [csc_matrix(ht.central_intra[i][i].shape) for i in range(nb)]
      rc = [csc_matrix(ht.central_intra[i][i].shape) for i in range(nb)]
      lc[0] = csc_matrix(ht.left_coupling)
      rc[nb-1] = csc_matrix(ht.right_coupling)
  else: 
      print("Not implemented")
      raise # no central part
  # convert the central to sparse form
  central = [[None for i in range(nb)] for j in range(nb)]
  for i in range(nb):
    for j in range(nb):
      if ht.central_intra[i][j] is None: continue
      else:
        central[i][j] = csc_matrix(ht.central_intra[i][j])
  from scipy.sparse import vstack
  if sparse:
    ho.left_coupling = vstack(lc)
    ho.right_coupling = vstack(rc)
    ho.central_intra = bmat(ht.central_intra) # as sparse matrix
  else:
    ho.left_coupling = vstack(lc).todense()
    ho.right_coupling = vstack(rc).todense()
    ho.central_intra = bmat(central).todense() # as dense matrix
  return ho








def create_leads_and_central_list(h_right,h_left,list_h_central,
        right_coupling = None,left_coupling = None,
        scale_right_coupling = 1.0,
        scale_left_coupling = 1.0,
        coupling=lambda i,j: 1.0):
    """ Creates an HTjunction by giving the hamiltonian
       of the leads and the list of the center """
    # check the hamiltonians
  #  h_right.check()
  #  h_left.check()
    # convert to the classical way
    h_right = h_right.get_no_multicell()
    h_left = h_left.get_no_multicell()
    list_h_central = [h.get_no_multicell() for h in list_h_central]
    if len(list_h_central)==1: # only one central part
      return create_leads_and_central(h_right,h_left,list_h_central[0])
    ht = HTstructure(h_right) # create HTstructure
    ht.Hr = h_right.copy() # store
    ht.Hl = h_left.copy() # store
    # assign matrices of the leads
    ht.right_intra = h_right.intra.copy() 
    ht.right_inter = h_right.inter.copy()  
    ht.left_intra = h_left.intra.copy()  
    ht.left_inter = dagger(h_left.inter).copy() 
    # elecron hole stuff
    ht.has_eh = h_right.has_eh # if it has electron-hole
    ht.get_eh_sector = h_right.get_eh_sector # if it has electron-hole
    # create matrix of the central part and couplings to the leads
    from scipy.sparse import csc_matrix,bmat
    z = csc_matrix(h_right.intra*0.0j) # zero matrix
    num_central = len(list_h_central) # length of the central hamiltonian
    # create list of lists for the central part
    hc = [[None for i in range(num_central)] for j in range(num_central)]
    # create elements of the central hamiltonian
    for i in range(num_central):  # intra term of the central blocks
      hc[i][i] = list_h_central[i].intra.copy()  # intra term of the iesim
    for i in range(num_central-1):  # intra term of the central blocks
      tr = list_h_central[i].inter + list_h_central[i+1].inter
      tr = tr/2.*coupling(i,i+1)   # mean value of the hoppings
      hc[i][i+1] = tr # inter term of the iesim
      hc[i+1][i] = dagger(tr) # inter term of the iesim
  
    # hoppings to the leads
    tcr = h_right.inter.copy()    # this is a matrix
    tcl = dagger(h_left.inter).copy()   # this is a matrix
    ht.block_diagonal = True
    # hoppings from the center to the leads
    if right_coupling is not None:
      ht.right_coupling = right_coupling*scale_right_coupling
    else:
      ht.right_coupling = h_right.inter*scale_right_coupling
    if left_coupling is not None:
      ht.left_coupling = left_coupling*scale_left_coupling
    else:
      ht.left_coupling = dagger(h_left.inter)*scale_left_coupling
  
    # assign central hamiltonian
    ht.central_intra = hc
#    # and modify the geometry of the central part
#    ht.central_geometry.supercell(num_central) 
#    # put if it si sparse
#    ht.is_sparse = list_h_central[0].is_sparse
    return ht


def eigenvalues(HT,numeig=10,effective=False,gf=None,full=False):
  """ Gets the lowest eigenvalues of the central part of the hamiltonian"""
  if not HT.block_diagonal:
    print(""" HTunction in eigenvalues must be block diagonal""")
    raise
  # if effective hamiltonian, just calculate the eigenvalues
  if effective: # effective hamiltonian
    print("Calculating eigenvalues of effective hamiltonian...")
    if HT.heff is None:  #if hasn't been calculated so far
      effective_central_hamiltonian(HT,write=False)
    heff = HT.heff # store the list of list with central EFF ham
    import scipy.sparse.linalg as lg
    eig,eigvec = lg.eigs(heff,k=numeig,which="LM",sigma=0.0)
    return eig

  from scipy.sparse import csc_matrix,bmat
  import scipy.sparse.linalg as lg
  if not effective: # do not use the effective hamiltonian, only the central
    intra = HT.central_intra # store the list of list with central ham
  numb = len(intra) # number of central blocks
  intrasp = [[None for i in range(numb)] for j in range(numb)]
  # assign onsite and couplings in sparse form
  if not HT.is_sparse:
    for i in range(numb):
      intrasp[i][i] = csc_matrix(intra[i][i])
    for i in range(numb-1):
      intrasp[i][i+1] = csc_matrix(intra[i][i+1])
      intrasp[i+1][i] = csc_matrix(intra[i+1][i])
  if HT.is_sparse:
    for i in range(numb):
      intrasp[i][i] = intra[i][i]
    for i in range(numb-1):
      intrasp[i][i+1] = intra[i][i+1]
      intrasp[i+1][i] = intra[i+1][i]
  intrasp = bmat(intrasp) # create sparse matrix
  if effective:
    evals,evecs = lg.eigs(intrasp,k=numeig,which="LM",sigma=0.0)
  if not effective:
    if full:  # full diagonalization
      from scipy.linalg import eigvalsh
      evals = eigvalsh(intrasp.todense())
    else:
      evals,evecs = lg.eigsh(intrasp,k=numeig,which="LM",sigma=0.0)
  return evals




def effective_central_hamiltonian(HT,energy=0.0,delta=0.0001,write=False):
   """ Plots the local density of states in the central part"""
   from .green import green_renormalization
   from .green import dyson
   from .hamiltonians import is_number
   # perform dyson calculation
   intra = HT.right_intra
   inter = HT.right_inter
#   gr = dyson(intra,inter,is_sparse=HT.is_sparse)
   ggg,gr = green_renormalization(intra,inter,energy=energy,delta=delta)
   HT.right_green = gr # save green function
   # left green function
   intra = HT.left_intra
   inter = HT.left_inter
#   gl = dyson(intra,inter,is_sparse=HT.is_sparse)
   ggg,gl = green_renormalization(intra,inter,energy=energy,delta=delta)
   HT.left_green = gl # save green function
   # save green functions
#   HT.write_green()
   # left selfenergy
   inter = HT.left_coupling
   selfl = inter@gl@dagger(inter) # left selfenergy
   # right selfenergy
   inter = HT.right_coupling
   selfr = inter@gr@dagger(inter) # right selfenergy
   # central green function
   intra = HT.central_intra
# dyson equation for the center     
   # full matrix
   if not HT.block_diagonal:
     heff = intra + selfl + selfr
     HT.heff = heff
     if save_heff:
       print("Saving effective hamiltonian in ",HT.file_heff)
       if write: # save hamiltonian if desired
         HT.write_heff() 
   # reduced matrix
   if HT.block_diagonal: 
     from copy import deepcopy
     heff = deepcopy(intra)
     heff[0][0] = intra[0][0] + selfl
     heff[-1][-1] = intra[-1][-1] + selfr
    # save the green function
     from scipy.sparse import bmat
#     HT.heff = bmat(heff)
     if write: # save hamiltonian
       print("Saving effective hamiltonian in ",HT.file_heff)
       HT.write_heff() 
   return heff




def get_tmatrix(ht,energy=0.0,delta=0.0001):
  """Calculate the S-matrix of an HTstructure"""
  if ht.block_diagonal: raise  # not implemented
  smatrix = get_smatrix(ht,energy=energy)
  return smatrix[0][1]






def get_surface_green(HT,energy=0.0,delta=1e-5):
   """Calculate left and right greeen functions"""
   from .green import green_renormalization
   # right lead
   intra = HT.right_intra
   inter = HT.right_inter
   ggg,gr = green_renormalization(intra,inter,energy=energy,delta=delta)
#   HT.right_green = gr # save green function
   # left lead
   intra = HT.left_intra
   inter = HT.left_inter
   ggg,gl = green_renormalization(intra,inter,energy=energy,delta=delta)
   return (gl,gr)




def get_bulk_green(HT,energy=0.0,delta=0.0001):
   """Calculate left and right bulk green functions"""
   from .green import green_renormalization
   # right lead
   intra = HT.right_intra
   inter = HT.right_inter
   gr,gr2 = green_renormalization(intra,inter,energy=energy,delta=delta)
#   HT.right_green = gr # save green function
   # left lead
   intra = HT.left_intra
   inter = HT.left_inter
   gl,gl2 = green_renormalization(intra,inter,energy=energy,delta=delta)
   return (gl,gr)









def get_central_selfenergies(HT,energy=0.0,delta=0.0001):
   """Calculate left and right selfenergies, coupled to central part"""
   (gl,gr) = get_surface_green(HT,energy=energy,delta=delta)
   inter = HT.left_coupling
   selfl = inter*gl*inter.H # left selfenergy
   # right selfenergy
   inter = HT.right_coupling
   selfr = inter*gr*inter.H # right selfenergy
   return (selfl,selfr) # return selfenergies


def get_surface_selfenergies(HT,energy=0.0,delta=0.0001,pristine=False):
   """Calculate left and right selfenergies"""
   (gl,gr) = get_surface_green(HT,energy=energy,delta=delta)
   if pristine:
     inter = HT.left_inter
   else:
     inter = HT.left_coupling
   selfl = inter@gl@np.conjugate(inter).T # left selfenergy
   # right selfenergy
   if pristine:
     inter = HT.right_inter
   else:
     inter = HT.right_coupling
   selfr = inter@gr@np.conjugate(inter).T # right selfenergy
   return (selfl,selfr) # return selfenergies




from .transporttk.builder import build

from .transporttk.smatrix import get_smatrix

from .transporttk.localprobe import LocalProbe
