# library to calculate transpot in multiterminal devices

import numpy as np
from . import neighbor

class Device():
  """ Device with leads and scattering part"""
  def __init__(self):
    self.leads = [] # empty list of leads
  def biterminal(self,right_g=None,left_g=None,central_g=None,fun=None,
                    disorder=0.0):
    """Create the matrices for a biterminal device, based on geometries"""
    if fun is None: # no function provided
      def fun(r1,r2):
        dr = r1-r2
        if .7<dr.dot(dr)<1.3: return True
        else: return False
    leadr = Lead() # right lead
    leadl = Lead() # left lead
    Rr = right_g.r # positions
    Lr = left_g.r # positions
    Cr = central_g.r # positions
    leadr.intra = neighbor.parametric_hopping(Rr,Rr,fun) # intra term
    leadl.intra = neighbor.parametric_hopping(Lr,Lr,fun) # intra term
    intra = neighbor.parametric_hopping(Cr,Cr,fun) # intra term
    for i in range(intra.shape[0]): # add disorder
      intra[i,i] += disorder*(np.random.random()-.5)
    self.intra = intra # store
    leadr.coupling = neighbor.parametric_hopping(Rr,Cr,fun) # coupling
    leadl.coupling = neighbor.parametric_hopping(Lr,Cr,fun) # coupling
    # now coupling within the lead
    Rr_dis = [r-right_g.a1 for r in Rr] # displace
    Lr_dis = [r-left_g.a1 for r in Lr] # displace
    leadr.inter = neighbor.parametric_hopping(Rr,Rr_dis,fun) # intra term
    leadl.inter = neighbor.parametric_hopping(Lr,Lr_dis,fun) # intra term
    # store positions
    leadr.r = Rr
    leadl.r = Lr
    self.r = Cr
    # store leads
    self.leads = [leadr,leadl] # store the leads
  def write(self):
    """Write positions of the atoms"""
    np.savetxt("CENTRAL.XYZ",self.r) # write central
    for i in range(len(self.leads)): 
      np.savetxt("LEAD_"+str(i)+".XYZ",self.leads[i].r) # write lead
    
  def transmission(self,energy=0.0):
    """Calculate the transmission"""
    return landauer(self,energy)
  def write_current(self,energy=0.0):
    """Calculate the transmission"""
    den = central_density(self,energy=energy)
    from .ldos import write_ldos
    write_ldos(self.r[:,0],self.r[:,1],den,output_file="CURRENT.OUT")



class Lead():
  """ Class for a lead"""
  intra = None  # intraterm
  inter = None  # interterm
  coupling = None  # coupling to the center
  def get_green(self,energy,error=0.00001,delta=0.00001):
    """ Get surface green function"""
    from . import green 
    grb,gr = green.green_renormalization(self.intra,self.inter,error=error,
                                          energy=energy,delta=delta)
    return gr
  def get_selfenergy(self,energy,error=0.0001,delta=0.0001):
    """ Get selfenergy"""
    gr = self.get_green(energy,error=error,delta=delta) # get greenfunction
    t = self.coupling # coupling
    selfenergy = t.H * gr * t 
    return selfenergy


def landauer(d,energy,ij=[(0,1)],error=0.000001,delta=0.00001):
  """ Calculate landauer tranmission between leads i,j """
  Ms = landauer_matrix(d,energy,ij=ij,error=error,delta=delta) # matrix
  Ts = [m.trace()[0,0] for m in Ms]
  return Ts


def landauer_matrix(d,energy,ij=[(0,1)],error=0.000001,delta=0.00001):
  """Return the Landauer matrices"""
  ss = [l.get_selfenergy(energy,error=error,delta=delta) for l in d.leads]
  # sum of selfenergies
  ssum = ss[0]*0.  
  for s in ss:  ssum += s # sum the selfenergies
  # identity matrix
  iden = np.identity(d.intra.shape[0])
  # calculate central green function
  gc = ((energy + delta*1j)*iden -  d.intra - ssum). I
  # calculate transmission
  ts = [] # empty list
  for (i,j) in ij: # loop over pairs
    # calculate spectral functions of the leads
    gammai = 1j*(ss[i]-ss[i].H) # gamma function
    gammaj = 1j*(ss[j]-ss[j].H) # gamma function
    # calculate transmission
    t = (gammai*gc*gammaj.H*gc.H).real 
    ts.append(t) # add to the list
  return ts # return list




def central_density(d,energy,ij=[(0,1)],error=0.000001,delta=0.00001):
  """Return the Landauer matrices"""
  ss = [l.get_selfenergy(energy,error=error,delta=delta) for l in d.leads]
  # sum of selfenergies
  ssum = ss[0]*0.  
  for s in ss:  ssum += s # sum the selfenergies
  # identity matrix
  iden = np.identity(d.intra.shape[0])
  # calculate central green function
  gc = ((energy + delta*1j)*iden -  d.intra - ssum). I
  # calculate transmission
  den = np.array([gc[i,i].imag for i in range(gc.shape[0])])
  return den # return list

