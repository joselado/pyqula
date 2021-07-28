from __future__ import print_function
import numpy as np

from . import extract

def swave(h,name="SWAVE.OUT",nrep=3):
  """Write the swave pairing of a Hamiltonian"""
  if not h.has_eh: raise
  d = h.extract("swave") # get the pairing
  g = h.geometry # get the geometry
  g.write_profile(np.abs(d),name="AMPLITUDE_"+name,
          normal_order=True,nrep=nrep)
  g.write_profile(np.angle(d)/(2*np.pi),name="PHASE_"+name,
          normal_order=True,nrep=nrep)
  return



  
  
def anomalous_hopping(h,name="ANOMALOUS_HOPPING.OUT",nrep=3,
        cutoff=1e-6):
    """Write in a file the s-wave hoppings"""
    from. import superconductivity
    if not h.has_eh: raise
    h = h.supercell(nrep)
    m = superconductivity.get_eh_sector_odd_even(h.intra,i=0,j=1)
    (ii,jj,ts) = extract.hopping_spinful(m)
    f = open(name,"w") # write file
    for (i,j,t) in zip(ii,jj,np.abs(ts)):
    #  if skip(h.geometry.r[i],h.geometry.r[j]): continue
      if t<cutoff: continue
      if i==j: continue
      f.write(str(h.geometry.r[i][0])+"  ")
      f.write(str(h.geometry.r[i][1])+"  ")
      f.write(str(h.geometry.r[j][0])+"  ")
      f.write(str(h.geometry.r[j][1])+"  ")
      f.write(str(np.round(t,4))+"\n")
    f.close()



def mz(h,name="MZ.OUT"):
  if h.has_eh: raise








def hopping(h,name="HOPPING.OUT",nrep=3,skip = lambda r1,r2: False,
        spin_imbalance=False,cutoff=1e-2):
  """Write the magnitude of the hopping in a file"""
  if h.has_eh: 
      h = h.copy()
      h.remove_nambu()
  h = h.supercell(nrep)
  if h.has_spin: 
      if spin_imbalance:
          (ii,jj,ts) = extract.hopping_spinful_difference(h.intra,
                  skip_same_site=True)
      else:
          h0 = h.copy()
          h0.remove_spin()
          (ii,jj,ts) = extract.hopping_spinless(h0.intra)
  #        (ii,jj,ts) = extract.hopping_spinful(h.intra)
  else: (ii,jj,ts) = extract.hopping_spinless(h.intra)
  f = open(name,"w") # write file
  for (i,j,t) in zip(ii,jj,np.abs(ts)):
    if skip(h.geometry.r[i],h.geometry.r[j]): continue
    if t<cutoff: continue
    if i==j: continue
    f.write(str(h.geometry.r[i][0])+"  ")
    f.write(str(h.geometry.r[i][1])+"  ")
    f.write(str(h.geometry.r[j][0])+"  ")
    f.write(str(h.geometry.r[j][1])+"  ")
    f.write(str(np.round(t,4))+"\n")
  f.close()



def mz(h,name="MZ.OUT"):
  if h.has_eh: raise
  if h.has_spin: ms = extract.mz(h.intra)
  else: raise
  np.savetxt(name,np.matrix([range(len(ms)),ms]).T)



def magnetization(h):
  """Write all the magnetizations"""
  if h.has_eh: raise
  if h.has_spin: 
    mx = extract.mx(h.intra)
    my = extract.my(h.intra)
    mz = extract.mz(h.intra)
  else: raise
  np.savetxt("MAGNETIZATION_X.OUT",np.matrix([h.geometry.x,h.geometry.y,mx]).T)
  np.savetxt("MAGNETIZATION_Y.OUT",np.matrix([h.geometry.x,h.geometry.y,my]).T)
  np.savetxt("MAGNETIZATION_Z.OUT",np.matrix([h.geometry.x,h.geometry.y,mz]).T)




