# library to write in file the different states
import scipy.linalg as lg
import numpy as np
from .ldos import spatial_dos as spatial_wave
from .ldos import write_ldos as write_wave

def states0d(h,ewindow=[-.5,.5],signed=False,prefix=""):
  """Write in files the different states"""
  if not h.dimensionality==0: raise
  (evals,evecs) = lg.eigh(h.intra)
  evecs = evecs.transpose() # transpose list
  for i in range(len(evals)):
    if ewindow[0]<evals[i]<ewindow[1]: # if in the interval
      print("Printing state",evals[i])
      den = np.abs(evecs[i])**2
      den = spatial_wave(h,den) # resum if other degrees of freedom
      name=prefix+"WAVE_"+str(i)+"_energy_"+str(evals[i])+"_.OUT"
      write_wave(h.geometry.x,h.geometry.y,den,output_file=name)
      if signed: # if you want to write signed waves
        v = evecs[i]
        ii = np.sum(v.imag)
        if ii<0.00001:
          den = spatial_wave(h,v.real) # resum if other degrees of freedom   
          name="WAVE_SIGNED_"+str(i)+"_energy_"+str(evals[i])+".OUT"
          write_wave(h.geometry.x,h.geometry.y,den,output_file=name)
        else:
          print("Warning, non real function")




def states2d(h,ewindow=[-.5,.5],signed=False,prefix="",k=[0.,0.,0.],nrep=3):
  """Write in files the different states in 2d"""
  if not h.dimensionality==2: raise
  hk = h.get_hk_gen() # get generator
  m = hk(k) # stre this matrix
  (evals,evecs) = lg.eigh(m)
  evecs = evecs.transpose() # transpose list
  densum = h.geometry.x*0.0 # storage
  for i in range(len(evals)):
    if ewindow[0]<evals[i]<ewindow[1]: # if in the interval
      print("Printing state",evals[i])
      den = np.abs(evecs[i])**2
      den = spatial_wave(h,den) # resum if other degrees of freedom
      densum += den # store density
      name=prefix+"WAVE_"+str(i)+"_energy_"+str(evals[i])+"_.OUT"
      go = h.geometry.copy() # copy geometry
      go = go.supercell(nrep) # create supercell
      write_wave(go.x,go.y,den.tolist()*(nrep**2),output_file=name,z=go.z) 
      if signed: # if you want to write signed waves
        v = evecs[i]
        ii = np.sum(np.abs(v.imag))
        if ii<0.00001:
          den = spatial_wave(h,v.real) # resum if other degrees of freedom   
          name="WAVE_SIGNED_"+str(i)+"_energy_"+str(evals[i])+".OUT"
          go = h.geometry.copy() # copy geometry
          go = go.supercell(nrep) # create supercell
          write_wave(go.x,go.y,den.tolist()*(nrep**2),output_file=name,z=go.z) 
        else:
          print("Warning, non real function")
  go = h.geometry.copy() # copy geometry
  go = go.supercell(nrep) # create supercell
  write_wave(go.x,go.y,densum.tolist()*(nrep**2),output_file="WAVESUM.OUT",z=go.z) 






def multi_states(h,ewindow=[-0.5,0.5],signed=False,
          ks=None):
  """Calculate the bands and states in an energy window,
  and write them in a folder"""
  import os
  if ks is None:
    from . import klist
    ks = klist.default(h.geometry) # generate default klist
  os.system("rm -rf MULTISTATES_KPATH") # delete folder
  os.system("mkdir MULTISTATES_KPATH") # create folder
  os.chdir("MULTISTATES_KPATH") # go to the folder
  ik = 0
  for k in ks: # loop over kpoints
    print("MULTISTATES for",k)
    states2d(h,ewindow=ewindow,signed=signed,k=k,prefix="K_"+str(ik)+"_") 
    ik += 1
  os.chdir("..") # go to the folder




