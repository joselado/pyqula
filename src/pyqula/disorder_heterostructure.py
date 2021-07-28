# different types of disorder for a heterostructure

import numpy as np
import random

def anderson(hetero,w=0.2,write=False):
  """Adds anderson disorder, asumes spinpol calculation"""
  hc = hetero.central_intra # this is a nxn tridiagonal list of matrices
  nb = len(hc)  # number of blocks
  norb = len(hc[0][0]) # number of orbitals with spin degree of freedom
  if hetero.has_spin:
    norb = norb/2 # without spin degree
  if write: fa = open("ANDERSON_DISORDER.OUT","w")
  for ib in range(nb): # loop over blocks
    for iorb in range(norb): # loop over orbitals
      wi = (random.random()-0.5)*w  # stregth of the disorder
      if hetero.has_spin:
        hc[ib][ib][2*iorb,2*iorb] += wi  # up channel
        hc[ib][ib][2*iorb+1,2*iorb+1] += wi  # same for down channel
      else:
        hc[ib][ib][iorb,iorb] += wi  # spinless channel
      if write: # write in file
        fa.write(str(ib)+"   "+str(iorb)+"   "+str(wi)+"\n") # save the quantity
  if write: fa.close()  # close file


def magnetic_z(hetero,w=0.2):
  """Adds off-plane magnetic disorder, asumes spinpol calculation"""
  hc = hetero.central_intra # this is a nxn tridiagonal list of matrices
  nb = len(hc)  # number of blocks
  norb = len(hc[0][0]) # number of orbitals with spin degree of freedom
  norb = norb/2 # without spin degree
  fa = open("MZ_DISORDER.OUT","w")
  for ib in range(nb): # loop over blocks
    for iorb in range(norb): # loop over orbitals
      wi = (random.random()-0.5)*w  # stregth of the disorder
      hc[ib][ib][2*iorb,2*iorb] += wi
      hc[ib][ib][2*iorb+1,2*iorb+1] += -wi
      fa.write(str(ib)+"   "+str(iorb)+"   "+str(wi)+"\n") # save the quantity
  fa.close()



def magnetic_x(hetero,w=0.2):
  """Adds in-plane magnetic disorder, asumes spinpol calculation"""
  hc = hetero.central_intra # this is a nxn tridiagonal list of matrices
  nb = len(hc)  # number of blocks
  norb = len(hc[0][0]) # number of orbitals with spin degree of freedom
  norb = norb/2 # without spin degree
  fa = open("MX_DISORDER.OUT","w")
  for ib in range(nb): # loop over blocks
    for iorb in range(norb): # loop over orbitals
      wi = (random.random()-0.5)*w  # stregth of the disorder
      hc[ib][ib][2*iorb,2*iorb+1] += wi
      hc[ib][ib][2*iorb+1,2*iorb] += wi
      fa.write(str(ib)+"   "+str(iorb)+"   "+str(wi)+"\n") # save the quantity
  fa.close()


def magnetic_y(hetero,w=0.2):
  """Adds in-plane magnetic disorder, asumes spinpol calculation"""
  hc = hetero.central_intra # this is a nxn tridiagonal list of matrices
  nb = len(hc)  # number of blocks
  norb = len(hc[0][0]) # number of orbitals with spin degree of freedom
  norb = norb/2 # without spin degree
  fa = open("MY_DISORDER.OUT","w")
  for ib in range(nb): # loop over blocks
    for iorb in range(norb): # loop over orbitals
      wi = (random.random()-0.5)*w  # stregth of the disorder
      hc[ib][ib][2*iorb,2*iorb+1] += 1j*wi
      hc[ib][ib][2*iorb+1,2*iorb] += -1j*wi
      fa.write(str(ib)+"   "+str(iorb)+"   "+str(wi)+"\n") # save the quantity
  fa.close()






def random_vacancies(hetero,w=300.0,p_vac=0.0):
  """Adds a random vacancy, asumes spinpol calculation"""
  hc = hetero.central_intra # this is a nxn tridiagonal list of matrices
  nb = len(hc)  # number of blocks
  norb = len(hc[0][0]) # number of orbitals with spin degree of freedom
  fa = open("VACANCY_DISORDER.OUT","w")
  norb = norb/2 # without spin degree
  ind_vac_tot = []  # indexes of the vacancies
  for ib in range(nb): # loop over blocks
    ind_vac = []  # indexes of the vacancies
    for iorb in range(norb): # loop over orbitals
      r = random.random() # random number
      if r<p_vac: # if not chosen yet, add and exit while
        ind_vac.append(iorb)
    ind_vac_tot.append(ind_vac)
    for iorb in ind_vac:
      hc[ib][ib][2*iorb,2*iorb] += w
      hc[ib][ib][2*iorb+1,2*iorb+1] += w
      fa.write(str(ib)+"   "+str(iorb)+"   "+str(1.0)+"\n") # save the quantity
  fa.close()



