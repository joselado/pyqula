from __future__ import print_function
import numpy as np
from scipy import interpolate




def diagram2d(getquantity,x=np.linspace(0.,1.,20),
               y=np.linspace(0.,1.,20),filename="PHASE_DIAGRAM.OUT",
               nite=4):
  """Evaluates a phase diagram, writting the quantities in a file"""
  mz = np.zeros((len(x),len(y))) # 2d array
  mx = np.array([[ix for iy in y] for ix in x]) # 2d array
  my = np.array([[iy for iy in y] for ix in x]) # 2d array
  for i in range(len(x)): # loop over x
    for j in range(len(y)): # loop over x
      q = getquantity(mx[i,j],my[i,j]) # get the value
      mz[i,j] = q # store value

  # now interpolate and call
  mx,my,mz = selected_interpolation(getquantity,mx,my,mz,nite=nite) # interpolate

  fo = open(filename,"w") # open file
  for i in range(len(mx)): # loop over x
    for j in range(len(my)): # loop over x
      fo.write(str(mx[i,j])+"    "+str(my[i,j])+"    "+str(mz[i,j])+"\n") # write in file
  fo.close() # close file




def boundary2d(getquantity,xlim=[0.,1.],
               ylim=[0.,1.],filename="BOUNDARY_PHASE_DIAGRAM.OUT",
               nite=4,tol=0.1,np=50):
  x = np.linspace(xlim[0],xlim[1],np)
  y = np.linspace(ylim[0],ylim[1],4)
  for ix in x: # loop over s parameter
    for iy in y: # loop over s parameter
      raise



def selected_interpolation(fin,mx,my,mz,nite=3):
  """Call in a function in a mesh which is two nite times finer, but only
  in those points that are expected to change, in the others interpolate"""
  if nite<1.01: return mx,my,mz 
  f = interpolate.interp2d(mx[:,0], my[0,:], mz, kind='linear') # interpolation
  x = np.linspace(np.min(mx),np.max(mx),len(mx)*2) # twice as big
  y = np.linspace(np.min(my),np.max(my),len(my)*2) # twice as big
  mx2, my2 = np.meshgrid(x, y) # new grid
  mz2 = f(x,y) # interpolate
  mz2 = mz2.transpose()
  dmz = np.gradient(mz2) # gradient
  dmz = dmz[0]*dmz[0] + dmz[1]*dmz[1] # norm of the derivative
  maxdm = np.max(dmz) # maximum derivative
  for i in range(len(mx2)):
    for j in range(len(my2)):
      if dmz[i,j]>0.001*maxdm: # if derivative is large in this point
        mz2[i,j] = fin(mx2[i,j],my2[i,j]) # re-evaluate the function
  mx2 = mx2.transpose()
  my2 = my2.transpose()
  mz2 = mz2.transpose()
  if nite>1:
    return selected_interpolation(fin,mx2,my2,mz2,nite=nite-1)
  return mx2,my2,mz2 # return function





