# Add the root path of the pyqula library
import os ; import sys 
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/../../../src")

import numpy as np
from pyqula import geometry
import matplotlib.pyplot as plt

g0 = geometry.chain()
n  = 30
g = g0.get_supercell(n,store_primal=True)
h = g.get_hamiltonian(has_spin=False,non_hermitian=True,tij=[-1])
omega = 1./n
ons = lambda r: 0.3*(np.cos(np.pi*2*omega*r[0])+1.)
h.add_onsite(ons)
h.add_onsite(2.)

ks,es = h.get_bands(kpath=[[0.]])
numbands = 10 # number of bands to plot
es = es[0:10].real # get the first numbands bands
ie = 0
fig = plt.figure(figsize=[12,numbands*2])
nrep = 4 # number of copies

v = h.extract("onsite",nrep=nrep) # extract the potential
v = geometry.replicate_array(h.geometry,v,nrep=nrep) # replicat the potential
v = v - np.min(v)
v = v/np.max(v)
for e in es:
    plt.subplot(len(es),1,ie+1) ; ie += 1
    (xi,yi,di) = h.get_ldos(e=e,delta=0.01,nrep=nrep)
    di = di/np.max(di) # normalize
    # make a smooth interpolation
    from scipy.interpolate import interp1d
    xnew = np.linspace(xi.min(), xi.max(), len(xi)*10)
    dnew = interp1d(xi, di,kind="cubic")(xnew)
    label = "Band #"+str(ie)+", E = "+str(np.round(e,2))
    plt.plot(xnew,dnew,c="blue",label=label) # plot the interpolated one
    # now plot the potential
    plt.plot(xi,v,linestyle="dashed",c="red",label="V(x)")
    plt.ylim([0.,1.1]) ; plt.xticks([]) ; plt.yticks([])
    plt.xlabel("x") ; plt.ylabel("$|\\Psi^2|$")
    plt.xlim([min(xi),max(xi)])
    plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
