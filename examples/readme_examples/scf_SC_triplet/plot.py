import numpy as np
import matplotlib.pyplot as plt

m = np.genfromtxt("KDOS_BANDS.OUT").T
k = np.unique(m[0])
e = np.unique(m[1])
d = m[2].reshape((len(k),len(e))).T

plt.contourf(k,e,d,levels=40,cmap="inferno")
plt.show()
