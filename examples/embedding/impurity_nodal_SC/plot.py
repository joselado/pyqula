import numpy as np
import matplotlib.pyplot as plt

m1 = np.genfromtxt("DOS_DEFECTIVE.OUT").T
m2 = np.genfromtxt("DOS_PRISTINE.OUT").T

plt.plot(m1[0],m1[1])
plt.plot(m2[0],m2[1])
plt.show()
