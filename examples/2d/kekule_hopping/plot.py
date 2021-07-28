import numpy as np
import matplotlib.pyplot as plt

m = np.genfromtxt("POSITIONS.OUT").transpose()
mc = np.genfromtxt("R.OUT").transpose()


plt.scatter(m[0],m[1],c="red",s=60)
plt.scatter(mc[0],mc[1],c="blue",s=100)
plt.show()

