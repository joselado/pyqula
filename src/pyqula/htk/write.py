import numpy as np

def write_magnetization(self,nrep=5):
    """Extract the magnetization and write it in a file"""
    mx = self.extract("mx")
    my = self.extract("my")
    mz = self.extract("mz")
    g = self.geometry
    g.write_profile(mx,name="MX.OUT",normal_order=True,nrep=nrep)
    g.write_profile(my,name="MY.OUT",normal_order=True,nrep=nrep)
    g.write_profile(mz,name="MZ.OUT",normal_order=True,nrep=nrep)
    # this is just a workaround
    m = np.genfromtxt("MX.OUT").transpose()
    (x,y,z,mx) = m[0],m[1],m[2],m[3]
    my = np.genfromtxt("MY.OUT").transpose()[3]
    mz = np.genfromtxt("MZ.OUT").transpose()[3]
    np.savetxt("MAGNETISM.OUT",np.array([x,y,z,mx,my,mz]).T)
    return np.array([mx,my,mz])
