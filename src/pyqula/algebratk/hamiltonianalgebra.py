

def add(h1,h2):
    mo1 = h1.get_multihopping()
    mo2 = h2.get_multihopping()
    h3 = h1.copy()*0. # copy
    h3.set_multihopping((mo1+mo2).copy())
    return h3

def rmul(h,a):
    mo = h.get_multihopping()
    h1 = h.copy() # copy
    h1.set_multihopping(a*mo)
    return h1

def mul(h,a):
    mo = h.get_multihopping()
    h1 = h.copy() # copy
    h1.set_multihopping(mo*a)
    return h1

