
import numpy as np



def label2k(g,kl):
    """Given a kpoint label, return the kpoint"""
    if kl=="G": return [0.,0.,0.]
    elif kl=="M": return [0.5,0.,0.]
    elif kl=="K":
        from .locate import target_moduli
        from .mapping import unitary
        vm = np.sqrt(3) # moduli
        k = g.k2K(target_moduli(unitary(g.b1),unitary(g.b2),vm))
        return k*2/(3.*np.sqrt(3))
    elif kl=="K'": return -label2k(g,"K")
    elif kl=="M1": return [.5,.0,.0]
    elif kl=="M2": return [.0,.5,.0]
    elif kl=="M3": return [.5,.5,.0]
    elif kl=="X": return [.5,.0,.0]
    elif kl=="Y": return [.0,.5,.0]
    else: raise



