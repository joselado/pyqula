from . import operators
import numpy as np


def get_operator(self,name,**kwargs):
      """Return the conventional operator"""
      if name=="None": return None
      elif name in ["berry","Berry"]: 
          return operators.get_berry(self,**kwargs)
      elif name=="valleyberry": 
          return operators.get_operator_berry(self,"valley",**kwargs)
      elif name=="szvalleyberry": 
          return operators.get_operator_berry(self,"sz_valley",**kwargs)
      elif name=="szberry": return operators.get_sz_berry(self,**kwargs)
      elif name in ["sx","Sx"]: return operators.get_sx(self)
      elif name in ["sy","Sy"]: return operators.get_sy(self)
      elif name in ["sz","Sz"]: return operators.get_sz(self)
      elif name=="current": return operators.get_current(self)
      elif name in ["bulk","Bulk"]: 
          return operators.get_bulk(self,**kwargs)
      elif name in ["edge","Edge","surface","Surface"]: 
          return operators.get_surface(self,**kwargs)
      elif name=="velocity": return operators.get_velocity(self)
      elif name=="sublattice": return operators.get_sublattice(self,mode="both")
      elif name=="sublatticeA": return operators.get_sublattice(self,mode="A")
      elif name=="sublatticeB": return operators.get_sublattice(self,mode="B")
      elif name=="interface": return operators.get_interface(self)
      elif name=="spair": return operators.get_pairing(self,ptype="s")
      elif name=="deltax": return operators.get_pairing(self,ptype="deltax")
      elif name=="deltay": return operators.get_pairing(self,ptype="deltay")
      elif name=="deltaz": return operators.get_pairing(self,ptype="deltaz")
      elif name=="electron": return operators.get_electron(self)
      elif name=="hole": return operators.get_hole(self)
      elif name in ["up"]: return operators.get_up(self)
      elif name in ["dn","down"]: return operators.get_dn(self)
      elif name=="zposition": return operators.get_zposition(self)
      elif name=="surface": return operators.get_surface(self)
      elif name=="ldos": 
          from . import ldos
          return ldos.ldos_projector(self,**kwargs)
      elif name=="yposition": return operators.get_yposition(self)
      elif name=="xposition": return operators.get_xposition(self)
      elif name=="electrons": return operators.get_electron(self)
      elif name=="mx":
        return self.get_operator("sx")@self.get_operator("electron")
      elif name=="my":
        return self.get_operator("sy")@self.get_operator("electron")
      elif name=="mz":
        return self.get_operator("sz")@self.get_operator("electron")
      elif name=="layer": return operators.get_layer(self,**kwargs)
      elif name=="valley":  return operators.get_valley(self)
      elif name=="valley_x":  return operators.get_valley_taux(self)
      elif name=="inplane_valley": return operators.get_inplane_valley(self)
      elif name=="valley_upper" or name=="valley_top":
        return operators.get_valley_layer(self,n=-1)
      elif name=="inplane_valley_upper":
        print("This operator only makes sense for TBG")
        ht = self.copy()
        ht.geometry.sublattice = self.geometry.sublattice * (np.sign(self.geometry.z)+1.0)/2.0
        return operators.get_inplane_valley(ht)
      elif name=="valley_lower" or name=="valley_bottom":
        return operators.get_valley_layer(self,n=0)
      elif name in ["ipr","IPR"]: return operators.ipr
      elif name=="potential":  return operators.get_potential(self,**kwargs)
      else: raise


get_scalar_operator = get_operator

