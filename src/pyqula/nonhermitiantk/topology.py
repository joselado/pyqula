
# redefine topological invariants

def get_berry_curvature(self,**kwargs):
    """Compute Berry curvature"""
    from ..topology import get_berry_curvature_master as BC
    return BC(self,mode="Green",**kwargs)
