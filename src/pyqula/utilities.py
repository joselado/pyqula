import numpy as np

def obj2fun(x):
    if callable(x): return x
    else: return lambda i: x

get_callable = obj2fun


def check_delta(delta,name="delta"):
    """Complain about a non-positive analytic continuation / broadening.

    The broadening enters the spectral functions as a Lorentzian
    d/(d^2+(E-Ei)^2), so a negative d silently returns a negative density
    of states and a zero d gives a division by zero."""
    import numpy as np
    d = np.array(delta).real
    if not np.all(np.isfinite(d)) or np.any(d<=0.0):
        raise ValueError(name+" is the spectral broadening (the imaginary "
          +"part of the energy) and must be a finite positive number, got "
          +str(delta)+". A negative broadening returns a negative density "
          +"of states.")


def rename_kwarg(kwargs,alias,name):
    """Accept an alternative spelling for a keyword argument.

    Several entry points of the library spell the same quantity
    differently (e.g. the energy is `e` for the LDOS but `energy` for the
    Green's function and transport routines). Those aliases used to be
    swallowed by a trailing **kwargs and silently ignored."""
    if alias not in kwargs: return kwargs # nothing to do
    if name in kwargs:
        raise TypeError("got both '"+name+"' and its alias '"+alias
          +"'; pass only one of them")
    kwargs = dict(kwargs) # do not mutate the caller's dictionary
    kwargs[name] = kwargs.pop(alias) # rename
    return kwargs
