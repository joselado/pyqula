"""Global switch for the files pyqula writes to the working directory.

Many routines write their result to a file as well as returning it
(BANDS.OUT, DOS.OUT, CHERN.OUT...), each with a write= keyword and a
default of its own. This switch sets write= for all of them at once:

    from pyqula import filewrite
    filewrite.set_write(False)   # no routine writes, unless asked to
    (k,e) = h.get_bands()        # returns the bands, writes nothing
    h.get_bands(write=True)      # a keyword given in the call wins
    filewrite.set_write(None)    # back to each routine's own default

set_write(True) makes every routine write, including the few whose own
default is not to (chern_density, for instance).

The order is: a write= passed in the call, then the global switch, then
the routine's own default. The switch only reaches routines that take a
write= keyword; functions whose only job is writing a file (g.write(),
h.write_hopping()...) are not affected by it.
"""

_write = None # None: every routine keeps its own default


def get_write():
    """The global setting: True, False, or None when unset"""
    return _write


def set_write(value):
    """Make every routine that takes write= write its files (True), not
    write them (False), or follow its own default (None). A write= given
    in the call still wins over this. Returns what was set"""
    global _write
    if value is not None and not isinstance(value,bool):
        raise ValueError("set_write takes True, False or None (each "
                "routine's own default), got "+repr(value))
    _write = value
    return _write


def resolve(write,default):
    """The write= a routine should use: the one it was called with if any,
    else the global setting if set, else the routine's own default"""
    if write is not None: return write
    if _write is not None: return _write
    return default
