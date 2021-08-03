def get_docstring(original):
  def wrapper(target):
    target.__doc__ = original.__doc__
    return target
  return wrapper
