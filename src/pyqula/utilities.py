import numpy as np

def obj2fun(x):
    if callable(x): return x
    else: return lambda i: x

get_callable = obj2fun
