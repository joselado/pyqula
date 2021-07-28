import os
import shutil
import subprocess
from pathlib import Path,PurePath


## this is just a wrapper to several functions to deal with files ##


def rmdir(a):
    """Remove a directory with its contents"""
    try: shutil.rmtree(a)
    except: pass

mkdir = os.makedirs # create a directory
cpdir = shutil.copytree
chdir = os.chdir

def execute(ll,background=True):
    if type(ll)!=list: ll = [ll] # convert to list
    subprocess.Popen(ll)

def joinpath(*args):
    return PurePath(*args)
