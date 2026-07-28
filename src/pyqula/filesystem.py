import os
import glob
import shutil
import subprocess
from pathlib import Path,PurePath


## this is just a wrapper to several functions to deal with files ##
## routing file/process operations through here (instead of os.system
## shell strings like "rm -f", "cp", "mkdir", "touch") is what keeps the
## library working on Linux, Mac and Windows alike ##


def rmdir(a):
    """Remove a directory with its contents"""
    try: shutil.rmtree(a)
    except: pass

mkdir = os.makedirs # create a directory
cpdir = shutil.copytree
chdir = os.chdir

def rmfile(a):
    """Remove a single file if it exists"""
    try: os.remove(a)
    except: pass

def rmglob(pattern):
    """Remove every file or directory matching a glob pattern"""
    for name in glob.glob(pattern):
        if os.path.isdir(name): rmdir(name)
        else: rmfile(name)

def cpfile(source,target):
    """Copy a single file"""
    shutil.copyfile(source,target)

def touch(a):
    """Create an empty file, equivalent to the Unix touch command"""
    Path(a).touch()

def execute(ll,background=True):
    if type(ll)!=list: ll = [ll] # convert to list
    subprocess.Popen(ll)

def joinpath(*args):
    return PurePath(*args)
