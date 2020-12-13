DEBUG = True
VERBOSE = False 

def log(*args):
    if VERBOSE:
        print(*args)
