import numpy as np
from finitediff import get_weights

def make_D_fornberg(y,m,npoints=5):
    """Returns a differentiation matrix for the mth derivative on a nonuniform grid y, using a npoints-point stencil (uses Fornberg's algorithm)
    """
    N=len(y)
    assert N>=npoints
    D=np.zeros((N,N))
    for i in range(npoints//2):
        D[ i,:npoints] = get_weights(y[:npoints],y[i],-1,m)[:,m]
        D[-i-1,-npoints:] = get_weights(y[-npoints:],y[-i-1],-1,m)[:,m] 
    for i in range(npoints//2,N-npoints//2):
        D[i,i-npoints//2:i+npoints//2+1] = get_weights(y[i-npoints//2:i+npoints//2+1],y[i],-1,m)[:,m]   
    return D
