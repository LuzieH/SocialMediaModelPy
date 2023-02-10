from SocialMediaModelPy import abm
import numpy as np

def test_dimensions():
    N = 10
    M = 2
    L = 4 
    NT = 2
    x0,y0,z0,A,B,C0 = abm.init(N)
    ops = abm.opinions(x0,y0,z0,A,B,C0)
    xs,ys,zs,Cs = ops.run(timesteps=NT)
    assert np.shape(x0)==(N,2), "The initial dimension of x is wrong, it should be of the shape N x 2."
    assert np.shape(y0)==(M,2), "The initial dimension of y is wrong, it should be of the shape M x 2."
    assert np.shape(z0)==(L,2), "The initial dimension of z is wrong, it should be of the shape L x 2."
    assert np.shape(A)==(N,N), "The dimension of A is wrong, it should be of the shape N x N."
    assert np.shape(B)==(N,M), "The dimension of B is wrong, it should be of the shape N x M."
    assert np.shape(C0)==(N,L), "The initial dimension of C is wrong, it should be of the shape N x L."
    assert len(xs) == NT+1, "The output realization contains too few time points."
    assert np.shape(xs[-1])==(N,2), "The output dimension of x is wrong."
    assert np.shape(ys[-1])==(M,2), "The output dimension of y is wrong."
    assert np.shape(zs[-1])==(L,2), "The output dimension of z is wrong."
    assert np.shape(Cs[-1])==(N,L), "The output dimension of C is wrong."


test_dimensions()