import numpy as np
import scipy.signal

from numba import jit
from fixed_point_x import SaturationMode, TruncationMode, SFixed2


@jit(nopython=True)
def shrink(u, lambd):
    u = np.sign(u) * np.maximum(np.abs(u) - lambd, 0)
    return u


@jit(nopython=True)
def LBFiberHotStartQ(y, FirstColumnScaling = 2**(-10),
                     lambd = 0.5, NMaxIterations = None,
                     BInt = 0, BFrac = -17,
                     TMode = TruncationMode.TRUNCATION,
                     SMode = SaturationMode.WRAP,
                     RAM_NUMBER = 32):
    m = np.int64(y.size)
    p = np.int64(m)

    beta = np.zeros(y.size, np.float64)
    v = np.zeros(y.size, np.float64)

    FirstColumnScaling = np.float64(FirstColumnScaling)
    ScalingSquared = FirstColumnScaling**2
    lambd = np.float64(lambd)

    # s = SFixed2(0, BInt, BFrac, SaturationMode.WRAP, TruncationMode.TRUNCATION)
    if NMaxIterations is None:
        NMaxIterations = 500 * m
    NMaxIterations = np.int64(NMaxIterations)
    k = np.int64(1)

    for i in range(NMaxIterations):
        mu = 1.0 / k
        ScaledGradient = np.sum(beta[0:k-1])
        InstantaneousError = y[k-1] - ScaledGradient
        ScaledGradient = mu * InstantaneousError

        #for j in range(k):
        #    v[j] = v[j] + ScaledGradient
        v[0:k] += ScaledGradient

        #for j in range(k):
        #    beta[j] = shrink(beta[j], lambd)
        beta[0:k] = shrink(v[0:k], lambd)

        if k < m:
            k += 1
        else:
            k = 1

##        if (i % (NMaxIterations/100) == 0):
##            print(100.0*i / NMaxIterations)
##            print(time.time()-ts)
##            print(f"{100.0 * i / NMaxIterations}% in {time.time()-ts:5.3f}s")

    return beta
