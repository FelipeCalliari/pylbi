import numpy as np
import scipy.signal

from numba import jit


@jit(nopython=True)
def shrink(u, lambd):
    u = np.sign(u) * np.maximum(np.abs(u) - lambd, 0)
    return u


# Compute LBI with Slope
def LBFiberHotStartP1(y, lambd = 0.5, NumberOfIterations = None, FirstColumnScaling = 2**(-10), NonSparseThresholdFactor = 0.0125):
    betaP1  = LBFiberHotStartCoreP1(y, lambd, NumberOfIterations, FirstColumnScaling)
    betaP1n = LBFiberHotStartCoreP1PosProc(y, betaP1, FirstColumnScaling, NonSparseThresholdFactor)
    return betaP1n


@jit(nopython=True)
def LBFiberHotStartCoreP1(y, lambd = 0.5, NMaxIterations = None, FirstColumnScaling = 2**(-10)):
    m = np.int64(y.size)
    p = np.int64(m + 1)

    if NMaxIterations is None:
        NMaxIterations = 450 * m
    NMaxIterations = np.int64(NMaxIterations)

    beta = np.zeros(p, np.float64)
    v = np.zeros(p, np.float64)

    lambd = np.float64(lambd)
    FirstColumnScaling = np.float64(FirstColumnScaling)
    ScalingSquared = FirstColumnScaling**2
    ScaledGradient = np.float64(0.0)
    k = np.int64(1)

    for i in range(NMaxIterations):
        beta[0:k+1] = shrink(v[0:k+1], lambd)

        if k == 1:
            mu = 1.0 / ( np.power(k,2) * ScalingSquared + 1)
        else:
            mu = 1.0 / ( np.power(k,2) * ScalingSquared + k-1)

        FirstColumnScalingTimesk = np.float64(FirstColumnScaling*k)
        ScaledGradient = FirstColumnScalingTimesk * beta[0]
        ScaledGradient += np.sum(beta[1:k+1])

        InstantaneousError = y[k-1] - ScaledGradient
        ScaledGradient = mu * InstantaneousError

        v[0]     += FirstColumnScalingTimesk*ScaledGradient
        v[1:k+1] += ScaledGradient

        if k < m:
            k += 1
        else:
            k = 1

    return beta


def LBFiberHotStartCoreP1PosProc(y, beta, FirstColumnScaling = 2**(-10), NonSparseThresholdFactor = 0.0125):
    # pos-processing
    beta = np.copy(beta)
    beta[0] = beta[0] / FirstColumnScaling

    selection = scipy.signal.find_peaks(np.abs(beta), NonSparseThresholdFactor)[0]
    if beta[0] == 0.0:
        selection = np.r_[0, selection]

    #A = np.c_[np.arange(1,y.size+1), np.tril(np.ones((y.size, y.size)))]
    A = np.c_[np.arange(1,y.size+1), np.tri(y.size)]
    nonzerow = np.matmul( np.matmul( np.linalg.inv( np.matmul(np.transpose(A[:, selection]), A[:, selection]) ), np.transpose(A[:, selection]) ), y)

    beta = np.zeros(beta.size, np.float64)
    beta[selection] = nonzerow

    return beta


# Compute LBI without Slope
def LBFiberHotStart(y, lambd = 0.5, NumberOfIterations = None, FirstColumnScaling = 2**(-10), NonSparseThresholdFactor = 0.0125):
    beta  = LBFiberHotStartCore(y, lambd, NumberOfIterations, FirstColumnScaling)
    betan = LBFiberHotStartCorePosProc(y, beta, NonSparseThresholdFactor)
    return betan


@jit(nopython=True)
def LBFiberHotStartCore(y, lambd = 0.5, NMaxIterations = None, FirstColumnScaling = 2**(-10)):
    m = np.int64(y.size)

    if NMaxIterations is None:
        NMaxIterations = 450 * m
    NMaxIterations = np.int64(NMaxIterations)

    beta = np.zeros(m, np.float64)
    v = np.zeros(m, np.float64)

    lambd = np.float64(lambd)
    FirstColumnScaling = np.float64(FirstColumnScaling)
    ScalingSquared = FirstColumnScaling**2
    ScaledGradient = np.float64(0.0)
    k = np.int64(0)

    for i in range(NMaxIterations):
        beta[0:k] = shrink(v[0:k], lambd)

        mu = 1.0 / k

        ScaledGradient += np.sum(beta[0:k])

        InstantaneousError = y[k] - ScaledGradient
        ScaledGradient = mu * InstantaneousError

        v[0:k] += ScaledGradient

        if k < m-1:
            k += 1
        else:
            k = 0

    return beta


def LBFiberHotStartCorePosProc(y, beta, NonSparseThresholdFactor = 0.0125):
    # pos-processing
    beta = np.copy(beta)

    selection = scipy.signal.find_peaks(np.abs(beta), NonSparseThresholdFactor)[0]
    if beta[0] == 0.0:
        selection = np.r_[0, selection]

    A = np.tri(y.size)
    nonzerow = np.matmul( np.matmul( np.linalg.inv( np.matmul(np.transpose(A[:, selection]), A[:, selection]) ), np.transpose(A[:, selection]) ), y)

    beta = np.zeros(beta.size, np.float64)
    beta[selection] = nonzerow

    return beta


