import os
import glob
import profile
import sys
import time
import random

import numpy as np

from numba import jit


def printNonZero(beta1, beta2 = None):
    beta1 = np.array(beta1, dtype=np.float64)
    if beta2 is None:
        print("\n{:>8s}  -->  {:>15s}".format("INDEX", "VARIABLE 1"))
        for i in range(beta1.size):
            if beta1[i] != 0.0:
                print(f"{i:8d}  -->  {beta1[i]: 15.10f}")
    else:
        beta2 = np.array(beta2, dtype=np.float64)
        if beta1.size == beta2.size:
            print("\n{:>8s}  -->  {:>15s}   {:>15s}".format("INDEX", "VARIABLE 1", "VARIABLE 2"))
            for i in range(beta1.size):
                if (beta1[i] != 0.0) or (beta2[i] != 0.0):
                    print(f"{i:8d}  -->  {beta1[i]: 15.10f}   {beta2[i]: 15.10f}")
    print("")


def time_report(func):
    def wrapper(*arg, **kw):
        t1 = time.perf_counter()
        res = func(*arg, **kw)
        t2 = time.perf_counter()
        print(f"""Function "{func.__name__}" took {t2-t1} seconds..""")
        return res
    return wrapper


def time_wrapper(func):
    def wrapper(*arg, **kw):
        t1 = time.perf_counter()
        res = func(*arg, **kw)
        t2 = time.perf_counter()
        return res, func.__name__, (t2 - t1)
    return wrapper


@time_report
def read_file(filename):
    with open(filename) as f:
        d = f.read().split("\n")
        if d[-1] == "":
            d = d[0:-1]
        d = np.array(d, np.float64)
    return d


def countnz(x):
    return np.sum(x != 0.0)


def awgn_CRN(y, sigmaCRN):
    noise = np.random.normal(size = y.size)
    noise = noise / np.ndarray.max( np.abs(noise) )
    noise = noise * sigmaCRN
    return y + noise


def awgn_SNRdB(y, SNRdB):
    noise = np.random.normal(size = y.size)
    # SNR_lin = 10 ** (SNRdB / 10)
    SNR_lin = SNRdB
    noise = noise / np.sqrt( SNR_lin )
    return y + noise


@jit(nopython=True)
def CalcYFromWP1(beta):
    y = np.zeros(beta.size - 1)
    y_sum = 0.0
    for i in range(beta.size - 1):
        y_sum += beta[i+1]
        y[i] = (i+1)*beta[0] + y_sum
    return y


def GenerateFiberProfileP1(N, F, addNoise = True):
    # Create a Fiber Profile with:
    #   N: length of the profile
    #   F: number of faults
    #   addNoise: True or False
    #   SRNdB: the CRN is calculated by Equation 9. [ref]

    # Sort fault's position
    P = np.zeros(F, np.int64)
    value = 0
    for j in range(F):
        while np.sum(P == value): # avoid duplicates
            # N-3 --> the first two entries [0, 1] of beta shouldn't be faults
            value = np.int64( np.ceil((N-2) * np.random.random()) + 1 )
        P[j] = value
    
    # Sort the fault's magnitude (descending order)
    f = np.zeros(F)
    for j in range(F):
        if j == 0:
            f[j] = 5 + np.random.random()
        else:
            f[j] = 1/(j+1) * f[j-1] + np.random.random()

    # Compose vector beta
    beta = np.zeros(N+1)
    beta[P] = -f

    # Fiber slope!
    slope = -0.0002
    beta[0] = slope
    beta[1] = sum(f) - slope*N + 10

    Y = CalcYFromWP1(beta)

    # LinearY is the number of counts for each position.
    linearY = np.power(10.0, (Y/10.0))

    if addNoise:
        # PoissonNoise is the counting noise for each position
        PoissonNoise = np.sqrt(linearY)

        # This is an approximation! Adds AWGN with variance equal to the
        # variance of the counting noise to each position.
        for j in range(N):
            linearY[j] = awgn_CRN(linearY[j], PoissonNoise[j])

        # Adds white gaussian noise relative to the Coherent Rayleigh Noise.
        # Variance of the CRN noise as presented in Equation 9 of the manuscript
        DeltaNu1 = 100*1e9 # 100 GHz = 0.8 nm -- full DWDM channel occupied
        DeltaNu2 = 6*1e6   # 6 MHz -- linewidth of a bad DFB single-mode laser
        DeltaNu3 = 10*1e3  # 10 kHz -- linewidth of a descent external cavity single-mode laser
        sigmaCRN = np.sqrt( 2e8 / (4 * N * DeltaNu3) )
        # In the experimental measurements, we used a tunable external cavity single-mode laser.
        # Therefore, in order to truly simulate the experimental conditions,
        # we use DeltaNu3 in the generation script. This information has been included into the manuscript text.
        linearY = awgn_CRN(linearY, sigmaCRN)

        # Avoiding problems when returning to the logarithmic scale.
        # Remove any negative values, if any...
        linearY = np.where(linearY < 0, 0, linearY)

    # Return to logarithmic scale.
    Y = 10 * np.log10(linearY)

    return Y, beta


def GenerateFiberProfileNoSlope(N, F, addNoise = True):
        # Create a Fiber Profile with:
    #   N: length of the profile
    #   F: number of faults
    #   SRNdB: the CRN is calculated by Equation 9. [ref]

    # Sort fault's position
    P = np.zeros(F, np.int64)
    value = 0
    for j in range(F):
        while np.sum(P == value): # avoid duplicates
            # N-3 --> the first two entries [0, 1] of beta shouldn't be faults
            value = np.int64( np.ceil((N-2) * np.random.random()) + 1 )
        P[j] = value
    
    # Sort the fault's magnitude (descending order)
    f = np.zeros(F)
    for j in range(F):
        if j == 0:
            f[j] = 5 + np.random.random()
        else:
            f[j] = 1/(j+1) * f[j-1] + np.random.random()

    # Compose vector beta
    beta = np.zeros(N+1)
    beta[P] = -f

    # Fiber slope!
    slope = -0.0002
    beta[0] = slope
    beta[1] = sum(f) - slope*N + 10

    Y = CalcYFromWP1(beta)

    # LinearY is the number of counts for each position.
    linearY = np.power(10.0, (Y/10.0))

    if addNoise:
        # PoissonNoise is the counting noise for each position
        PoissonNoise = np.sqrt(linearY)

        # This is an approximation! Adds AWGN with variance equal to the
        # variance of the counting noise to each position.
        for j in range(N):
            linearY[j] = awgn_CRN(linearY[j], PoissonNoise[j])

        # Adds white gaussian noise relative to the Coherent Rayleigh Noise.
        # Variance of the CRN noise as presented in Equation 9 of the manuscript
        DeltaNu1 = 100*1e9 # 100 GHz = 0.8 nm -- full DWDM channel occupied
        DeltaNu2 = 6*1e6   # 6 MHz -- linewidth of a bad DFB single-mode laser
        DeltaNu3 = 10*1e3  # 10 kHz -- linewidth of a descent external cavity single-mode laser
        sigmaCRN = np.sqrt( 2e8 / (4 * N * DeltaNu3) )
        # In the experimental measurements, we used a tunable external cavity single-mode laser.
        # Therefore, in order to truly simulate the experimental conditions,
        # we use DeltaNu3 in the generation script. This information has been included into the manuscript text.
        linearY = awgn_CRN(linearY, sigmaCRN)

        # Avoiding problems when returning to the logarithmic scale.
        # Remove any negative values, if any...
        linearY = np.where(linearY < 0, 0, linearY)

    # Return to logarithmic scale.
    Y = 10 * np.log10(linearY)

    return Y, beta[1:]


def GenerateA(N):
    A = np.tri(N)
    return A


def GenerateAP1(N):
    A = np.c_[np.arange(1,N+1), np.tri(N)]
    return A


def AppendNtoFilename(prefix, filename, N = 1, N_pad = 4, ext = ".txt", filename_PrePost = False):
    if filename_PrePost:
        # PreFix  => "prefix_filename_[000N].txt"
        return "".join([prefix, "_", filename, "_", str(N).zfill(N_pad), ext])
    else:
        # PostFix => "prefix_[000N]_filename.txt"
        return "".join([prefix, "_", str(N).zfill(N_pad), "_", filename, ext])
    

# BatchGenerateFiberProfileP1("5k", 10, 5000, 5, "5000", plotGraph = True)
def BatchGenerateFiberProfileP1(path, numberOfProfiles, profileLength, numberOfFaults, prefix = "", N_pad = 4, ext = ".txt", filename_PrePost = False, plotGraph = False):
    """
    prefix: add a prefix to the filename, ex: "prefix_y.txt"
    PrePostN: add the profile's number as PreFix or PostFix
        true  => ex: "[0001]_y.txt"
        false => ex: "y_[0001].txt"
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        os.mkdir(path)

    N = profileLength
    F = numberOfFaults

    fileN = 1
    fn_y = AppendNtoFilename(prefix, "y", fileN, N_pad, ext, filename_PrePost)
    while os.path.isfile(os.path.join(path, fn_y)):
        fileN += 1
        fn_y = AppendNtoFilename(prefix, "y", fileN, N_pad, ext, filename_PrePost)

    if plotGraph:
        try:
            import pyqtgraph as pg
            import pyqtgraph.exporters
        except:
            plotGraph = False

    if plotGraph:
        app = 0
        app = pg.QtGui.QApplication([])
        win = pg.GraphicsWindow()
        win.setBackground('w')
        plot = win.addPlot()
        plot.addLegend(offset = (1, 0)) # right or left, bottom or top

        # preallocate
        y_x = np.arange(N)
        beta_x = np.arange(N+1)

    for i in range(numberOfProfiles):
        y, beta = GenerateFiberProfileP1(N, F)
        
        fn_y    = AppendNtoFilename(prefix, "y",    fileN, N_pad, ext, filename_PrePost)
        fn_beta = AppendNtoFilename(prefix, "beta", fileN, N_pad, ext, filename_PrePost)

        np.savetxt(os.path.join(path, fn_y),    y,    "%0.20f")
        np.savetxt(os.path.join(path, fn_beta), beta, "%0.20f")

        if plotGraph:
            plot.clear()
            plot.plot(y_x, y, pen = 'r', name = "y")
            plot.plot(beta_x, beta, pen = 'k', name = "beta")
            pg.QtGui.QApplication.processEvents()
            exporter = pg.exporters.ImageExporter(win.scene())
            fn_fig = AppendNtoFilename(prefix, "fig", fileN, N_pad, ".png", filename_PrePost)
            exporter.export(os.path.join(path, fn_fig))

        # last line of this for loop
        fileN += 1
    
    if plotGraph:
        win.close()


def BatchGenerateFiberProfileNoSlope(path, numberOfProfiles, profileLength, numberOfFaults, prefix = "", N_pad = 4, ext = ".txt", filename_PrePost = False, plotGraph = False):
    """
    prefix: add a prefix to the filename, ex: "prefix_y.txt"
    PrePostN: add the profile's number as PreFix or PostFix
        true  => ex: "[0001]_y.txt"
        false => ex: "y_[0001].txt"
    """
    path = os.path.abspath(path)
    if not os.path.exists(path):
        os.mkdir(path)

    N = profileLength
    F = numberOfFaults

    fileN = 1
    fn_y = AppendNtoFilename(prefix, "y", fileN, N_pad, ext, filename_PrePost)
    while os.path.isfile(os.path.join(path, fn_y)):
        fileN += 1
        fn_y = AppendNtoFilename(prefix, "y", fileN, N_pad, ext, filename_PrePost)

    if plotGraph:
        app = 0
        app = pg.QtGui.QApplication([])
        win = pg.GraphicsWindow()
        win.setBackground('w')
        plot = win.addPlot()
        plot.addLegend(offset = (1, 0)) # right or left, bottom or top

        # preallocate
        x = np.arange(N)

    for i in range(numberOfProfiles):
        y, beta = GenerateFiberProfileNoSlope(N, F)
        
        fn_y    = AppendNtoFilename(prefix, "y",    fileN, N_pad, ext, filename_PrePost)
        fn_beta = AppendNtoFilename(prefix, "beta", fileN, N_pad, ext, filename_PrePost)

        np.savetxt(os.path.join(path, fn_y),    y,    "%0.20f")
        np.savetxt(os.path.join(path, fn_beta), beta, "%0.20f")

        if plotGraph:
            try:
                import pyqtgraph as pg
                import pyqtgraph.exporters
            except:
                plotGraph = False

        if plotGraph:
            plot.clear()
            plot.plot(x, y, pen = 'r', name = "y")
            plot.plot(x, beta, pen = 'k', name = "beta")
            pg.QtGui.QApplication.processEvents()
            exporter = pg.exporters.ImageExporter(win.scene())
            fn_fig = AppendNtoFilename(prefix, "fig", fileN, N_pad, ".png", filename_PrePost)
            exporter.export(os.path.join(path, fn_fig))

        # last line of this for loop
        fileN += 1
    
    if plotGraph:
        win.close()


# TODO:
def ProcessFolderP1(path, filename_wildcard = "*y_*.txt"):
    fn_wc = os.path.join(path, filename_wildcard)
    yProfiles = glob.glob(fn_wc)

    return yProfiles
