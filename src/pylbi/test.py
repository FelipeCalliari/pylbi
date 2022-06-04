import os
import sys
import time

import numpy as np
import scipy.signal

from numba import jit
from matplotlib import pyplot

from pylbi.utils import *
from pylbi.core import *
#from pylbi.lbi_core_sfixed import *


if __name__ == "__main__":
    if os.path.dirname(__file__) != os.getcwd():
        os.chdir(os.path.dirname(__file__))

    ## ========================================
    ## Configuration
    ## ========================================

    forNLoop = 3
    plotAtEnd = True
    lbi_len = "15k" # 5k, 10k, 15k

    y_ref = read_file("../../../profiles/" + lbi_len + "_w_slope_y_0001" + ".txt")
    beta_ref = read_file("../../../profiles/" + lbi_len + "_w_slope_beta_0001" + ".txt")

    lambd = 0.5 # I am avoid using the word "lambda", as it is a python reserved word
    IterationsPerSample = 450
    NumberOfIterations = y_ref.size*IterationsPerSample

    print(f"lambda              = {lambd}")
    print(f"Y_len               = {y_ref.size}")
    print(f"IterationsPerSample = {IterationsPerSample}")
    print(f"NumberOfIterations  = {NumberOfIterations}")

    for i in range(forNLoop):
        print("")
        ts = time.perf_counter()

        ## ========================================
        
        ## beta = PosProc(betaLBI = LBI_Core())
        #betaP1 = LBFiberCoreP1(y_ref, lambd, NumberOfIterations) # beta from LBI Core
        #betaP1n = LBFiberCoreP1PosProc(y_ref, betaP1)            # beta after LSM (Least Squares Method)
        
        ## beta = LBI_Core + PosProc
        betaP1n = LBFiberP1(y_ref, lambd, NumberOfIterations)
        
        ## ========================================

        tp = time.perf_counter() - ts
        print(f"Elapsed time: {tp} s")

        printNonZero(betaP1n)

    if plotAtEnd:
        # Create two subplots and unpack the output array immediately
        f, (ax1, ax2, ax3) = pyplot.subplots(3, 1, sharey=True)
        ax1.plot(range(y_ref.size), y_ref)
        ax1.set_title("LBI")
        ax2.plot(range(beta_ref.size), beta_ref)
        ax2.set_title("Reference Beta")
        ax3.plot(range(betaP1n.size), betaP1n)
        ax3.set_title("LBI's Beta")
        pyplot.show()
else:
    print("\n[*] Creating 10000 points fiber profile with noise...")
    GenProfile = time_report(GenerateFiberProfileP1)
    y, beta_ref = GenProfile(N=10_000, F=5, addNoise=True)
    LBI = time_report(LBFiberP1)
    print("[*] Running LBI...")
    beta_lbi = LBI(y, 0.5, 450*y.size)
    printNonZero(beta_ref, beta_lbi)
    # beta_c = np.c_[beta_ref, beta_lbi]
    # print("\n   INDEX          REF              LBI")
    # for i in range(beta_c.shape[0]):
    #     if (beta_c[i][0] != 0.0) or (beta_c[i][1] != 0.0):
    #         print(f"{i:>5d}  -->   {beta_c[i][0]:+0.10f}   {beta_c[i][1]:+0.10f}")

    print("[*] Creating 10000 points fiber profile without noise...")
    GenProfile = time_report(GenerateFiberProfileP1)
    y, beta_ref = GenProfile(N=10_000, F=5, addNoise=False)
    LBI = time_report(LBFiberP1)
    print("[*] Running LBI...")
    beta_lbi = LBI(y, 0.5, 450*y.size)
    printNonZero(beta_ref, beta_lbi)
    # beta_c = np.c_[beta_ref, beta_lbi]
    # print("\n   INDEX          REF              LBI")
    # for i in range(beta_c.shape[0]):
    #     if (beta_c[i][0] != 0.0) or (beta_c[i][1] != 0.0):
    #         print(f"{i:>5d}  -->   {beta_c[i][0]:+0.10f}   {beta_c[i][1]:+0.10f}")

