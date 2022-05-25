import os
import sys
import time

import numpy as np
import scipy.signal

from numba import jit
from matplotlib import pyplot

from utils import *
from core import *
#from lbi_core_sfixed import *


if __name__ == "__main__":
    if os.path.dirname(__file__) != os.getcwd():
        os.chdir(os.path.dirname(__file__))

    ## ========================================
    ## Configuration
    ## ========================================

    forNLoop = 3
    plotAtEnd = True
    lbi_len = "15k" # 5k, 10k, 15k

    y_ref = read_file("../profiles/" + lbi_len + "_w_slope_y_0001" + ".txt")[0]
    beta_ref = read_file("../profiles/" + lbi_len + "_w_slope_beta_0001" + ".txt")[0]

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
        #betaP1 = LBFiberHotStartCoreP1(y_ref, lambd, NumberOfIterations, 2**(-10)) # beta from LBI Core
        #betaP1n = LBFiberHotStartCoreP1PosProc(y_ref, betaP1)                      # beta after LSM (Least Squares Method)
        
        ## beta = LBI_Core + PosProc
        betaP1n = LBFiberHotStartP1(y_ref, lambd, NumberOfIterations, 2**(-10))
        
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

