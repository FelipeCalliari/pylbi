r"""A simple example:

>>> import pylbi
>>> y, beta_ref = pylbi.utils.GenerateFiberProfileP1(N=10_000, F=5)
>>> beta_lbi = pylbi.core.LBFiberHotStartP1(y=y, lambd=0.5, NumberOfIterations=450*y.size)
>>> pylbi.utils.printNonZero(beta_ref)
>>> pylbi.utils.printNonZero(beta_lbi)
"""


__version__ = "0.0.3"


import pylbi.core as core
import pylbi.utils as utils
