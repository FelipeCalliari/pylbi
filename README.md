# pylbi
Python implementation of Linearized Bregman Iterations

----------
## **Installation**

It's recommended to install the package's requirements using `pip install -r requirements.txt` before installing the package.

To install run: `python setup.py install`

After the installation process it's possible to run `>>> import pylbi.test`, to verify that the installation went well.

----------
## **Functions**

* LBFiberP1(y, <textorange>lambd, NumberOfIterations, FirstColumnScaling, NonSparseThresholdFactor</textorange>)
   - This function uses the routines: `LBFiberCoreP1` and `LBFiberCoreP1PosProc`.
* LBFiberHotStartP1(y, beta, v, <textorange>lambd, NumberOfIterations, FirstColumnScaling, NonSparseThresholdFactor</textorange>)
   - This function uses the routines: `LBFiberHotStartCoreP1` and `LBFiberCoreP1PosProc`.

The arguments in <textorange>orange</textorange> are **optional**. If not set, these optional arguments have default values as follows: `lambd = 0.5`, `NumberOfIterations = 450*y.size`, `FirstColumnScaling = 2**(-10)` and `NonSparseThresholdFactor = 0.0125`.

The *P1* in the function name indicates that this function detects slope. 

There are also the functions: `LBFiber`, `LBFiberHotStart`, `LBFiberCore` and `LBFiberCorePosProc` without the slope detection feature. 

----------
## **Usage**

```python
>>> import numpy as np
>>> import pylbi
>>> # Generate Profile
>>> y, beta_ref = pylbi.utils.GenerateFiberProfileP1(N=10_000, F=5, addNoise=False)
>>> # Run LBI
>>> beta_lbi = pylbi.core.LBFiberHotStartP1(y, np.zeros(y.size+1), np.zeros(y.size+1), lambd=0.5, NumberOfIterations=450*y.size)
>>> # Print results
>>> pylbi.utils.printNonZero(beta_ref, beta_lbi)
```

To measure the execution time, we can use the python decorator `pylbi.utils.time_report` as folows:

```python
>>> import numpy as np
>>> import pylbi
>>> # time_report
>>> LBI = pylbi.utils.time_report(pylbi.core.LBFiberP1)
>>> # Generate Profile
>>> y, beta_ref = pylbi.utils.GenerateFiberProfileP1(N=10_000, F=5, addNoise=False)
>>> # Run LBI
>>> beta_lbi = LBI(y, lambd=0.5, NumberOfIterations=450*y.size)
>>> # Print results
>>> pylbi.utils.printNonZero(beta_ref, beta_lbi)
```

Finally to see the raw results from **Linearized Bregman Iterations** and its **post-processing**, we can use:

```python
>>> import numpy as np
>>> import pylbi
>>> # time_report
>>> LBI_Core = pylbi.utils.time_report(pylbi.core.LBFiberCore)
>>> LBI_PosProc = pylbi.utils.time_report(pylbi.core.LBFiberHotStartCoreP1PosProc)
>>> # Generate Profile
>>> y, beta_ref = pylbi.utils.GenerateFiberProfileNoSlope(N=10_000, F=5, addNoise=False)
>>> # Run LBI
>>> beta_lbi_raw = LBI_Core(y=y, lambd=0.5, NumberOfIterations=450*y.size)
>>> beta_lbi_withPP = LBI_PosProc(y, beta_lbi_raw)
>>> # Print results
>>> pylbi.utils.printNonZero(beta_ref, beta_lbi_withPP)
```

Running LBI without slope:

```python
>>> import numpy as np
>>> import pylbi
>>> # time_report
>>> LBI_Core = pylbi.utils.time_report(pylbi.core.LBFiberCore)
>>> LBI_PosProc = pylbi.utils.time_report(pylbi.core.LBFiberCorePosProc)
>>> # Generate Profile
>>> y, beta_ref = pylbi.utils.GenerateFiberProfileNoSlope(N=10_000, F=5, addNoise=False)
>>> # Run LBI
>>> beta_lbi_raw = LBI_Core(y, lambd=0.5, NumberOfIterations=450*y.size)
>>> beta_lbi_withPP = LBI_PosProc(y, beta_lbi_raw)
>>> # Print results
>>> pylbi.utils.printNonZero(beta_ref, beta_lbi_withPP)
```

----------
## ***TODOs***

 - <textgreen>DONE:</textgreen> Changed `requirements.txt`
 - <textred>TODO:</textred> Add SFixed implementation.
 - <textred>TODO:</textred> Add missing functions.




<style>
   textred { color: Red }
   textorange { color: Orange }
   textgreen { color: Green }
   textblue { color: Blue }
</style>