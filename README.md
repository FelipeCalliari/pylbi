# pylbi
Python implementation of Linearized Bregman Iterations

## **Installation**

To install run: `python setup.py install`

## **Usage**
```python
>>> import pylbi
>>> y, beta_ref = pylbi.utils.GenerateFiberProfileP1(N=10_000, F=5)
>>> beta_lbi = pylbi.core.LBFiberHotStartP1(y=y, lambd=0.5, NumberOfIterations=450*y.size)
>>> pylbi.utils.printNonZero(beta_ref)
>>> pylbi.utils.printNonZero(beta_lbi)
```

## ***TODO***
 - Add SFixed implementation.
 - Add missing functions.