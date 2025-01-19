# SQP (Sequential Quadratic Programming)

While using the builtin SQP optimizer, you can follow the same process as in the previous section
except when importing the optimizer.

You need to import the optimizer as shown in the following code:

```py
from modopt.optimization_algorithms import SQP
```

Options available are: `max_iter`, `opt_tol`, and `feas_tol`.
Options could be set by just passing them as kwargs when 
instantiating the SQP optimizer object.
For example, we can set the maximum number of iterations `max_itr` 
and the optimality tolerance `opt_tol` shown below.

```py
optimizer = SQP(prob, max_itr=20, opt_tol=1e-8)
```