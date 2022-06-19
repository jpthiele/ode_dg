# ode_dg
Solver for a simple ODE using discontinuous Galerkin Finite Elements.
This is based on https://dealii.org/current/doxygen/deal.II/step_3.html which solves a simple PDE.

## Problem Description ##
Here we solve the ODE
$u'(t) = \lambda u(t)$ on $I=(0,T)$  
with $u(0) = u^0 = 1$.  
Next we split the closure of our interval, i.e. $[0,T]$, into M intervals $I_m(t_{m-1},t_m}$ such that $\bar{I} = \{0\}\cup I_1 \cup I_2 \cup \dots \cup I_M$ 
and $t_M =T$.  
By multiplying with suitable discontinuous Galerkin test functions $\varphi\in dG(r)$of order $r$, including jump terms and integrating over the temporal interval 
we obtain the weak formulation.  
Find $u\in dG(r)$ such that:  
$\sum\limits_{m=1}^M \int\limits_{I_m} \varphi(t)u'(t) - \varphi(t)u(t)\mathrm{d}t + \varphi_{m-1}(t_{m-1})(u_m-u_{m-1})$ 
and $u_0 = u^0$.
## Main differences to step-3 ##
Compared to solving Poisson's equation there are a few key differences
+ We need to add the jump term entries to our sparsity pattern. This is done in setup_system()
+ We have no rhs, but instead have $u(t)\varphi(t)$ to assemble
+ The first argument of the previous Laplace term now has only one derivative/shape_grad
+ The jump term also have to be assembled.
+ As the problem is nonsymmertric so the CG solver does not work anymore and was changed to the direct solver UMFPACK
## 1D Output ##
The function output_results() is written in a way that enables us to also visualize discontinuous solutions of higher order (r>1).  
For this reason an iterated quadrature rule is used to obtain the solution behavior on a single element.  
The resulting time points and solution values are printed as matrices in Matlab Syntax.  
This way the plots will be properly graphed as discontinuous.
## How to run the code ##
You need to install [deal.II](https://dealii.org/) in your system.  
Then clone or download this code and change to it's directory. Then call
```
cmake -DDEAL_II_DIR=<path_to_your_deal_install> .
make run
```
to run the program.
