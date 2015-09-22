Introduction
============

This is the companion repository to the paper, "Finite Element Exterior
Calculus with Lower-order Terms", by Douglas N. Arnold and Lizao Li.  It
contains the complete numerical results and the source code for the
numerical experiments.

Numerical results
=================

The numerical results are collected in the directory named `results`.  Raw
output from the code conforms to the naming scheme

    results-3d-[form degree]-form-deg[r].txt

where the form degree is either 1 or 2 and r is the polynomial degree for
the FEEC family (please refer to the paper to see explicitly which
polynomial degree is used for an individual variable given r for a
particular element pair). Inside each file, every section looks like

    [pair: ++   deg: 2   lots: FFFTF]
      h       sigma  rate        dsigma  rate  
    ---  ----------  ------  ----------  ------ 
      2  7.3338e-02          1.0180e+00          ....
      4  9.4712e-03  2.38    2.5864e-01  1.59  
      8  1.1064e-03  2.83    6.0866e-02  1.91  
     16  1.3089e-04  3.03    1.4532e-02  2.03  

In the header, `pair ++ deg: 2` means that P3Lambda x P2Lambda elements are
used. `lots: FFFTF` means that we only have a nonzero l4 lower-order
term. The first column is the number of nodes per edge. For example, for
the first row, we have a 2x2x2 random quasi-uniform mesh of the unit
cube. The named columns are absolute L2 errors. For example, `7.3338e-02`
is the L2 error in sigma. `rate` is computed from the successive rows.

The file `rates_summary.txt` gives a more readable summary of the raw
output. Every section is of the form

    [3d, 2-form, N2curl2 x RT2]
            3       2       2       2
    l1      3       2       2       2
    l2      2       2       2       2
    l3      3       2       2       2
    l4      3       2       2       2
    l5      3       2       2       2

The header says that this is for the 2-form Hodge Laplacian on the 3D unit
cube. The element pair used is Nedelec edge elements of the second kind of
degree 2 and Raviart-Thomas elements of degree 2. The first column
indicates which lower-order term is present (empty means no lower-order
terms). The subsequent four columns are the convergence rates for sigma,
dsigma, u, du, respectively. For example, the first row reads, without any
lower-order terms, the L2-convergences rates are h^3, h^2, h^2, h^2, for
sigma, dsigma, u, du, respectively.

Source code
===========

To reproduce all the results in this repository and try other examples
yourself, you can directly run the source code included in the directory
named `src`.

First, you need a working FEniCS environment. It is an open source project
and easy to install on Linux systems. Please refer to its official website
<http://fenicsproject.org/> for installation instruction.  Alternatively,
you can use the official Virtualbox image or Docker container image at
<https://github.com/FEniCS/docker>.

In addition to that, you also need python packages `tabulate` and
`sympy`. On most Linux systems, they can be installed by

    sudo easy_install tabulate sympy

Both are open source and easy to install. `tabulate` is used to print
pretty tables while `sympy` is used for the symbolic computation of the
right-hand side and derivatives of the exact solution.

Once the environment is ready, the two scripts `3d-1-form.py`,
`3d-2-form.py` can be executed by:

    python 3d-1-form.py

or with MPI (recommended, these problems are pretty large):

    mpirun -np 8 python 3d-1-form.py

Both scripts are well-documented. Basically, it will loop through all the
cases specified at the end of the script and print out the numerical errors
in tables mentioned in the [Numerical results] section. The lower-order
terms are set in the `make_data` function.
