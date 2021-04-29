# JaDaPy

JaDaPy is a Python package that implements JDQR and JDQZ using SciPy and NumPy and has optional Trilinos integration.

## Eigenvalue computation

A given generalized eigenvalue problem of the form

βAv = αBv

can be solved using JDQZ using
```Python
    alpha, beta = jdqz.jdqz(A, B)
```
or
```Python
    alpha, beta, v = jdqz.jdqz(A, B, return_eigenvectors=True)
```

## Installation

JaDaPy is best installed in a [virtual environment](https://docs.python.org/3/library/venv.html).
We state the most common steps for creating and using a virtual environment here.
Refer to the documentation for more details.

To create a virtual environment run
```
python3 -m venv /path/to/new/virtual/environment
```

and to activate the virtual environment, run
```
source /path/to/new/virtual/environment/bin/activate
```

After this, we can install JaDaPy from the JaDaPy source directory.
```
pip install .
```

This will also install all of the requirements.
If one does not want to install JaDaPy, but instead just wants to run it from the source directory, one can install the requirements by running
```
pip install -r requirements.txt
```