This repository contains code to solve for 1D & 2D internal waves using spectral methods, as well as performing internal wave tomography.

A 1D tomography example is included in `tomo1D.ipynb`. It uses synthetic 1D internal wave data generated using a finite-difference model. The same 1D wave equation is solved using a spectral method in `spectral1D.ipynb`. `spectral2D.ipynb` contains a preliminary spectral code to solve for internal waves in 2D.

A .yml file is included that contains all necessary python packages to run the code and produce the figures. Create a conda environment using `conda env create -f environment.yml`, then activate with `conda activate iwtomo`, launch a Jupyter notebook and you are hopefully all set!

