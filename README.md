# Are AI weather models learning atmospheric physics? A sensitivity analysis of cyclone Xynthia
This repository contains the material and guidelines to reproduce the results presented in the manuscript entitled **Are AI weather models learning atmospheric physics? A sensitivity analysis of cyclone Xynthia**, submitted to *npj Climate and Atmospheric Science*. The scripts provided shows how to compute the sensitivities (or gradients of the target metric of interest with respect to the atmospheric variables at initial time) with the Spherical Fourier Neural Operator (SFNO) AI model, and generate perturbation fields based on them. Scripts to download the initial condition fields from the European Centre for Medium-Range Weather Forecasts Reanalysis version 5 (ERA5), are also provided. The resulting simulations and sensitivities can be found in a companion data repository to this publication [2]. The repository is structured in two folders:

* scripts --> Main executable scripts.
  * scripts-computations: A folder containing Python scripts to compute the sensitivities and produce the AI-simulated runs.
    * `gradients.py`: Python script that computes the sensitivity fields at different lead times for the kinetic energy at the Bay of Biscay, using the SFNO AI-model. Note that this script can be easily modified to suit other variable/s, lead time/s and domain/s of interest.
    * `generate-sensitivity-perturbations.py`: Python script that scales the sensitivity fields up to values compatible to estimates of initial condition uncertainty following the methodology described in [3]. The resulting fields are the perturbation fields which are later used to perturb the initial condition in the perturbed simulations.
    * `inference.py`: Python script that simulates the evolution of cyclone Xynthia based on the control and perturbed intitial conditions.
  * `scripts-download-era5`: A folder containing Python scripts to download ERA5 pressure and surface variables from the Climate Data Store (CDS). 
  * `scripts-figures`: A folder containing Python scripts that allow one to reproduce the figures of the manuscript.
* utils --> This folder contains the auxiliary scripts that are sourced by the main scripts during execution.

The SFNO model and related statistics (means and standard deviations fields) can be downloaded from the European Centre for Medium-Range Weather Forecasts (ECMWF) library for AI-models, namely `ecmwf-ai`. Instructions on the installation of this library can be found at the following Github repository: https://github.com/ecmwf-lab/ai-models. 

## References
[1] Bonev, B., Kurth, T., Hundt, C., Pathak, J., Baust, M., Kashinath, K., & Anandkumar, A. (2023, July). Spherical fourier neural operators: Learning stable dynamics on the sphere. In International conference on machine learning (pp. 2806-2823). PMLR.

[2] 

[3] Doyle, J. D., Amerault, C., Reynolds, C. A., & Reinecke, P. A. (2014). Initial condition sensitivity and predictability of a severe extratropical cyclone using a moist adjoint. Monthly Weather Review, 142(1), 320-342.