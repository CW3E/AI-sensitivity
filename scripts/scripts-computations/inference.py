################ LICENSE ######################################
# This software is Copyright © 2024 The Regents of the University of California.
# All Rights Reserved. Permission to copy, modify, and distribute this software and its documentation
# for educational, research and non-profit purposes, without fee, and without a written agreement is
# hereby granted, provided that the above copyright notice, this paragraph and the following three paragraphs
# appear in all copies. Permission to make commercial use of this software may be obtained by contacting:
#
# Office of Innovation and Commercialization 9500 Gilman Drive, Mail Code 0910 University of California La Jolla, CA 92093-0910 innovation@ucsd.edu
# This software program and documentation are copyrighted by The Regents of the University of California. The software program and documentation are
# supplied “as is”, without any accompanying services from The Regents. The Regents does not warrant that the operation of the program will
# be uninterrupted or error-free. The end-user understands that the program was developed for research purposes and is advised not to rely exclusively on the program for any reason.
#
# IN NO EVENT SHALL THE UNIVERSITY OF CALIFORNIA BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT, SPECIAL,
# INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE
# AND ITS DOCUMENTATION, EVEN IF THE UNIVERSITY OF CALIFORNIA HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE. THE UNIVERSITY OF CALIFORNIA SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER
# IS ON AN “AS IS” BASIS, AND THE UNIVERSITY OF CALIFORNIA HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE, SUPPORT,
# UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
################################################################
#################### Import libraries ##########################
## Import libraries
import os
import numpy as np
import xarray as xr
import dask as da
import pandas as pd
import torch 
import gc
#################### Define working directory ##################
## Set the working directory
workdir='/expanse/nfs/cw3e/cwp167/projects/test-adjoint/github/test/'
os.chdir(workdir)
# Scripts available at: https://github.com/ecmwf-lab/ai-models-fourcastnetv2/blob/main/ai_models_fourcastnetv2/fourcastnetv2/sfnonet.py
exec(open(workdir+'/utils-do-not-upload-to-github/sfnonet.py').read())
exec(open(workdir+'/utils-do-not-upload-to-github/load_sfno.py').read())
for file in os.listdir(workdir+'/utils/'):
    if file.endswith('.py') and file.strip('._')==file:
        exec(open(workdir+'/utils/'+file).read())
#################### Define setup parameters ##################
modelName='sfno'
sfno_vars=["uas","vas","u100","v100","tas","sp","mslp","tcwv",
           "ua50","ua100","ua150","ua200","ua250","ua300","ua400","ua500","ua600","ua700","ua850","ua925","ua1000",
           "va50","va100","va150","va200","va250","va300","va400","va500","va600","va700","va850","va925","va1000",
           "z50","z100","z150","z200","z250","z300","z400","z500","z600","z700","z850","z925","z1000",
           "ta50","ta100","ta150","ta200","ta250","ta300","ta400","ta500","ta600","ta700","ta850","ta925","ta1000",
           "hur50","hur100","hur150","hur200","hur250","hur300","hur400","hur500","hur600","hur700","hur850","hur925","hur1000"]
date_peak=np.datetime64('2010-02-28T00:00:00')
lead_time=36
################################################################################
# Device
torch.backends.cudnn.benchmark=True
device=torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
##########################################
# Neural weather model
# Download the .tar file from the ECMWF ai-models Github repository: https://github.com/ecmwf-lab/ai-models
# Create a folder called "models" in your working directory and store the .tar file here 
# The load_sfno function is available at as "load_model":   
model=load_sfno(workdir+'/models/weights.tar', device) 
##########################################
# Type of forecast? Control forecast or perturbed forecast
forecast_type='perturbed' # Options are: "control" or "perturbed"
perturbation_type='neg' # Options are: "pos" or "neg". Only applies to perturbed forecasts.
##########################################
##### Perform inference #####
date_ic=date_peak-np.timedelta64(lead_time, 'h')
##########################################
# Initial condition
data_ic=get_date_ic_sfno(date_ic=date_ic, 
                            path_data=workdir+'/data/era5/', 
                            vars=sfno_vars)
data_ic_raw=data_ic
# Load the objects "data_mean" and "data_std", which will be used for scaling purposes
# Download the .npy files (means and standard deviations) from the ECMWF ai-models Github repository: https://github.com/ecmwf-lab/ai-models
# Create a folder called "stats" in your working directory and store the .npy files here 
grid_delete, data_mean, data_std=scaleGrid_sfno(data_ic, 
                                                path_data_mean=workdir+'/stats/global_means.npy', 
                                                path_data_std=workdir+'/stats/global_stds.npy',
                                                return_params=True)
##########################################
# Add perturbation?
predName='control-forecast'
# Add perturbation?
if forecast_type=="perturbed":
    perturbation_field=xr.open_dataset(workdir+'data/'+modelName+'/perturbations/perturbations-lt'+str(lead_time)+'.nc')
    if perturbation_type=='pos':
        data_ic=perturb_input_vars(grid=data_ic,
                                   perturbation=perturbation_field,
                                   vars_set=sfno_vars,
                                   along=sfno_vars)
    else:
        data_ic=perturb_input_vars(grid=data_ic,
                                   perturbation=perturbation_field*(-1),
                                   vars_set=sfno_vars,
                                   along=sfno_vars)
    predName='perturbed-forecast-'+perturbation_type
##########################################
# Scale grid
data_ic=scaleGrid_sfno(data_ic, 
                        path_data_mean=workdir+'/stats/global_means.npy', 
                        path_data_std=workdir+'/stats/global_stds.npy',
                        return_params=False)
# Recursive N-step forecast
iters=int(lead_time/6)
pred_list=[]
for iter in range(iters):
    # Scale data
    if iter!=0:
        data_ic=scaleGrid_sfno(data_ic, 
                                path_data_mean=workdir+'/stats/global_means.npy', 
                                path_data_std=workdir+'/stats/global_stds.npy')
    ##########################################
    # Predict with neural network
    data_ic=predictNWM(grid=data_ic, vars=sfno_vars, device=device, data_mean=data_mean, data_std=data_std)
    pred_list.append(data_ic)
##########################################
# Concatenate prediction
pred_iter=xr.concat(pred_list, dim='time')
##########################################
# Fill out metadata and add initial condition
pred=xr.concat([data_ic_raw, pred_iter], dim='time')
##########################################
# Save prediction
pred.to_netcdf(workdir+'data/'+modelName+'/predictions/'+predName+'-lt'+str(lead_time)+'.nc')
