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
import torch.cuda.amp as amp
import gc
import wrf
# from ai_models.model import Model
#################### Define working directory ##################
## Set the working directory
workdir='/expanse/nfs/cw3e/cwp167/projects/test-adjoint/github/test/'
os.chdir(workdir)
exec(open(workdir+'/utils-do-not-upload-to-github/sfnonet.py').read())
exec(open(workdir+'/utils-do-not-upload-to-github/load_sfno.py').read())
for file in os.listdir(workdir+'/utils/'):
    if file.endswith('.py') and file.strip('._')==file:
            exec(open(workdir+'/utils/'+file).read())
#################### Define get_neuron ########################
def get_neuron(template, latitudes, longitudes):
    idx_lats=[]
    for lat in latitudes:
        idx_lats.append(np.where(template.latitude==lat)[0][0])
    idx_lons=[]
    for lon in longitudes:
        idx_lons.append(np.where(template.longitude==lon)[0][0])
    return idx_lats, idx_lons
#################### Define combinedModel #####################
class combinedModel(nn.Module):
  def __init__(self, model, forecast_step, idx_vars, idx_lats, idx_lons, mean, sigma):
    super(combinedModel, self).__init__()
    self.model=model
    self.forecast_step=forecast_step
    self.idx_vars=idx_vars
    self.idx_lats=idx_lats
    self.idx_lons=idx_lons
    self.mean=mean
    self.sigma=sigma
  def forward(self, x):
    with amp.autocast(True):
        x=(x-self.mean)/self.sigma
        for i in range(int(self.forecast_step)):
            x=self.model(x)
        x=x*self.sigma+self.mean
        x0=x[:,self.idx_vars,self.idx_lats[0]:self.idx_lats[-1],self.idx_lons[-1], None]
        x_west=x[:,self.idx_vars,self.idx_lats[0]:self.idx_lats[-1],self.idx_lons[0]:self.idx_lons[-2]]
        x_cynthia=torch.cat((x0,x_west), axis=3)
        ec_wind=(x_cynthia[:,0,:,:]**2+x_cynthia[:,1,:,:]**2)
        x_cynthia=0.5*torch.mean(ec_wind)
    return x_cynthia
#################### Define setup parameters ##################
modelName='sfno'
sfno_vars=["uas","vas","u100","v100","tas","sp","mslp","tcwv",
           "ua50","ua100","ua150","ua200","ua250","ua300","ua400","ua500","ua600","ua700","ua850","ua925","ua1000",
           "va50","va100","va150","va200","va250","va300","va400","va500","va600","va700","va850","va925","va1000",
           "z50","z100","z150","z200","z250","z300","z400","z500","z600","z700","z850","z925","z1000",
           "ta50","ta100","ta150","ta200","ta250","ta300","ta400","ta500","ta600","ta700","ta850","ta925","ta1000",
           "hur50","hur100","hur150","hur200","hur250","hur300","hur400","hur500","hur600","hur700","hur850","hur925","hur1000"]
################################################################################
date_peak=np.datetime64('2010-02-28T00:00:00')
lead_times=[24,36,48]
longitudes=list(np.arange(354,359.75+0.25,0.25))
longitudes.append(0)
latitudes=np.arange(43,48+0.25,0.25)
################################################################################
# Device
torch.backends.cudnn.benchmark=True
device=torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
##########################################
for lead_time in lead_times:
    print('Lead time: '+str(lead_time))
    date_ic=date_peak-np.timedelta64(lead_time, 'h')
    ##########################################
    # Initial condition
    data_ic=get_date_ic_sfno(date_ic=date_ic, 
                             path_data=workdir+'/data/era5/', 
                             vars=sfno_vars)
    # Scale data
    # Download the .npy files (means and standard deviations) from the ECMWF ai-models Github repository: https://github.com/ecmwf-lab/ai-models
    # Create a folder called "stats" in your working directory and store the .npy files here
    grid_delete, mean_xrr, sigma_xrr=scaleGrid_sfno(data_ic,
                                                    path_data_mean=workdir+'/stats/global_means.npy',
                                                    path_data_std=workdir+'/stats/global_stds.npy',
                                                    return_params=True)
    x=get_input_array(data_ic)
    mean_arr=np.array(mean_xrr.load().to_array())[None,:,:,:]
    sigma_arr=np.array(sigma_xrr.load().to_array())[None,:,:,:]
    x=torch.tensor(x).to(device, dtype=torch.float)
    x.requires_grad=True
    mean_arr=torch.tensor(mean_arr).to(device, dtype=torch.float)
    mean_arr.requires_grad=True
    sigma_arr=torch.tensor(sigma_arr).to(device, dtype=torch.float)
    sigma_arr.requires_grad=True
    ##########################################
    # Neural weather model
    # Download the .tar file from the ECMWF ai-models Github repository: https://github.com/ecmwf-lab/ai-models
    # Create a folder called "models" in your working directory and store the .tar file here 
    # The load_sfno function is available at as "load_model":   
    model_sfno=load_sfno(workdir+'/models/weights.tar', device) 
    # Get output neuron of interest
    idx_lats, idx_lons=get_neuron(template=data_ic,latitudes=latitudes,longitudes=longitudes)
    # Define new model comprising >1 forecast step
    # Surface winds have indices [0,1] in the variable list
    model=combinedModel(model_sfno, forecast_step=lead_time/6, idx_vars=[0,1], idx_lats=np.flip(idx_lats), idx_lons=idx_lons, mean=mean_arr, sigma=sigma_arr)
    # Set "evaluation" mode and send to device
    model=model.eval()
    model=model.to(device)
    ##########################################
    # Compute the sensitivities (matrix)
    g=torch.autograd.grad(outputs=model(x), inputs=x, retain_graph=True)[0]
    # Build xarray object
    g=array_to_xarray(g, sfno_vars, data_ic[sfno_vars[0]].dims, data_ic)
    # Save the sensitivities
    # g.to_netcdf(workdir+'data/'+modelName+'/gradients/rgradients-lt'+str(lead_time)+'.nc')
    ##########################################
    # Post-process the sensitivities (smooth the gradients with a convolutional filter)
    g_smooth=[]
    for var in sfno_vars:
        gs=wrf.smooth2d(g[var],passes=25)
        g_smooth.append(gs)
    g2=xr.merge(g_smooth)
    g2=[g2.rename({'smooth_'+var: var})[var] for var in sfno_vars]
    g2=xr.merge(g2)
    g2.to_netcdf(workdir+'data/'+modelName+'/gradients/gradients-lt'+str(lead_time)+'.nc')
    # Free memory
    del g, g2
    gc.collect()
    gc.collect()
#############################################
