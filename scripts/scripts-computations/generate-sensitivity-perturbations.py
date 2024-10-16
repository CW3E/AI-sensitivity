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
################################################################################
date_peak=np.datetime64('2010-02-28T00:00:00')
lead_time=36
longitudes=list(np.arange(354,359.75+0.25,0.25))
longitudes.append(0)
latitudes=np.arange(43,48+0.25,0.25)
################################################################################
# Device
torch.backends.cudnn.benchmark=True
device=torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
##########################################
print('Lead time: '+str(lead_time))
date_ic=date_peak-np.timedelta64(lead_time, 'h')
##########################################
# Load data
data_ic=get_date_ic_sfno(date_ic=date_ic, path_data=workdir+'data/era5/', vars=sfno_vars)
data_peak=get_date_ic_sfno(date_ic=date_peak, path_data=workdir+'data/era5/', vars=sfno_vars).assign_coords({'time': data_ic.time})
w=1/((data_peak-data_ic)**2)
# Load gradients
gradients=xr.open_dataset(workdir+'data/'+modelName+'/gradients/gradients-lt'+str(lead_time)+'.nc')
# Generate perturbations
s=1/2.5
perturbations=s/w*gradients
# Save perturbations
perturbations.to_netcdf(workdir+'data/'+modelName+'/perturbations/perturbations-lt'+str(lead_time)+'.nc')
#############################################
# s=1/2.5
# perturbations=s/w*gradients
# for var in list(perturbations.keys()):
#     print(var+': '+str(perturbations[var].values.max()))
