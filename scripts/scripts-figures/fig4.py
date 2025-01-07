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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors
import gc
from matplotlib import ticker
#################### Define setup parameters ##################
## Set the working directory
workdir='/expanse/nfs/cw3e/cwp167/projects/test-adjoint/github/test/'
os.chdir(workdir)
for file in os.listdir(workdir+'/utils/'):
    if file.endswith('.py') and file.strip('._')==file:
        exec(open(workdir+'/utils/'+file).read())
###############################################################
dates=[np.datetime64('2010-02-27T00:00:00'),np.datetime64('2010-02-27T12:00:00'),np.datetime64('2010-02-28T00:00:00')]
lead_times=[12,24,36]
modelName='sfno'
lead_time=36
lons=np.arange(-30,4+0.25,0.25)
lats=np.arange(25,55+0.25,0.25)
forecast_type='perturbed-forecast'
types=['pos','neg']
#############
vars_era5=['uas','vas']
grid=[]
for var in vars_era5:
    grid.append(xr.open_dataset(workdir+'data/era5/'+var+'.nc'))
grid=xr.merge(grid)
grid=changeLongitudeProjection(grid).sel(longitude=lons, latitude=lats)
#######
fig, ax=plt.subplots(2, 3, figsize=(15,10), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)})
# Loop over dates
for ind_exp, type in enumerate(types):
    i=ind_exp
    for ind_date, date in enumerate(dates):
        j=ind_date
        # Load control forecast
        pred_control=xr.open_dataset(workdir+'data/'+modelName+'/predictions/control-forecast-lt'+str(lead_time)+'.nc').sel(time=date)
        pred_control=0.5*((pred_control['uas']**2)+(pred_control['vas']**2))
        # Load perturbed forecast
        predName=forecast_type+'-'+type
        pred_exp=xr.open_dataset(workdir+'data/'+modelName+'/predictions/'+predName+'-lt'+str(lead_time)+'.nc').sel(time=date)
        pred_exp=0.5*((pred_exp['uas']**2)+(pred_exp['vas']**2))
        # Compute the evolved perturbations at each forecast step (perturbed-control)
        evolved_perturbation=pred_exp-pred_control
        evolved_perturbation=changeLongitudeProjection(evolved_perturbation).sel(longitude=lons, latitude=lats)
        #######
        # Display the evolved perturbation fields
        levels=np.linspace(-50, 50, 21)
        ticks=[-50,-25,0,25,50]
        cmap=matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue","turquoise","mediumaquamarine","white","white","yellow","red","pink"])
        fig_=ax[i,j].contourf(evolved_perturbation.longitude,
                              evolved_perturbation.latitude,
                              evolved_perturbation,
                              levels=levels,
                              cmap=cmap,
                              extend='both',
                              linewidths=0.5,
                              transform=ccrs.PlateCarree())
        cbar=fig.colorbar(fig_, ax=ax[i,j], ticks=ticks, fraction=0.035, pad=0.045, orientation="horizontal")
        cbar.set_label('ΔKE (m²/s²)', rotation=0, labelpad=5, fontsize=10)
        cbar.ax.tick_params(labelsize=8)
        #######
        # Display the wind vector field
        grid_date=grid.sel(time=date)
        grid_quiver=grid_date.isel(longitude=slice(None, None, 8),latitude=slice(None, None, 8))
        quiver=grid_quiver.plot.quiver(x='longitude',
                                       y='latitude',
                                       u='uas',
                                       v='vas',
                                       ax=ax[i,j],
                                       transform=ccrs.PlateCarree())
        veclenght=10
        maxstr='%3.1f m/s'%veclenght
        plt.quiverkey(quiver,0.1,0.1,veclenght, maxstr, labelpos='S', coordinates='axes').set_zorder(11)
        #######
        # Metadata of the plot
        ax[i,j].coastlines(linewidth=2)
        ax[i,j].set_title('Perturbation at '+str(lead_times[ind_date])+'h of lead time')
        if j==0:
            ax[i,j].text(-32,35, types[ind_exp], rotation='vertical')
###
plt.savefig(workdir+'figures/fig4.jpg', bbox_inches='tight')
plt.savefig(workdir+'figures/fig4.pdf', bbox_inches='tight')
plt.close()
