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
#################### Define setup parameters ##################
## Set the working directory
workdir='/expanse/nfs/cw3e/cwp167/projects/test-adjoint/github/test/'
os.chdir(workdir)
for file in os.listdir(workdir+'/utils/'):
    if file.endswith('.py') and file.strip('._')==file:
        exec(open(workdir+'/utils/'+file).read())
###############################################################
date_peak=np.datetime64('2010-02-28T00:00:00')
modelName='sfno'
lead_time=36
lons=np.arange(-16,5+0.25,0.25)
lats=np.arange(38,55+0.25,0.25)
experiments=['Control','Positive','Negative','Perturbation-Positive','Perturbation-Negative']
forecast_type='perturbed-forecast'
#######
fig, ax=plt.subplots(2, 3, figsize=(12,8), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)})
i=0
j=-1
# Loop over experiments
for ind_exp, exp in enumerate(experiments):
    j=j+1
    if j==3:
        i=i+1
        j=1
    # Load control prediction
    pred_control=xr.open_dataset(workdir+'data/'+modelName+'/predictions/control-forecast-lt'+str(lead_time)+'.nc').sel(time=date_peak)
    pred_control=changeLongitudeProjection(pred_control).sel(longitude=lons, latitude=lats)
    pred_control_z=pred_control['mslp']
    pred_control=0.5*((pred_control['uas']**2)+(pred_control['vas']**2))
    # Load perturbed prediction
    if exp=='Positive' or exp=='Perturbation-Positive':
        type='pos'
    if exp=='Negative' or exp=='Perturbation-Negative':
        type='neg'
    if exp!='Control':
        predName=forecast_type+'-'+type
        pred_exp=xr.open_dataset(workdir+'data/'+modelName+'/predictions/'+predName+'-lt'+str(lead_time)+'.nc').sel(time=date_peak)
        pred_exp=changeLongitudeProjection(pred_exp).sel(longitude=lons, latitude=lats)
        pred_exp_z=pred_exp['mslp']
        pred_exp=0.5*((pred_exp['uas']**2)+(pred_exp['vas']**2))
    # Plots
    if exp=='Control' or exp=='Positive' or exp=='Negative':
        if exp=='Control':
            grid_to_plot=pred_control
            grid_to_plot_z=pred_control_z
        else:
            grid_to_plot=pred_exp
            grid_to_plot_z=pred_exp_z
        # Plot wind
        cmap=matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","white","blue","turquoise","yellow","orange","red","pink"])
        fig_=ax[i,j].contourf(grid_to_plot.longitude,
                                grid_to_plot.latitude,
                                grid_to_plot,
                                levels=np.arange(0, 250, 25),
                                cmap=cmap,
                                extend='both',
                                linewidths=0.5,
                                transform=ccrs.PlateCarree())
        cbar=fig.colorbar(fig_, ax=ax[i,j], fraction=0.035, pad=0.045,
                            orientation="horizontal")
        cbar.set_label('KE (m²/s²)', rotation=0,
                        labelpad=5, fontsize=10)
        cbar.ax.tick_params(labelsize=8)
        ax[i,j].coastlines(linewidth=2)
        # Plot contour
        grid_mslp=grid_to_plot_z/100
        f=grid_mslp.plot.contour(kwargs=dict(inline=True),
                                            levels=np.arange(960, 1026, 4),
                                            ax=ax[i,j],
                                            colors=['gray'],
                                            extend='both',
                                            linewidths=1.2,
                                            transform=ccrs.PlateCarree())
        ax[i,j].clabel(f, levels=[960,976,992], fontsize=9)
    # Compute the evolved perturbations at lead time
    if exp=='Perturbation-Positive' or exp=='Perturbation-Negative':
        evolved_perturbation=pred_exp-pred_control
        evolved_perturbation_z=pred_exp_z-pred_control_z
        #######
        cmap=matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue","turquoise","mediumaquamarine","white","white","yellow","red","pink"])
        fig_=ax[i,j].contourf(evolved_perturbation.longitude,
                                evolved_perturbation.latitude,
                                evolved_perturbation,
                                levels=np.linspace(-50, 50, 21),
                                cmap=cmap,
                                extend='both',
                                linewidths=0.5,
                                transform=ccrs.PlateCarree())
        cbar=fig.colorbar(fig_, ax=ax[i,j], ticks=[-50,-25,0,25,50], fraction=0.035, pad=0.045, orientation="horizontal")
        cbar.set_label("ΔKE (m²/s²)", rotation=0, labelpad=5, fontsize=10)
        cbar.ax.tick_params(labelsize=8)
        # Metadata of the plot
        ax[i,j].coastlines(linewidth=2)
        ax[i,j].set_title(exp)
        #######
        evolved_perturbation_z=evolved_perturbation_z/100
        f=evolved_perturbation_z.plot.contour(kwargs=dict(inline=True),
                                                levels=np.arange(0, 2, 0.25),
                                                ax=ax[i,j],
                                                colors=['gray'],
                                                extend='both',
                                                linewidths=1,
                                                transform=ccrs.PlateCarree())
        ax[i,j].clabel(f, levels=[1,0.5], fontsize=9)
        f2=evolved_perturbation_z.plot.contour(kwargs=dict(inline=True),
                                                levels=np.arange(-2, 0, 0.25),
                                                ax=ax[i,j],
                                                colors=['gray'],
                                                extend='both',
                                                linewidths=1,
                                                linestyles='dashed',
                                                transform=ccrs.PlateCarree())
        ax[i,j].clabel(f2, levels=[-1,-0.5], fontsize=9)
###
plt.savefig(workdir+'figures/fig5.jpg', bbox_inches='tight')
plt.savefig(workdir+'figures/fig5.pdf', bbox_inches='tight')
plt.close()
