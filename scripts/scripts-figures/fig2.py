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
vars=['tcwv','ta700','va700','z700','z700','z700']
leadtimes=[36,36,36,48,36,24]
variable_names=['Total column water vapour','Air temperature at 700 hPa','Meridional wind at 700 hPa',
                'Geopotential at 700 hPa (Lead time: 24 hours)','Geopotential at 700 hPa (Lead time: 36 hours)','Geopotential at 700 hPa (Lead time: 48 hours)']
lons=np.arange(-60,20+0.25,0.25)
lats=np.arange(20,80+0.25,0.25)
lons_inner=np.arange(-6,0+0.25,0.25)
lats_inner=np.arange(43,48+0.25,0.25)
text_lat=70
text_lon=-70
#######
fig, ax=plt.subplots(3, 3, figsize=(15,15), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)})
# Loop over variables
ind_x=0
ind_y=-1
for ind_var, var in enumerate(vars):
    ind_y=ind_y+1
    if ind_y==3:
        ind_x=ind_x+1
        ind_y=0
    print(str(ind_x)+'---'+str(ind_y))
    leadtime=leadtimes[ind_var]
    date_peak=np.datetime64('2010-02-28T00:00:00')
    date_ic=date_peak-np.timedelta64(leadtime, 'h')
    print('Lead time: '+str(leadtime)+' --- Variable: '+var)
    # Load grid
    grid_gradients=xr.open_dataset(workdir+'data/sfno/gradients/gradients-lt'+str(leadtime)+'.nc').isel(time=0)
    grid_gradients=changeLongitudeProjection(grid_gradients).sel(longitude=lons, latitude=lats)
    # np.sum(np.abs(grid_gradients[var].values))
    #######
    if var=='tcwv' or var=='ta700':
        ticks=[-0.005,0,0.005]
        levels=np.linspace(-0.005, 0.005, 21)
    elif var=='va700':
        ticks=[-0.002,0,0.002]
        levels=np.linspace(-0.002, 0.002, 21)
    else:
        ticks=[-0.0002,0,0.0002]
        levels=np.linspace(-0.0002, 0.0002, 21)
    cmap=matplotlib.colors.LinearSegmentedColormap.from_list("", ["purple","blue","turquoise","white","white","yellow","red","pink"])
    fig_=ax[ind_x,ind_y].contourf(grid_gradients.longitude,
                                           grid_gradients.latitude,
                                           grid_gradients[var],
                                           levels=levels,
                                           cmap=cmap,
                                           extend='both',
                                           linewidths=0.5,
                                           transform=ccrs.PlateCarree())
    cbar=fig.colorbar(fig_, ax=ax[ind_x,ind_y], ticks=ticks, fraction=0.035, pad=0.045, orientation="horizontal")
    cbar.set_label('Gradient (dKE/dXi)', rotation=0, labelpad=5, fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    if var=='z700':
        date_contour=date_peak-np.timedelta64(leadtime, 'h')
        grid_era5=xr.open_dataset(workdir+'data/era5/z700.nc').sel(time=date_contour)
        grid_era5=changeLongitudeProjection(grid_era5).sel(longitude=lons, latitude=lats)
        ###
        grid_z700=grid_era5['z700']/9.8
        f=grid_z700.plot.contour(kwargs=dict(inline=True),
                                 levels=np.arange(2800, 3200, 50),
                                 ax=ax[ind_x,ind_y],
                                 colors=['gray'],
                                 extend='both',
                                 linewidths=1,
                                 transform=ccrs.PlateCarree())
        ax[ind_x,ind_y].clabel(f, levels=[2800,3050,3100], fontsize=9)
    if var=='ua700' or var=='va700':
        grid_ua=xr.open_dataset(workdir+'data/era5/ua700.nc').sel(time=date_ic)
        grid_ua=changeLongitudeProjection(grid_ua).sel(longitude=lons, latitude=lats)
        grid_va=xr.open_dataset(workdir+'data/era5/va700.nc').sel(time=date_ic)
        grid_va=changeLongitudeProjection(grid_va).sel(longitude=lons, latitude=lats)
        grid_era5=xr.merge((grid_ua,grid_va))
        ###
        grid_quiver=grid_era5.isel(longitude=slice(None, None, 16),latitude=slice(None, None, 16))
        quiver=grid_quiver.plot.quiver(x='longitude',
                                       y='latitude',
                                       u='ua700',
                                       v='va700',
                                       ax=ax[ind_x,ind_y],
                                       transform=ccrs.PlateCarree())
        veclenght=10
        maxstr='%3.1f m/s'%veclenght
        plt.quiverkey(quiver,0.1,0.1,veclenght, maxstr, labelpos='S', coordinates='axes').set_zorder(11)
    if var=='tcwv':
        ###
        grid_era5=xr.open_dataset(workdir+'data/era5/tcwv.nc').sel(time=date_ic)
        grid_era5=changeLongitudeProjection(grid_era5).sel(longitude=lons, latitude=lats)
        ###
        grid_tcwv=grid_era5['tcwv']
        f=grid_tcwv.plot.contour(kwargs=dict(inline=True),
                                 levels=np.arange(10, 30, 4),
                                 ax=ax[ind_x,ind_y],
                                 colors='gray',
                                 extend='both',
                                 linewidths=1,
                                 transform=ccrs.PlateCarree())
        ax[ind_x,ind_y].clabel(f, levels=[10,14,18,22,26], fontsize=9)
    if var=='ta700':
        ###
        grid_era5=xr.open_dataset(workdir+'data/era5/ta700.nc').sel(time=date_ic)
        grid_era5=changeLongitudeProjection(grid_era5).sel(longitude=lons, latitude=lats)
        ###
        grid_tcwv=grid_era5['ta700']
        f=grid_tcwv.plot.contour(kwargs=dict(inline=True),
                                 levels=np.arange(260, 320, 4),
                                 ax=ax[ind_x,ind_y],
                                 colors='gray',
                                 extend='both',
                                 linewidths=1,
                                 transform=ccrs.PlateCarree())
        ax[ind_x,ind_y].clabel(f, levels=[260,272,284,292,312], fontsize=9)
    # Metadata of the plot
    ax[ind_x,ind_y].coastlines(linewidth=2)
    ax[ind_x,ind_y].plot([lons_inner[0],lons_inner[-1]],[lats_inner[0],lats_inner[0]], color='green', linewidth=3)
    ax[ind_x,ind_y].plot([lons_inner[0],lons_inner[-1]],[lats_inner[-1],lats_inner[-1]], color='green', linewidth=3)
    ax[ind_x,ind_y].plot([lons_inner[0],lons_inner[0]],[lats_inner[0],lats_inner[-1]], color='green', linewidth=3)
    ax[ind_x,ind_y].plot([lons_inner[-1],lons_inner[-1]],[lats_inner[0],lats_inner[-1]], color='green', linewidth=3)
    ax[ind_x,ind_y].set_title(variable_names[ind_var])
###
plt.savefig(workdir+'figures/fig2.jpg', bbox_inches='tight')
plt.savefig(workdir+'figures/fig2.pdf', bbox_inches='tight')
plt.close()
