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
vars=['mslp','uas','vas']
datasets=['era5','sfno']
date_end=np.datetime64('2010-03-02T12:00:00')
dates=[np.datetime64('2010-02-26T12:00:00'),np.datetime64('2010-02-27T12:00:00'),np.datetime64('2010-02-28T00:00:00')]
year=str(date_end)[:4]
lons=np.arange(-60,20+0.25,0.25)
lats=np.arange(20,80+0.25,0.25)
lons_inner=np.arange(-6,0+0.25,0.25)
lats_inner=np.arange(43,48+0.25,0.25)
text_lat=70
text_lon=-70
#######
fig, ax=plt.subplots(len(datasets), len(dates), figsize=(15,10), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=0)})
# Loop over dates
for ind_date, date in enumerate(dates):
    # Loop over datasets
    for ind_dataset, dataset in enumerate(datasets):
        print('Dataset: '+dataset+' --- Date: '+str(date))
        # Load grid
        if dataset=='era5':
            grid=[]
            for var in ['mslp','uas','vas']:
                grid.append(xr.open_dataset(workdir+'data/era5/'+var+'.nc'))
            grid=xr.merge(grid)
        elif dataset=='sfno':
            grid=xr.open_dataset(workdir+'data/'+dataset+'/predictions/control-forecast-lt36.nc')
        grid=changeLongitudeProjection(grid).sel(longitude=lons, latitude=lats)
        grid_date=grid.sel(time=date)
        if not (ind_dataset==1 and ind_date==0):
            # Wind (mesh)
            cmap=matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","white","blue","turquoise","yellow","orange","red","pink"])
            grid_wind=0.5*((grid_date['uas']**2)+(grid_date['vas']**2))
            fig_=ax[ind_dataset,ind_date].contourf(grid_wind.longitude,
                                                   grid_wind.latitude,
                                                   grid_wind,
                                                   levels=np.arange(0, 250, 25),
                                                   cmap=cmap,
                                                   extend='both',
                                                   linewidths=0.5,
                                                   transform=ccrs.PlateCarree())
            cbar=fig.colorbar(fig_, ax=ax[ind_dataset,ind_date], fraction=0.035, pad=0.045,
                               orientation="horizontal")
            cbar.set_label('KE (m²/s²)', rotation=0,
                           labelpad=5, fontsize=10)
            cbar.ax.tick_params(labelsize=8)
            # Mean sea level pressure (countour)
            grid_mslp=grid_date['mslp']/100
            f=grid_mslp.plot.contour(kwargs=dict(inline=True),
                                             levels=np.arange(980, 1026, 4),
                                             ax=ax[ind_dataset,ind_date],
                                             colors=['gray'],
                                             extend='both',
                                             linewidths=1.2,
                                             transform=ccrs.PlateCarree())
            # ax[ind_dataset,ind_date].clabel(f, levels=[1008,1024], fontsize=6)
            # Wind (vector)
            grid_quiver=grid_date.isel(longitude=slice(None, None, 16),latitude=slice(None, None, 16))
            quiver=grid_quiver.plot.quiver(x='longitude',
                                           y='latitude',
                                           u='uas',
                                           v='vas',
                                           ax=ax[ind_dataset,ind_date],
                                           transform=ccrs.PlateCarree())
            veclenght=10
            maxstr='%3.1f m/s'%veclenght
            plt.quiverkey(quiver,0.1,0.1,veclenght, maxstr, labelpos='S', coordinates='axes').set_zorder(11)
            #######
            # Metadata of the plot
            ax[ind_dataset,ind_date].coastlines(linewidth=2)
            ax[ind_dataset,ind_date].set_title(str(abs(np.array(date-dates[-1]))/3600).strip(" seconds")+' hours prior to the event')
            ax[ind_dataset,ind_date].plot([lons_inner[0],lons_inner[-1]],[lats_inner[0],lats_inner[0]], color='pink', linewidth=1.5)
            ax[ind_dataset,ind_date].plot([lons_inner[0],lons_inner[-1]],[lats_inner[-1],lats_inner[-1]], color='pink', linewidth=1.5)
            ax[ind_dataset,ind_date].plot([lons_inner[0],lons_inner[0]],[lats_inner[0],lats_inner[-1]], color='pink', linewidth=1.5)
            ax[ind_dataset,ind_date].plot([lons_inner[-1],lons_inner[-1]],[lats_inner[0],lats_inner[-1]], color='pink', linewidth=1.5)
            if ind_date==0:
                ax[ind_dataset,ind_date].text(text_lon+5,45, dataset, rotation='vertical')
#######
plt.savefig(workdir+'figures/fig1.jpg', bbox_inches='tight')
plt.savefig(workdir+'figures/fig1.pdf', bbox_inches='tight')
plt.close()
###############################################################
