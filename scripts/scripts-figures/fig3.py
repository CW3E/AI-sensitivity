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
# Cross-section, info here: https://unidata.github.io/MetPy/latest/examples/cross_section.html
import os
import numpy as np
import xarray as xr
import dask as da
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors
import metpy.calc as mpcalc
from metpy.cbook import get_test_data
from metpy.interpolate import cross_section
import gc
import matplotlib.ticker as ticker
#################### Define setup parameters ##################
## Set the working directory
workdir='/expanse/nfs/cw3e/cwp167/projects/test-adjoint/github/test/'
os.chdir(workdir)
for file in os.listdir(workdir+'/utils/'):
    if file.endswith('.py') and file.strip('._')==file:
        exec(open(workdir+'/utils/'+file).read())
############
def fmt(x, pos):
    a, b = '{:.0e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)
###############################################################
modelName='sfno'
lead_time=36
date_peak=np.datetime64('2010-02-28T00:00:00')
lons=np.arange(-60,20+0.25,0.25)
lats=np.arange(20,80+0.25,0.25)
sfno_vars=["uas","vas","u100","v100","tas","sp","mslp","tcwv",
           "ua50","ua100","ua150","ua200","ua250","ua300","ua400","ua500","ua600","ua700","ua850","ua925","ua1000",
           "va50","va100","va150","va200","va250","va300","va400","va500","va600","va700","va850","va925","va1000",
           "z50","z100","z150","z200","z250","z300","z400","z500","z600","z700","z850","z925","z1000",
           "ta50","ta100","ta150","ta200","ta250","ta300","ta400","ta500","ta600","ta700","ta850","ta925","ta1000",
           "hur50","hur100","hur150","hur200","hur250","hur300","hur400","hur500","hur600","hur700","hur850","hur925","hur1000"]
levels=[50,100,150,200,250,300,400,500,600,700,850,925,1000]
start=(55,-38)
end=(22,-10)
start=(55,-38)
end=(22,-15)
start=(45,-38)
end=(22,-10)
###########################################################################################
## Load adjoint fields
grid_gradients=xr.open_dataset(workdir+'data/sfno/gradients/gradients-standardized-lt'+str(lead_time)+'.nc').isel(time=0)
grid_gradients=changeLongitudeProjection(grid_gradients).sel(longitude=lons, latitude=lats)
grid_gradients=xr4D(grid_gradients, levels=levels,
                    pressure_vars=["ta","z","ua","va","hur"],
                    surface_vars=["uas","vas","u100","v100","tas","sp","mslp","tcwv"])
grid_gradients=grid_gradients.metpy.parse_cf()
## Compute the cross-section from point A to A'
cross=cross_section(grid_gradients, start, end).set_coords(('latitude', 'longitude'))
## Load era5
data_era5=date_peak-np.timedelta64(lead_time, 'h')
data_era5=get_date_ic_sfno(date_ic=data_era5, 
                           path_data=workdir+'data/era5/', 
                           vars=sfno_vars)
data_era5=changeLongitudeProjection(data_era5).sel(longitude=lons, latitude=lats).isel(time=0)
data_era5=xr4D(data_era5, levels=levels,
             pressure_vars=["ta","z","ua","va","hur"],
             surface_vars=["uas","vas","u100","v100","tas","sp","mslp","tcwv"])
data_era5=data_era5.metpy.parse_cf()
## Compute the cross-section from point A to A'
cross_era5=cross_section(data_era5, start, end).set_coords(('latitude', 'longitude'))
## Compute the potential temperature
cross_era5['isobaric'].attrs={'units':'hPa'}
cross_era5['ta'].attrs={'units':'K'}
cross_era5['Potential_temperature']=mpcalc.potential_temperature(
    cross_era5['isobaric'],
    cross_era5['ta'])
###########################################################################################
## Define the figure object and primary axes
fig=plt.figure(1)
ax=plt.axes()
## Adjoint of relative humidity (contourf)
cmap=matplotlib.colors.LinearSegmentedColormap.from_list("", ["purple","blue","turquoise","white","white","yellow","red","pink"])
rh_contour=ax.contourf(cross['longitude'], cross['isobaric'], cross['hur'],
                       levels=np.linspace(-0.0005, 0.0005, 21), cmap=cmap, extend="both")
rh_colorbar=fig.colorbar(rh_contour,format=ticker.FuncFormatter(fmt))
# ## Potential temperature (contour)
theta_contour=ax.contour(cross_era5['longitude'], cross_era5['isobaric'], cross_era5['Potential_temperature'],
                         levels=np.arange(256, 600, 4), colors='k', linewidths=1)
theta_contour.clabel(theta_contour.levels[1::2], fontsize=8, colors='k', inline=1,
                     inline_spacing=8, fmt='%i', rightside_up=True, use_clabeltext=True)
## Adjust the y-axis to be logarithmic
ax.set_yscale('symlog')
ax.set_yticklabels(np.arange(1000, 50, -100))
ax.set_ylim(cross['isobaric'].max(), cross['isobaric'].min())
ax.set_yticks(np.arange(1000, 50, -100))
## Plot the inner map
data_proj=data_era5['z'].metpy.cartopy_crs
ax_inset=fig.add_axes([0.155, 0.665, 0.20, 0.20], projection=data_proj)
ax_inset.contour(data_era5['longitude'], data_era5['latitude'], data_era5['z'].sel(isobaric=500)/9.81,
                 levels=np.arange(4500, 6500, 100), cmap='inferno')
# Plot the path of the cross section
endpoints=data_proj.transform_points(ccrs.Geodetic(),
                                    *np.vstack([start, end]).transpose()[::-1])
ax_inset.scatter(endpoints[:, 0], endpoints[:, 1], c='k', zorder=2)
ax_inset.plot(cross_era5['longitude'], cross_era5['latitude'], c='k', zorder=2)
# Add geographic features
ax_inset.coastlines()
# ## Set the titles and axes labels
ax_inset.set_title('')
# ax.set_title(f'NARR Cross-Section \u2013 {start} to {end} \u2013 '
#              f'Valid: {cross["time"].dt.strftime("%Y-%m-%d %H:%MZ").item()}\n'
#              'Potential Temperature (K), Tangential/Normal Winds (knots), Relative Humidity '
#              '(dimensionless)\nInset: Cross-Section Path and 500 hPa Geopotential Height')
ax.set_ylabel('Pressure (hPa)')
ax.set_xlabel('Longitude (degrees east)')
rh_colorbar.set_label('∂KE/∂RH')
###
plt.savefig(workdir+'figures/fig3.jpg', bbox_inches='tight')
plt.savefig(workdir+'figures/fig3.pdf', bbox_inches='tight')
plt.close()
