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
import metpy
from metpy.units import units
def hydrostatic_balance(grid, vars, levels, estimated_vars):
    # Hydrostatic balance
    vars_z=[]
    for ind_level, level in enumerate(levels):
        # print(level)
        var_k='z'+level
        if level!="1000":
            var_k='z'+level
            var_k0='z'+level_prev
            term_0=new_field_xr[var_k0]
            term_ln=np.log(int(level))-np.log(int(level_prev))
            R=287 
            w_k=metpy.calc.mixing_ratio_from_relative_humidity(int(level)*units.hPa, 
                                                                grid['ta'+level]*units.K, 
                                                                grid['hur'+level]/100)
            T_k=metpy.calc.virtual_temperature(grid['ta'+level]*units.K, w_k)
            w_k0=metpy.calc.mixing_ratio_from_relative_humidity(int(level_prev)*units.hPa, 
                                                                grid['ta'+level_prev]*units.K, 
                                                                grid['hur'+level_prev]/100)                                             
            T_k0=metpy.calc.virtual_temperature(grid['ta'+level_prev]*units.K, w_k0)
            term_T=T_k+T_k0
            new_field=term_0.values-term_ln*R*term_T.values/2
            new_field_xr=xr.Dataset(data_vars={var_k: (new_field_xr[var_k0].dims, new_field)},
                                    coords=new_field_xr[var_k0].coords)
            level_prev=level
            vars_z.append(new_field_xr)
        else:
            new_field_xr=grid
            level_prev=level
            vars_z.append(new_field_xr[var_k])
    vars_z=xr.merge(vars_z)
    # Replace the original variables to the adjusted  Variables
    data=[]
    for v in vars:
        # print(v)
        if v in estimated_vars:
            data.append(vars_z[v])
        else:
            data.append(grid[v])
    return xr.merge(data)




