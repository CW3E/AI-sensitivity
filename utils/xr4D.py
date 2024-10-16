def xr4D(grid, levels, pressure_vars, surface_vars):
    grid_pressure=[]
    for pressure_var in pressure_vars:
        grid_var=[]
        for level in levels:
            var=pressure_var+str(level)
            grid_level=grid[var]
            grid_level=grid_level.assign_coords({'isobaric':level})
            grid_var.append(grid_level)
        grid_var=xr.concat(grid_var, dim='isobaric')
        grid_pressure.append(grid_var)
    grid_pressure=xr.merge(grid_pressure)
    grid_pressure=grid_pressure.rename({'ta50':'ta','z50':'z','ua50':'ua','va50':'va','hur50':'hur'})
    grid_surface=grid[surface_vars]
    grid=xr.merge([grid_surface,grid_pressure])
    return grid
