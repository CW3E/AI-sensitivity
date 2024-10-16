def perturb_input_vars(grid,perturbation,vars_set,along):
    ind=[vars_set.index(v) for v in along]
    grid_array=get_input_array(grid)
    if isinstance(perturbation, (int, float, complex)):
        grid_array[:,ind,:,:]=grid_array[:,ind,:,:]*perturbation
    else:
        perturbation_array=get_input_array(perturbation)
        grid_array[:,ind,:,:]=grid_array[:,ind,:,:]+perturbation_array[:,ind,:,:]
    grid=xr.Dataset(data_vars={var: (grid['tcwv'].dims, grid_array[:,ind_nv,:,:]) for ind_nv, var in enumerate(vars_set)}, coords=grid.coords)
    return grid
