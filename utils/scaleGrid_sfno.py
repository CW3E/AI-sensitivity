def scaleGrid_sfno(grid, path_data_mean, path_data_std, return_params=False):
    ## Load mean and std
    mean=np.load(path_data_mean)
    std=np.load(path_data_std)
    ## Put mean and std into xarray template
    mean_arr=np.ones((1,73,721,1440))
    std_arr=np.ones((1,73,721,1440))
    for v in range(73):
        mean_arr[:,v,:,:]=mean[:,v,:,:]
        std_arr[:,v,:,:]=std[:,v,:,:]
    mean_xrr=array_to_xarray(mean_arr,name_vars=list(grid.keys()),dims=grid['tcwv'].dims, template=grid).isel(time=0)
    std_xrr=array_to_xarray(std_arr,name_vars=list(grid.keys()),dims=grid['tcwv'].dims, template=grid).isel(time=0)
    ## Scale data
    if return_params is False:
        return (grid-mean_xrr)/std_xrr
    else:
        return (grid-mean_xrr)/std_xrr, mean_xrr, std_xrr
