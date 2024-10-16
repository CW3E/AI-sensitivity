def get_date_ic_sfno(date_ic,path_data,vars):
      grid=[]
      for var in vars:
          grid_i=xr.open_dataset(path_data+var+'.nc').sel(time=date_ic)
          grid.append(grid_i)
      grid=xr.merge(grid)
      grid=grid.expand_dims(dim ={'time': 1}, axis=0)
      return grid
