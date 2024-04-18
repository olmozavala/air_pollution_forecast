import numpy as np
import xarray as xr
from proj_preproc.utils import getEvenIndexForSplit


def crop_variables_xr(xr_ds, variables, bbox, times):
    """
    Crop variables in an xarray Dataset based on a given bounding box and time range.

    Parameters:
        xr_ds (xr.Dataset): The input xarray Dataset containing the variables to be cropped.
        variables (list): A list of variable names to be cropped.
        bbox (tuple): A tuple containing the bounding box coordinates in the order (minlat, maxlat, minlon, maxlon).
        times (list): A list of time values to be cropped.

    Returns:
        xr.Dataset: The cropped xarray Dataset containing the specified variables, cropped to the given bounding box and time range.
        newLat (ndarray): The latitude values of the cropped variables.
        newLon (ndarray): The longitude values of the cropped variables.
    """
    output_xr_ds = xr.Dataset()
    for cur_var_name in variables:
        cur_var = xr_ds[cur_var_name]
        cur_coords_names = list(cur_var.coords.keys())
        lat = cur_var.coords[cur_coords_names[0]].values
        lon = cur_var.coords[cur_coords_names[1]].values

        minlat, maxlat, minlon, maxlon = bbox
        croppedVar, newLat, newLon = crop_variable_np(cur_var, LON=lon, LAT=lat, minlat=minlat, maxlat=maxlat,
                                                      minlon=minlon, maxlon=maxlon, times=times)
        output_xr_ds[cur_var_name] = xr.DataArray(croppedVar.values, coords=[('newtime', times), ('newlat', newLat), ('newlon', newLon)])

    return output_xr_ds, newLat, newLon


def crop_variables_xr_cca_reanalisis(xr_ds, variables, bbox, times, LAT, LON):
    output_xr_ds = xr.Dataset()
    for cur_var_name in variables:
        # print(F"\t\t {cur_var_name}")
        cur_var = xr_ds[cur_var_name]
        minlat, maxlat, minlon, maxlon = bbox
        croppedVar, newLat, newLon = crop_variable_np(cur_var, LON=LON, LAT=LAT, minlat=minlat, maxlat=maxlat,
                                                      minlon=minlon, maxlon=maxlon, times=times)
        output_xr_ds[cur_var_name] = xr.DataArray(croppedVar.values, coords=[('newtime', times), ('newlat', newLat), ('newlon', newLon)])

    return output_xr_ds, newLat, newLon


def crop_variable_np(np_data, LON, LAT, minlat, maxlat, minlon, maxlon, times):
    """
    Crop a variable from a NumPy array based on latitude and longitude bounds.

    Args:
        np_data (ndarray): NumPy array containing the variable data.
        LON (ndarray): NumPy array containing the longitude values.
        LAT (ndarray): NumPy array containing the latitude values.
        minlat (float): Minimum latitude value for cropping.
        maxlat (float): Maximum latitude value for cropping.
        minlon (float): Minimum longitude value for cropping.
        maxlon (float): Maximum longitude value for cropping.
        times (int or slice): Index or slice object specifying the time dimension to crop.

    Returns:
        tuple: A tuple containing the cropped variable data, the new latitude array, and the new longitude array.

    Raises:
        None

    """
    dims = len(LAT.shape)
    if dims == 1:
        minLatIdx = np.argmax(LAT >= minlat)
        maxLatIdx = np.argmax(LAT >= maxlat)-1
        minLonIdx = np.argmax(LON >= minlon)
        maxLonIdx = np.argmax(LON >= maxlon)-1

        newLAT = LAT[minLatIdx:maxLatIdx]
        newLon = LON[minLonIdx:maxLonIdx]

        croppedVar = np_data[times,minLatIdx:maxLatIdx, minLonIdx:maxLonIdx]

    if dims == 3:
        minLatIdx = np.argmax(LAT[0,:,0] >= minlat)
        maxLatIdx = np.argmax(LAT[0,:,0] >= maxlat)-1
        minLonIdx = np.argmax(LON[0,0,:] >= minlon)
        maxLonIdx = np.argmax(LON[0,0,:] >= maxlon)-1

        # Just for debugging
        # minLatVal = LAT[0,minLatIdx,0]
        # minLonVal = LON[0,0,minLonIdx]
        # maxLatVal = LAT[0,maxLatIdx,0]
        # maxLonVal = LON[0,0,maxLonIdx]
        # Just for debugging end

        newLAT = LAT[0,minLatIdx:maxLatIdx, 0]
        newLon = LON[0,0,minLonIdx:maxLonIdx]

        croppedVar = np_data[times,minLatIdx:maxLatIdx, minLonIdx:maxLonIdx]

    return croppedVar, newLAT, newLon


def subsampleData(xr_ds, variables, num_rows, num_cols):
    """
    Subsamples xr_ds in the spacial domain (means for every hour in a subregion)

    :param xr_ds: information of NetCDF
    :type xr_ds: NetCDF
    :return : 4 submatrices
    :return type : matrix float32
    """

    output_xr_ds = xr.Dataset() # Creates empty dataset
    # Retrieving the new values for the coordinates
    cur_coords_names = list(xr_ds.coords.keys())

    # TODO hardcoded order
    # Resampling dimensions first (assume all variables have the same dimensions, not cool)
    lat_vals =xr_ds[cur_coords_names[1]].values
    lon_vals =xr_ds[cur_coords_names[2]].values

    lat_splits_idx = getEvenIndexForSplit(len(lat_vals), num_rows)
    lon_splits_idx = getEvenIndexForSplit(len(lon_vals), num_cols)

    newlat = [lat_vals[i:j].mean() for i,j in lat_splits_idx]
    newlon = [lon_vals[i:j].mean() for i,j in lon_splits_idx]

    for cur_var_name in variables:
        cur_var = xr_ds[cur_var_name].values
        num_hours = cur_var.shape[0]
        mean_2d_array = np.zeros((num_hours, num_rows, num_cols))
        for i in range(num_hours):
            # Here we split the original array into the desired columns and rows
            for cur_row in range(num_rows):
                lat_start = lat_splits_idx[cur_row][0]
                lat_end = lat_splits_idx[cur_row][1]
                for cur_col in range(num_cols):
                    lon_start = lon_splits_idx[cur_col][0]
                    lon_end = lon_splits_idx[cur_col][1]
                    mean_2d_array[i, cur_row, cur_col] = cur_var[i, lat_start:lat_end, lon_start:lon_end].mean()

        output_xr_ds[cur_var_name] = xr.DataArray(mean_2d_array, coords=[('newtime', range(num_hours)),
                                                                         ('newlat', newlat),
                                                                         ('newlon', newlon)])
        # viz_obj.plot_3d_data_singlevar_np(output_array, z_levels=range(len(output_array)),
        #                                   title=F'Shape: {num_rows}x{num_cols}',
        #                                   file_name_prefix='AfterCroppingAndSubsampling')

    return output_xr_ds, newlat, newlon