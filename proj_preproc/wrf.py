import numpy as np
import xarray as xr
from proj_preproc.utils import getEvenIndexForSplit


def crop_variables_xr(xr_ds, variables, bbox, times):
    output_xr_ds = xr.Dataset()
    for cur_var_name in variables:
        # print(F"\t\t {cur_var_name}")
        cur_var = xr_ds[cur_var_name]
        cur_coords_names = list(cur_var.coords.keys())
        # TODO the order here is hardcoded, need to verify it always work for WRF
        # Also, the lat and lon is obtained everytime, assuming it may use XLAT_V so not always the same
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
    Crops a numpy array 'np_data' with the desired bbox
    :param np_data:
    :param LON:
    :param LAT:
    :param LONsize:
    :param LATsize:
    :param minlat:
    :param maxlat:
    :param minlon:
    :param maxlon:
    :return:
    """

    dims = len(LAT.shape)
    if dims == 1:
        minLatIdx = np.argmax(LAT >= minlat)
        maxLatIdx = np.argmax(LAT >= maxlat)-1
        minLonIdx = np.argmax(LON >= minlon)
        maxLonIdx = np.argmax(LON >= maxlon)-1

        # Just for debugging
        # minLatVal = LAT[minLatIdx]
        # maxLatVal = LAT[maxLatIdx]
        # minLonVal = LON[minLonIdx]
        # maxLonVal = LON[maxLonIdx]
        # Just for debugging end

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

    return output_xr_ds
