
# This notebook contains a **hello world** example of neural networks with PyTorch. Basically a linear regression approximation
import torch
import os
import xarray as xr
from os.path import join
from torch.utils.data import Dataset, DataLoader
from shapely.geometry import Polygon
from io_utils.dates_utils import get_month_and_day_of_month_from_day_of_year
from viz_utils.eoa_viz import EOAImageVisualizer
from proj_io.contours import read_contours_polygons
import numpy as np
import pandas as pd


## ------- Custom dataset ------
class AirPollutionDataset(Dataset):

    def __init__(self, input_folder, start_year, end_year, ssh_folder, transform=None):

        datetimes_str = data.index.values
        datetimes = np.array([datetime.strptime(x, constants.datetime_format.value) for x in datetimes_str])


    def __len__(self):
        return self.total_days

    def __getitem__(self, idx):
        return ssh_day[:, :136, :].astype(np.float32), eddies[:, :136, :].astype(np.float32)

## ----- DataLoader --------
if __name__ == "__main__":
    # ----------- Skynet ------------
    # dataset = AirPollutionDataset(ssh_folder, eddies_folder, bbox, output_resolution)
# 
    # myloader = DataLoader(dataset, batch_size=2, shuffle=True)
    # --------- Just reading some lats lons ----------
    # for batch in myloader:
        # print(batch[0].shape)
        # ssh, eddies = batch
        # x = 1