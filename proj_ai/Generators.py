
# %%

from datetime import datetime
from ai_common.constants.AI_params import NormParams
from conf.localConstants import constants

from proj_ai.TrainingUtils import split_train_validation_and_test
from io_utils.io_common import create_folder
from proj_io.inout import add_forecasted_hours, add_previous_hours, filter_data, get_column_names, read_merged_files, save_columns
from proj_preproc.preproc import apply_bootstrap, normalizeData
import torch
from os.path import join
from torch.utils.data import Dataset, DataLoader
import numpy as np


## ------- Custom dataset ------
class AirPollutionDataset(Dataset):

    def __init__(self, input_folder, start_year, end_year, model_name, 
                 cur_pollutant, output_folder, norm_type,
                 forecasted_hours, hours_before, 
                 val_perc,  
                 bootstrap = False, boostrap_factor = 10, boostrap_threshold = 2.9,
                validation_data = False, transform=None):

        # %% ====== Creating the output folders 
        split_info_folder = join(output_folder, 'Splits')
        parameters_folder = join(output_folder, 'Parameters')
        weights_folder = join(output_folder, 'models')
        logs_folder = join(output_folder, 'logs')
        imgs_folder= join(output_folder, 'imgs')
        norm_folder = join(output_folder, 'norm')
        create_folder(split_info_folder)
        create_folder(parameters_folder)
        create_folder(weights_folder)
        create_folder(logs_folder)
        create_folder(norm_folder)
        create_folder(imgs_folder)

        # %% Reading the data
        data = read_merged_files(input_folder, start_year, end_year)

        datetimes_str = data.index.values
        datetimes = np.array([datetime.strptime(x, constants.datetime_format.value) for x in datetimes_str])

        # %% -------- Normalizing data
        now = datetime.utcnow().strftime("%Y_%m_%d_%H_%M")
        model_name = F'{model_name}_{cur_pollutant}_{now}'
        file_name_norm = join(norm_folder,F"{model_name}_scaler.pkl")  
        print("Normalizing data....")
        data_norm_df = normalizeData(data, norm_type, file_name_norm)

        # %% ====== Getting all the orignal columns by type
        all_contaminant_columns, all_meteo_columns, all_time_colums = get_column_names(data_norm_df)

        # %% ====== Remove columns for other pollutants
        # Here we remove all the data of other pollutants
        X_df = filter_data(data_norm_df, filter_type='single_pollutant',
                        filtered_pollutant=cur_pollutant) 

        print(X_df.columns.values)
        print(F'X {X_df.shape}, Memory usage: {X_df.memory_usage().sum()/1024**2:02f} MB')

        # %% ====== Adding the previous hours of the pollutants as extra columns (all contaminants)
        print(F"Building X and Y ....")
        X_df = add_previous_hours(X_df, hours_before=24)

        # %% ====== Adding the forecasted hours of the pollutants as the predicted column Y (specific contaminant)
        print("\tAdding the forecasted hours of the pollutant as the predicted column Y...")
        Y_df = add_forecasted_hours(X_df, cur_pollutant, range(1,forecasted_hours+1))

        # %% Remove the first hours because Print the final shape of X and Y
        X_df = X_df.iloc[hours_before:,:]
        Y_df = Y_df.iloc[hours_before:,:]
        save_columns(Y_df, join(output_folder, 'Y_columns.csv'))
        save_columns(X_df, join(output_folder, 'X_columns.csv'))
        print("Done!")

        print(F'Original {data_norm_df.shape}')
        print(F'X {X_df.shape}, Memory usage: {X_df.memory_usage().sum()/1024**2:02f} MB')
        print(F'Y {Y_df.shape}, Memory usage: {Y_df.memory_usage().sum()/1024**2:02f} MB')

        #%% Split the training data by year
        print("Splitting training and validation data by year....")

        splits_file = join(split_info_folder, F'splits_{model_name}.csv')
        train_idxs, val_idxs, test_idxs = split_train_validation_and_test(
            len(X_df), val_perc, 0, shuffle_ids=False, file_name=splits_file)

        # %% Remove time as index 
        # Here we remove the datetime indexes so we need to consider that 
        print("Removing time index...")
        X_df.reset_index(drop=True, inplace=True)
        Y_df.reset_index(drop=True, inplace=True)

        if validation_data:
            X_df = X_df.iloc[val_idxs]
            Y_df = Y_df.iloc[val_idxs]
        else:
            X_df = X_df.iloc[train_idxs]
            Y_df = Y_df.iloc[train_idxs]

        data_type = "validation" if validation_data else "train"
        print(F'X {data_type} {X_df.shape}, Memory usage: {X_df.memory_usage().sum()/1024**2:02f} MB')
        print(F'Y {data_type} {Y_df.shape}, Memory usage: {Y_df.memory_usage().sum()/1024**2:02f} MB')

        print("Done!")

        # %% ======= Bootstrapping the data
        if bootstrap and not validation_data:
            # -------- Bootstrapping the data
            # Se utiliza esta estacion para decidir que indices son los que se van a usar para el bootstrapping.
            # Only the indexes for this station that are above the threshold will be used for bootstrapping
            station = "MER" 
            print("Bootstrapping the data...")
            print(F'X train {X_df.shape}, Memory usage: {X_df.memory_usage().sum()/1024**2:02f} MB')
            print(F'Y train {Y_df.shape}, Memory usage: {Y_df.memory_usage().sum()/1024**2:02f} MB')
            X_df, Y_df = apply_bootstrap(X_df, Y_df, cur_pollutant, station, boostrap_threshold, forecasted_hours, boostrap_factor)
            print(F'X train bootstrapped {X_df.shape}, Memory usage: {X_df.memory_usage().sum()/1024**2:02f} MB')
            print(F'Y train bootstrapped {Y_df.shape}, Memory usage: {Y_df.memory_usage().sum()/1024**2:02f} MB')


        #%% Replace all the nan values with another value
        replace_value = 0
        print(f"Replacing nan values with {replace_value}...")
        X_df.fillna(replace_value, inplace=True)
        X_df.fillna(replace_value, inplace=True)

        self.data = X_df
        self.targets = Y_df

        self.total_examples = len(self.data)

    def __len__(self):
        return self.total_examples

    def __getitem__(self, idx):
        '''Returns the data for the given index '''
        return self.data.iloc[idx].values, self.targets.iloc[idx].values

## ----- DataLoader --------
if __name__ == "__main__":
    # Test inputs
    start_year = 2010
    end_year = 2011
    hours_before = 8 # How many hours of pollution data are we adding as input to the model (current - hours_before)
    forecasted_hours = 24
    cur_pollutant = 'otres'
    cur_station = 'MER'
    val_perc = .1
    grid_size = 4
    bootstrap = True
    boostrap_factor = 10  # Number of times to repeat the bootstrap
    boostrap_threshold = 2.9
    merged_specific_folder = f'{grid_size*grid_size}' # We may have multiple folders inside merge depending on the cuadrants

    now = datetime.utcnow().strftime("%Y_%m_%d_%H_%M")
    model_name_user = "Torch_Test1"
    model_name = F'{model_name_user}_{cur_pollutant}_{now}'

    data_folder = '/ZION/AirPollutionData/Data/'

    input_folder = join(data_folder, "MergedDataCSV", merged_specific_folder)
    output_folder = join(data_folder, 'TrainingTestsOZ', 
                        F"MultipleStations_MultiplePollutants_{start_year}_{end_year}")

    batch_size = 5000
    norm_type = NormParams.mean_zero


    dataset_training = AirPollutionDataset(input_folder, start_year, end_year, model_name, 
                 cur_pollutant, output_folder, norm_type,
                 forecasted_hours, hours_before, 
                 val_perc, 
                 bootstrap, boostrap_factor, boostrap_threshold,
                  validation_data=False, transform=None)

    # dataset_val = AirPollutionDataset(input_folder, start_year, end_year, model_name, 
    #              cur_pollutant, output_folder, norm_type,
    #              forecasted_hours, hours_before, 
    #              val_perc, 
    #              bootstrap, boostrap_factor, boostrap_threshold,
    #               validation_data=True, transform=None)


    myloader = DataLoader(dataset_training, batch_size=10, shuffle=True)
    # --------- Just reading some lats lons ----------
    for batch in myloader:
        # Print the shape of the data and target of this batch
        print(f"Batch size: {batch.shape[0]} Size input: {batch[0][0].shape}  Size output: {batch[0][1].shape} " )