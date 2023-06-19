#%%
import pandas as pd
from conf.MakeWRF_and_DB_CSV_UserConfiguration import getPreprocWRFParams, getPreprocDBParams
from conf.params import PreprocParams, DBToCSVParams

file_name = "/ZION/AirPollutionData/Data/MergedDataCSV/16/2010_AllStations.csv"
df = pd.read_csv(file_name, index_col=0, parse_dates=True)
conf = getPreprocWRFParams()

confdb = getPreprocDBParams()

# %%
# Print the total number of columns in the data frame
tot_cols = df.shape[1]
print(f"Total number of columns: {tot_cols}")

# %%
meteo_vars = len(conf[PreprocParams.variables])
tot_meteo = 24*16*meteo_vars
myregex = f"{'.*|'.join(conf[PreprocParams.variables])}.*"
print(myregex)
meteo_cols = df.filter(regex=myregex).shape[1]
print(f"\nAll meteorological columns {meteo_cols} = {tot_meteo} =  24 hrs * 16 Cuadrants * {meteo_vars} = {24*16*meteo_vars} ")
# Print the tot_count number of columns for each variable
tot_count = 0
for c_var in conf[PreprocParams.variables]:
    c_total = df.filter(regex=f'{c_var} *').shape[1]
    tot_count += c_total
    print(f"Number columns related to {c_var}: {c_total} sum: {tot_count}")
# %%
# Print the total number of stations used 
myregex = f"cont_.*"
print(myregex)
contaminants_columns = df.filter(regex=myregex).columns
print(f"\nAll contaminants columns {len(contaminants_columns)}/{tot_cols} ")
tot_count = 0
for c_var in confdb[DBToCSVParams.stations]:
    c_cols = df.filter(regex=f'_{c_var}').columns
    c_total = len(c_cols)
    if c_total > 0:
        tot_count += c_total
        print(f"Number columns related to {c_var}: {c_total} sum: {tot_count}: {list(c_cols)}")

# %%
# myregex = f"^(?!{'.*|'.join(conf[PreprocParams.variables])}).*$"