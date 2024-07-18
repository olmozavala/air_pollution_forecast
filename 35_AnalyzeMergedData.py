#%%
import pandas as pd
from conf.MakeWRF_and_DB_CSV_UserConfiguration import getPreprocWRFParams, getPreprocDBParams
from conf.params import PreprocParams, DBToCSVParams
import matplotlib.pyplot as plt

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
# Plot cont otres for station CUA 'cont_otres_CUA'

data = df.filter(regex='cont_otres_CUA')
# Just plot the first two days (24*2=48)
data.iloc[1000:1048].plot()
# %%

# Print all columns that contains CUA from df
myregex = f".*CUA.*"
print(df.reindex(columns=df.columns[df.columns.str.contains(myregex)]).columns)

# %%
# IN a single plot include otres, pm10, pm25, from CUA
station = 'MER'
data = df.filter(regex='cont_otres_MER')
data = data.join(df.filter(regex='cont_pmdiez_MER'))
data = data.join(df.filter(regex='cont_pmdoscinco_MER'))
# Include now meteorlogical variables from the first quadrant for the first hour

start = 1200
end = start+96
data = data.iloc[start:end]
# Normalize each column to the maximum value
data = (data - data.min())/ (data.max() - data.min())
# Change figure width to 15 and height to 5
data.plot(figsize=(15,5))

# %%
# Include now meteorlogical variables from the first quadrant for the first hour
data = df.filter(regex='T2_6_h0')
# data = data.join(df.filter(regex='SWDOWN_6_h0'))
# data = data.join(df.filter(regex='GLW_6_h0'))
data = data.join(df.filter(regex='U10_6_h0'))
data = data.join(df.filter(regex='V10_6_h0'))

start = 1100
end = start+96
data = data.iloc[start:end]
# Normalize each column to the maximum value
data = (data - data.min())/ (data.max() - data.min())
# Change figure width to 15 and height to 5
data.plot(figsize=(15,5))

# %% Print all columns of df with MER
myregex = f".*T2.*"
print(df.reindex(columns=df.columns[df.columns.str.contains(myregex)]).columns)