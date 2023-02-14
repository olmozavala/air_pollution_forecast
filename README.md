# Air pollution forecast with ML and meteorological forecast

## Installation
You need to create a `.netrc` file with your credentials to access the database at OWGIS. 
Change permissions to `chmod og-rw .netrc`

## Files summary 
### 1_MakeCSV_From_WRF.py
This file generates CSV files from the mounted WRF directories in ZION. 
The paths that this file needs access are:
`/ServerData/Wrf_Kraken` and `/ServerData/CHACKMOOL/Reanalisis`.

The output of this file is being saved at ZION at: `/ZION/AirPollutionData/Data/WRF_CSV`

### 2_MakeCSV_From_DB.py
This file generates CSV files from the data at the DB.
The paths that this file needs access are:
`/ServerData/Wrf_Kraken` and `/ServerData/CHACKMOOL/Reanalisis`.

The output of this file is being saved at ZION at: `/ZION/AirPollutionData/Data/DataPollutionDB_CSV`

### 3_MergeData.py
Merges the CVS data from the DB and WRF. 
The paths that this file needs access are:
`/ZION/AirPollutionData/Data/DataPollutionDB_CSV`
and `/ZION/AirPollutionData/Data/WRF_CSV`

The output of this file is being saved at ZION at: `/ZION/AirPollutionData/Data/MergedDataCSV`