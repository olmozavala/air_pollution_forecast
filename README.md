# Air pollution forecast with ML and meteorological forecast

## Installation
You need to create a `.netrc` file with your credentials to access the database at OWGIS. 
Change permissions to `chmod og-rw .netrc`

## Files summary 
### 1_MakeCSV_From_WRF.py
This file generates CSV files from the mounted WRF directories in ZION. 
The paths that this file needs access are:
`/ServerData/Wrf_Kraken` and `/ServerData/CHACKMOOL/Reanalisis`.

### inout.py

