from conf.MakeWRF_and_DB_CSV_UserConfiguration import getPreprocDBParams
from conf.params import DBToCSVParams
from db.sqlCont import getPostgresConn
from db.queries import *
from conf.localConstants import constants
from os.path import join
import os
import numpy as np
from pandas import DataFrame

## Reads user configuration
user_config = getPreprocDBParams()
pollutants = user_config[DBToCSVParams.tables]
output_folder = user_config[DBToCSVParams.output_folder]
start_date = user_config[DBToCSVParams.start_date]
end_date = user_config[DBToCSVParams.end_date]
stations = user_config[DBToCSVParams.stations]

if not (os.path.exists(output_folder)):
    os.makedirs(output_folder)

## Connect to db
conn = getPostgresConn()
for cur_station in stations:
# for cur_station in ['IZT']:
    print(F" ====================== {cur_station} ====================== ")
    # for cur_pollutant in pollutants:
    for cur_pollutant in ['cont_co']:
        print(F"\t ---------------------- {cur_pollutant} ---------------------- ")
        cur_data = np.array(getPollutantFromDateRange(conn, cur_pollutant, start_date, end_date, [cur_station]))
        if len(cur_data) > 0:
            dates = np.array([x[0] for x in cur_data])
            print(F"\tTotal number of rows obtained for {cur_station}-{cur_pollutant}: {len(dates)}")

            # Finding 'clean' dates (those dates with continuous 'num_hours' number of hours
            # clean_dates = []
            # for idx, cur_date in enumerate(dates):
            #     desired_date = cur_date + timedelta(hours=num_hours)
            #     if len(dates) > (idx + num_hours):
            #         if desired_date == dates[idx + num_hours]:
            #             clean_dates.append(idx)
            # print(F"\tTotal CLEANED # of rows for {cur_station}-{cur_pollutant}: {len(clean_dates)}")

            # Selects the dates of this 'file'
            df = DataFrame({cur_pollutant: cur_data[:,1]}, index=dates)
            file_name = F"{cur_pollutant}_{cur_station}.csv"
            df.to_csv(join(output_folder, file_name), index_label=constants.index_label.value)
        else:
            print("\t\t Warning!!!  NO DATA")

conn.close()
