from conf.params import PreprocParams
from conf.params import DBToCSVParams
from conf.localConstants import constants
from os.path import join

from db.names import *

output_folder = '/ZION/AirPollutionData/Data'  # Where are the CVS files saved

def getPreprocWRFParams():
    make_csv_config= {
        # PreprocParams.variables: ['U10', 'V10', 'RAINC', 'T2', 'TH2', 'RAINNC', 'PBLH', 'SWDOWN', 'GLW'],
        PreprocParams.variables: ['U10', 'V10', 'RAINC', 'T2', 'RAINNC', 'SWDOWN', 'GLW'],
        # Donde se guardan los csv
        # PreprocParams.input_folder_new: '/data/UNAM/Air_Pollution_Forecast/Data/WRF_Kraken/new_model',
        PreprocParams.input_folder_new: '/ServerData/WRF_Kraken',  # Paths at ZION
        # PreprocParams.input_folder_old: '/data/UNAM/Air_Pollution_Forecast/Data/WRF_Kraken/old_model_v4',
        PreprocParams.input_folder_old: '/ServerData/CHACMOOL/Reanalisis/RESPALDO_V4/',  # Paths at ZION
        PreprocParams.output_folder: join(output_folder, constants.wrf_output_folder.value),
        PreprocParams.output_imgs_folder: join(output_folder, 'imgs'), # Path to save temporal images (netcdfs preprocessing)
        PreprocParams.display_imgs: True,  # Boolean that indicates if we want to save the images
        # How to subsample the data
        PreprocParams.resampled_output_sizes: [{'rows': 1, 'cols': 1},
                                                 {'rows': 2, 'cols': 2},
                                                 {'rows': 4, 'cols': 4},
                                                 {'rows': 8, 'cols': 8},
                                                 {'rows': 16, 'cols': 16}],
        # How to crop the data [minlat, maxlat, minlon, maxlon]
        PreprocParams.bbox: [19.05,20,-99.46, -98.7],
        PreprocParams.times: range(72),  # What is this?
        # Start and end date to generate the CSVs. The dates are in python 'range' style. Start day
        # is included, last day is < than.
        # These next 4 all for ALL the years
        PreprocParams.start_date_newmodel: '2017-01-01',
        PreprocParams.end_date_newmodel: '2019-12-31',
        PreprocParams.start_date_oldmodel: '1980-01-01',
        PreprocParams.end_date_oldmodel: '2017-01-01',
        # PreprocParams.start_date_newmodel: '2019-01-01',
        # PreprocParams.end_date_newmodel: '2019-01-02',
        # PreprocParams.start_date_oldmodel: '1980-01-02',
        # PreprocParams.end_date_oldmodel: '1980-01-03',
        }

    return make_csv_config

# All stations: ["ACO", "AJM", "AJU", "ARA", "ATI", "AZC", "BJU", "CAM", "CCA", "CES", "CFE", "CHO", "COR", "COY", "CUA"
#,"CUI", "CUT", "DIC", "EAJ", "EDL", "FAC", "FAN", "GAM", "HAN", "HGM", "IBM", "IMP", "INN", "IZT", "LAA", "LAG", "LLA"
#,"LOM", "LPR", "LVI", "MCM", "MER", "MGH", "MIN", "MON", "MPA", "NET", "NEZ", "PED", "PER", "PLA", "POT", "SAG", "SFE"
#,"SHA", "SJA", "SNT", "SUR", "TAC", "TAH", "TAX", "TEC", "TLA", "TLI", "TPN", "UAX", "UIZ", "UNM", "VAL", "VIF", "XAL"
#,"XCH"]

def getPreprocDBParams():
    '''
    This function obtains the shared parameters regarding the access to the database.
    :return:
    '''
    make_csv_config= {
        # DBToCSVParams.tables: [getTables()[0]],
        DBToCSVParams.tables: getTables(),
        # DBToCSVParams.stations: ["AJM"],
        DBToCSVParams.stations: ["ACO", "AJM", "AJU", "ARA", "ATI", "AZC", "BJU", "CAM", "CCA", "CES", "CFE", "CHO", "COR", "COY", "CUA"
              ,"CUI", "CUT", "DIC", "EAJ", "EDL", "FAC", "FAN", "GAM", "HAN", "HGM", "IBM", "IMP", "INN", "IZT", "LAA", "LAG", "LLA"
              ,"LOM", "LPR", "LVI", "MCM", "MER", "MGH", "MIN", "MON", "MPA", "NET", "NEZ", "PED", "PER", "PLA", "POT", "SAG", "SFE"
              ,"SHA", "SJA", "SNT", "SUR", "TAC", "TAH", "TAX", "TEC", "TLA", "TLI", "TPN", "UAX", "UIZ", "UNM", "VAL", "VIF", "XAL"
              ,"XCH"],
        # Donde se guardan los csv
        DBToCSVParams.output_folder: join(output_folder, constants.db_output_folder.value),
        DBToCSVParams.display_imgs: True,  # Boolean that indicates if we want to save the images
        DBToCSVParams.num_hours: 72,
        # Start and end date to generate the CSVs. The dates are in python 'range' style. Start day
        # is included, last day is < than.
        DBToCSVParams.start_date: '1980-01-01',
        DBToCSVParams.end_date: '2019-12-31',
    }

    return make_csv_config

