from conf.params import PreprocParams
from os.path import join


def getPreprocWRFParams():

    # output_folder = '/data/UNAM/OUTPUT'
    output_folder = '/home/olmozavala/REMOTE_PROJECTS/OUTPUT'

    make_csv_config= {
        PreprocParams.variables: ['U10', 'V10', 'RAINC', 'T2', 'TH2', 'RAINNC', 'PBLH', 'SWDOWN', 'GLW'],
        # Donde se guardan los csv
        # PreprocParams.input_folder: '/data/UNAM/WRF_Kraken/',
        PreprocParams.input_folder: '/ServerData/Pronosticos/Salidas/WRF_Kraken',
        PreprocParams.output_folder: join(output_folder, 'DataCSV'),
        PreprocParams.output_imgs_folder: join(output_folder, 'imgs'), # Path to save temporal images (netcdfs preprocessing)
        PreprocParams.display_imgs: True,  # Boolean that indicates if we want to save the images
        # How to subsample the data
        PreprocParams.resampled_output_size: {'rows':10, 'cols':8},
        # How to crop the data [minlat, maxlat, minlon, maxlon]
        PreprocParams.bbox: [19.05,20,-99.46, -98.7],
        PreprocParams.times: range(48),
        # Start and end date to generate the CSVs. The dates are in python 'range' style. Start day
        # is included, last day is < than.
        PreprocParams.start_date: '2019-01-01',
        PreprocParams.end_date: '2019-01-02',
        }

    return make_csv_config