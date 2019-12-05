from enum import Enum

class PreprocParams(Enum):
    variables = 1   # a list of str with the variable names to preprocess
    input_folder = 2  # Folder path where the WRF files will be searched
    output_folder = 3  # Folder path where the preprocessed files will be saved
    output_imgs_folder = 4  # Where to output temporal images
    display_imgs = 5  # Bool, indicates if the images should be displayed
    resampled_output_sizes = 6  # Array with the subsampled size to be tenerated
    bbox = 8  # Boundary box to be used for cropping the data (minlat, maxlat, minlon, maxlon)
    times = 9  # Array of times indexes to be used
    start_date =12  # Start date that is used fo filter the files being used
    end_date =13  # Start date that is used fo filter the files being used


class DBToCSVParams(Enum):
    tables = 1  # A list of str with the names of the contaminants to process
    output_folder = 2  # Folder path where the preprocessed files will be saved
    output_imgs_folder = 3  # Where to output temporal images
    display_imgs = 4  # Bool, indicates if the images should be displayed
    start_date = 5  # Start date that is used fo filter the files being used
    end_date = 6  # Start date that is used fo filter the files being used
    num_hours = 7  # Integer indicating how many continuous times we need
    stations = 8  # List of stations to process

