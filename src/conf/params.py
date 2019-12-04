from enum import Enum

class PreprocParams(Enum):
    variables = 1   # a list of str with the variable names to preprocess
    input_folder = 2  # Folder path where the WRF files will be searched
    output_folder = 3  # Folder path where the preprocessed files will be saved
    output_imgs_folder = 4  # Where to output temporal images
    display_imgs = 5  # Bool, indicates if the images should be displayed
    resampled_output_size = 6  # Array with the subsampled size to be tenerated
    bbox = 8  # Boundary box to be used for cropping the data (minlat, maxlat, minlon, maxlon)
    times = 9  # Array of times indexes to be used
    start_date =12  # Start date that is used fo filter the files being used
    end_date =13  # Start date that is used fo filter the files being used
