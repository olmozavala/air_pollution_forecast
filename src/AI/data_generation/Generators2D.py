import numpy as np
import re
from inout.readData import *
from scipy.ndimage.filters import gaussian_filter
import preproc.utils as utils
import matplotlib.pyplot as plt

def data_gen_sarga(path, folders_to_read):
    """
    In this generator every input is interesected with the Prostate contour
    :param path:
    :param folders_to_read:
    :param img_names:
    :param roi_names:
    :param tot_ex_per_img
    :return:
    """
    print( "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CALLED Generator {} OZ Version 0s outside prostate, 1s in prostate and 2s in lesions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%".format(len(folders_to_read)))

    curr_idx = -1 # First index to use
    while True:
        # These lines are for sequential selection
        if curr_idx < (len(folders_to_read) - 1):
            curr_idx += 1
        else:
            curr_idx = 0
            np.random.shuffle(folders_to_read) # We shuffle the folders every time we have tested all the examples

        cur_folder = folders_to_read[curr_idx]
        file_name = join(path, cur_folder)
        try:

            # *********************** Reading files **************************
            tot_imgs = len(img_names)
            all_imgcube = np.zeros((3,168,168,168)) # Temporal variable that will hold the images of this patient
            # TODO hardcoded to 168cube change it to maybe the resolution of the image?
            all_imgcontour = np.zeros((1,168,168,168))
            for img_idx in range(tot_imgs):
                all_imgcube[img_idx] = sitk.GetArrayFromImage(sitk.ReadImage(join(path,file_name,img_names[img_idx])))

            file_names = listdir(join(path,file_name))
            foundProstate = False
            foundLesion = False

            # Get the prostat and adds it to the 'output contour'
            roi_name = roi_names[0]
            for filen in file_names:
                if not re.search(roi_name, filen) is None:
                    all_imgcontour[0] = sitk.GetArrayFromImage(sitk.ReadImage(join(path,file_name,filen)))
                    foundProstate = True
                    break

            # Remove everything outside the prostate as input (assuming it will be easier)
            all_imgcube[0,:,:,:] = all_imgcube[0,:,:,:]*all_imgcontour[0,:,:,:]
            all_imgcube[1,:,:,:] = all_imgcube[1,:,:,:]*all_imgcontour[0,:,:,:]
            all_imgcube[2,:,:,:] = all_imgcube[2,:,:,:]*all_imgcontour[0,:,:,:]

            # Get the lesions and adds them to th 'output contour' It assumes the whole prostate is 1 and this will create a 2
            roi_name = roi_names[1]
            for file in file_names:
                # Example re.search('^roi_ctr_lesion_[T|[2|3|4|5]].nrrd', 'roi_ctr_lesion_2.nrrd')
                if not re.search(roi_name, file) is None:
                    all_imgcontour[0] += sitk.GetArrayFromImage(sitk.ReadImage(join(path,file_name,file)))
                    foundLesion = True

            assert (foundProstate and foundLesion)

            successful_examples = 0

            # Select lesions with a lesion
            slicesWithLesion = np.array([slice for slice in range(0,168) if np.max(all_imgcontour[0][slice,:,:]) >= 1.5])
            np.random.shuffle(slicesWithLesion)
            for slice in slicesWithLesion:
                # Only use slices that have data (lesion inside)
                X1 = np.expand_dims(np.expand_dims(all_imgcube[0][slice,:,:], axis=3), axis=0)
                X2 = np.expand_dims(np.expand_dims(all_imgcube[1][slice,:,:], axis=3), axis=0)
                X3 = np.expand_dims(np.expand_dims(all_imgcube[2][slice,:,:], axis=3), axis=0)

                Y = np.expand_dims(np.expand_dims(all_imgcontour[0][slice,:,:], axis=3), axis=0)
                X = [X1,X2,X3]

                # To visualize the final input easy way
                # utilsviz.view_results = True
                # utilsviz.drawMultipleSeriesNumpy(all_imgcube, slices=[slice], contours=all_imgcontour)
                # utilsviz.drawMultipleSeriesNumpy(all_imgcontour, slices=[slice], contours=all_imgcontour, draw_only_contours=False)

                yield X, Y
                successful_examples += 1
                if successful_examples >= tot_ex_per_img:
                    break # Only one slice per patient

        except Exception as e:
            print("----- Not able to generate for: ", cur_folder , " ERROR: ", str(e))


def data_gen_multi_ozv(path, folders_to_read, img_names, roi_names, tot_ex_per_img):
    """
    In this generator every input is interesected with the Prostate contour
    :param path:
    :param folders_to_read:
    :param img_names:
    :param roi_names:
    :param tot_ex_per_img
    :return:
    """
    print( "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CALLED Generator {} OZ Version 0s outside prostate, 1s in prostate and 2s in lesions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%".format(len(folders_to_read)))

    curr_idx = -1 # First index to use
    while True:
        # These lines are for sequential selection
        if curr_idx < (len(folders_to_read) - 1):
            curr_idx += 1
        else:
            curr_idx = 0
            np.random.shuffle(folders_to_read) # We shuffle the folders every time we have tested all the examples

        cur_folder = folders_to_read[curr_idx]
        file_name = join(path, cur_folder)
        try:

            # *********************** Reading files **************************
            tot_imgs = len(img_names)
            all_imgcube = np.zeros((3,168,168,168)) # Temporal variable that will hold the images of this patient
            # TODO hardcoded to 168cube change it to maybe the resolution of the image?
            all_imgcontour = np.zeros((1,168,168,168))
            for img_idx in range(tot_imgs):
                all_imgcube[img_idx] = sitk.GetArrayFromImage(sitk.ReadImage(join(path,file_name,img_names[img_idx])))

            file_names = listdir(join(path,file_name))
            foundProstate = False
            foundLesion = False

            # Get the prostat and adds it to the 'output contour'
            roi_name = roi_names[0]
            for filen in file_names:
                if not re.search(roi_name, filen) is None:
                    all_imgcontour[0] = sitk.GetArrayFromImage(sitk.ReadImage(join(path,file_name,filen)))
                    foundProstate = True
                    break

            # Remove everything outside the prostate as input (assuming it will be easier)
            all_imgcube[0,:,:,:] = all_imgcube[0,:,:,:]*all_imgcontour[0,:,:,:]
            all_imgcube[1,:,:,:] = all_imgcube[1,:,:,:]*all_imgcontour[0,:,:,:]
            all_imgcube[2,:,:,:] = all_imgcube[2,:,:,:]*all_imgcontour[0,:,:,:]

            # Get the lesions and adds them to th 'output contour' It assumes the whole prostate is 1 and this will create a 2
            roi_name = roi_names[1]
            for file in file_names:
                # Example re.search('^roi_ctr_lesion_[T|[2|3|4|5]].nrrd', 'roi_ctr_lesion_2.nrrd')
                if not re.search(roi_name, file) is None:
                    all_imgcontour[0] += sitk.GetArrayFromImage(sitk.ReadImage(join(path,file_name,file)))
                    foundLesion = True

            assert (foundProstate and foundLesion)

            successful_examples = 0

            # Select lesions with a lesion
            slicesWithLesion = np.array([slice for slice in range(0,168) if np.max(all_imgcontour[0][slice,:,:]) >= 1.5])
            np.random.shuffle(slicesWithLesion)
            for slice in slicesWithLesion:
                # Only use slices that have data (lesion inside)
                X1 = np.expand_dims(np.expand_dims(all_imgcube[0][slice,:,:], axis=3), axis=0)
                X2 = np.expand_dims(np.expand_dims(all_imgcube[1][slice,:,:], axis=3), axis=0)
                X3 = np.expand_dims(np.expand_dims(all_imgcube[2][slice,:,:], axis=3), axis=0)

                Y = np.expand_dims(np.expand_dims(all_imgcontour[0][slice,:,:], axis=3), axis=0)
                X = [X1,X2,X3]

                # To visualize the final input easy way
                # utilsviz.view_results = True
                # utilsviz.drawMultipleSeriesNumpy(all_imgcube, slices=[slice], contours=all_imgcontour)
                # utilsviz.drawMultipleSeriesNumpy(all_imgcontour, slices=[slice], contours=all_imgcontour, draw_only_contours=False)

                yield X, Y
                successful_examples += 1
                if successful_examples >= tot_ex_per_img:
                    break # Only one slice per patient

        except Exception as e:
            print("----- Not able to generate for: ", cur_folder , " ERROR: ", str(e))

def data_gen_multi_prostate(path, folders_to_read, img_names, roi_names):
    """
    Generator to yield inputs and their labels in batches.
    IT ONLY WORKS FOR BATCH SIZES OF 1
    """
    print( "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CALLED Generator {} %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%".format(len(folders_to_read)))

    curr_idx = -1 # First index to use
    while True:
        # These lines are for sequential selection
        if curr_idx < (len(folders_to_read) - 1):
            curr_idx += 1
        else:
            curr_idx = 0
            np.random.shuffle(folders_to_read) # We shuffle the folders every time we have tested all the examples

        cur_folder = folders_to_read[curr_idx]
        file_name = join(path, cur_folder)
        try:

            # *********************** Reading files **************************
            tot_imgs = len(img_names)
            all_imgcube = np.zeros((3,168,168,168)) # Temporal variable that will hold the images of this patient
            # TODO hardcoded to 168cube change it to maybe the resolution of the image?
            all_imgcontour = np.zeros((1,168,168,168))
            for img_idx in range(tot_imgs):
                all_imgcube[img_idx] = sitk.GetArrayFromImage(sitk.ReadImage(join(path,file_name,img_names[img_idx])))

            file_names = listdir(join(path,file_name))
            foundLesion = False

            # Get the prostate
            roi_name = roi_names[0]
            for file in file_names:
                if not re.search(roi_name, file) is None:
                    all_imgcontour[0] += sitk.GetArrayFromImage(sitk.ReadImage(join(path,file_name,file)))
                    foundLesion = True

            assert (foundLesion)

            dims =all_imgcontour[0].shape
            tot_ex_per_img = 5 # How many examples are we using for each image
            successful_examples = 0
            for slice in np.random.randint(0, dims[0], 100):# Randomly select the slices
                # Only use slices that have data (lesion inside)
                # if np.max(all_imgcontour[0][slice,:,:]) >= 1.5: # Make sure there is a lesion there
                # TODO only if we are NOT Using the prostate
                if np.max(all_imgcontour[0][slice,:,:]) >= .5: # Make sure there is a lesion there

                    # Reordering the images
                    X1 = np.expand_dims(np.expand_dims(all_imgcube[0][slice,:,:], axis=3), axis=0)
                    X2 = np.expand_dims(np.expand_dims(all_imgcube[1][slice,:,:], axis=3), axis=0)
                    X3 = np.expand_dims(np.expand_dims(all_imgcube[2][slice,:,:], axis=3), axis=0)

                    Y = np.expand_dims(np.expand_dims(all_imgcontour[0][slice,:,:], axis=3), axis=0)
                    X = [X1,X2,X3]

                    # utilsviz.drawMultipleSeriesNumpy(all_imgcube, slices=[slice], contours=all_imgcontour)
                    # utilsviz.drawMultipleSeriesNumpy(all_imgcontour, slices=[slice], contours=all_imgcontour, draw_only_contours=False)

                    yield X, Y
                    successful_examples += 1
                    if successful_examples >= tot_ex_per_img:
                        break # Only one slice per patient

        except Exception as e:
            print("----- Not able to generate for: ", cur_folder , " ERROR: ", str(e))

def data_gen_multi_RM_input_prostate(path, folders_to_read, img_names, roi_names, tot_ex_per_img):
    """
    Generator to yield inputs and their labels in batches.
    IT ONLY WORKS FOR BATCH SIZES OF 1
    """
    print( "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CALLED Generator {} %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%".format(len(folders_to_read)))

    curr_idx = -1 # First index to use
    while True:
        # These lines are for sequential selection
        if curr_idx < (len(folders_to_read) - 1):
            curr_idx += 1
        else:
            curr_idx = 0
            np.random.shuffle(folders_to_read) # We shuffle the folders every time we have tested all the examples

        cur_folder = folders_to_read[curr_idx]
        file_name = join(path, cur_folder)
        try:

            # *********************** Reading files **************************
            tot_imgs = len(img_names)
            all_imgcube = np.zeros((3,168,168,168)) # Temporal variable that will hold the images of this patient
            # TODO hardcoded to 168cube change it to maybe the resolution of the image?
            all_imgcontour = np.zeros((1,168,168,168))
            for img_idx in range(tot_imgs):
                all_imgcube[img_idx] = sitk.GetArrayFromImage(sitk.ReadImage(join(path,file_name,img_names[img_idx])))

            file_names = listdir(join(path,file_name))
            foundLesion = False

            # Get the prostate
            roi_name = roi_names[0]
            for file in file_names:
                    if not re.search(roi_name, file) is None:
                        all_imgcontour[0] = sitk.GetArrayFromImage(sitk.ReadImage(join(path,file_name,file)))
                        foundProstate = True
                        break

            # ORDER IS T2, BVAL, ADC
            # Remove everything outside the prostate as input (assuming it will be easier)
            all_imgcube[0,:,:,:] = all_imgcube[0,:,:,:]*all_imgcontour[0,:,:,:]
            all_imgcube[1,:,:,:] = all_imgcube[1,:,:,:]*all_imgcontour[0,:,:,:]
            all_imgcube[2,:,:,:] = all_imgcube[2,:,:,:]*all_imgcontour[0,:,:,:]


            # TODO this part is only if we DO NOT want to use the prostate for nothing
            all_imgcontour[0,:,:,:] = 0

            # Get the lesions
            roi_name = roi_names[1]
            for file in file_names:
                if not re.search(roi_name, file) is None:
                    all_imgcontour[0] += sitk.GetArrayFromImage(sitk.ReadImage(join(path,file_name,file)))
                    foundLesion = True

            assert (foundProstate and foundLesion)

            dims =all_imgcontour[0].shape
            tot_ex_per_img = tot_ex_per_img # How many examples are we using for each image
            successful_examples = 0
            for slice in np.random.randint(0, dims[0], 100):# Randomly select the slices
                # Only use slices that have data (lesion inside)
                # if np.max(all_imgcontour[0][slice,:,:]) >= 1.5: # Make sure there is a lesion there
                # TODO only if we are NOT Using the prostate
                if np.max(all_imgcontour[0][slice,:,:]) >= .5: # Make sure there is a lesion there

                    # Reordering the images
                    X1 = np.expand_dims(np.expand_dims(all_imgcube[0][slice,:,:], axis=3), axis=0)
                    X2 = np.expand_dims(np.expand_dims(all_imgcube[1][slice,:,:], axis=3), axis=0)
                    X3 = np.expand_dims(np.expand_dims(all_imgcube[2][slice,:,:], axis=3), axis=0)

                    Y = np.expand_dims(np.expand_dims(all_imgcontour[0][slice,:,:], axis=3), axis=0)
                    X = [X1,X2,X3]

                    utilsviz.drawMultipleSeriesNumpy(all_imgcube, slices=[slice], contours=all_imgcontour)
                    utilsviz.drawMultipleSeriesNumpy(all_imgcontour, slices=[slice], contours=all_imgcontour, draw_only_contours=False)

                    yield X, Y
                    successful_examples += 1
                    if successful_examples >= tot_ex_per_img:
                        break # Only one slice per patient

        except Exception as e:
            print("----- Not able to generate for: ", cur_folder , " ERROR: ", str(e))
