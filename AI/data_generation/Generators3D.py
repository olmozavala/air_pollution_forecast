import numpy as np
from inout.readData import *
from AI.DataAugmentation import *
from inout.readDataPreproc import *
from img_viz.medical import MedicalImageVisualizer
from AI.data_generation.utilsDataFormat import format_for_nn_training

class Generator3D:
    def __init__(self, **kwargs):
        self.viz_obj = MedicalImageVisualizer(disp_images=True)
        # All the arguments that are passed to the constructor of the class MUST have its name on it.
        for arg_name, arg_value in kwargs.items():
            self.__dict__["_" + arg_name] = arg_value

    def __getattr__(self, attr):
        '''Generic getter for all the properties of the class'''
        return self.__dict__["_" + attr]

    def __setattr__(self, attr, value):
        '''Generic setter for all the properties of the class'''
        self.__dict__["_" + attr] = value

    def unet_3d_multi_streams(self, input_folder, folders_to_read, stream_file_names, ctr_file_name,
                              data_augmentation=True, batch_size=1):
        """
        Generator to yield inputs and their labels in batches.
        """
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CALLED Generator {} %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%".format(len(folders_to_read)))

        tot_num_streams = len(stream_file_names)
        curr_idx = -1 # First index to use
        while True:
            # These lines are for sequential selection
            if curr_idx < (len(folders_to_read) - batch_size):
                curr_idx += batch_size
            else:
                curr_idx = 0
                np.random.shuffle(folders_to_read) # We shuffle the folders every time we have tested all the examples

            c_folders = folders_to_read[curr_idx:curr_idx+batch_size]
            try:
                all_imgs, all_ctrs, _, _, _ = read_preproc_imgs_and_ctrs_np(input_folder, folders_to_read=c_folders,
                                                                            img_names=stream_file_names,
                                                                            ctr_names=[ctr_file_name])
                for c_folder_idx in range(batch_size):
                    if data_augmentation:
                        totalFilters = 3
                        # Making flipping
                        if np.random.random() <= (1.0/totalFilters): # Only 1/3 should be flipped
                            all_imgs[c_folder_idx,:], all_ctrs[c_folder_idx,:] = \
                                flipping(all_imgs[c_folder_idx,:], all_ctrs[c_folder_idx,:], flip_axis=3)
                        # Making random gauss (zoom)
                        if np.random.random() <= (1.0/totalFilters): # Only 1/3 should be blured
                            all_imgs[c_folder_idx,:] = gaussblur_3d(all_imgs[c_folder_idx,:])

                        # Shifting the image a little bit
                        if np.random.random() <= (1.0/totalFilters): # Only 1/3 should be blured
                            all_imgs[c_folder_idx,:], all_ctrs[c_folder_idx,:] = shifting_3d(all_imgs[c_folder_idx,:], all_ctrs[c_folder_idx,:])

                X = format_for_nn_training(all_imgs)
                Y = format_for_nn_training(all_ctrs)
                # Very useful to show we are generating the expected input for the NN
                # print(F'Input dimensions are: {np.array(X).shape}')
                # print(F'Output dimensions are: {np.array(Y).shape}')
                # self.viz_obj.plot_imgs_and_ctrs_np([X[0][0,:,:,:,0],X[1][0,:,:,:,0], X[2][0,:,:,:,0]],
                #                                    ctrs_np=[Y[0][0,:,:,:,0]],
                #                                    slices=SliceMode.MIDDLE, title=F'Generator {c_folders}')
                yield X, Y
                # Example of the required input in a 3-multistream 3D Unet
                # TestX = [np.zeros((batch_size,168,168,168,1)),np.zeros((batch_size,168,168,168,1)),np.zeros((batch_size,168,168,168,1))]
                # TestY = [np.zeros((batch_size,168,168,168,1))]
                # yield TestX, TestY

            except Exception as e:
                print("----- Not able to generate for: ", c_folders, " ERROR: ", str(e))
