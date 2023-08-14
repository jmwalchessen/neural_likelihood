#This python script is for processing the json files produced from generate_nn_data.R into numpy matrices for training
#the neural network
import json
import numpy as np

#main function: Process json files (one per sampled parameter in the parameter space) into numpy matrices. Specifically,
#the function produces three matrices--one each for parameters, spatial fields, and classes.
#function parameters:
    #folder_name: location of the json files
    #file_name: name of json files
    #file_numbers: list of numbers of the json files (the total number of json files is equal to the number of sampled)
                #parameters)
    #number_of_replications: number of realizations of the Gaussian process in each json file (per parameter)
    #image_size: the square root of the number of spatial observations at which the Gaussian Process is observed,
                #the number of pixels in the spatial field is image_size x image_size
def process_json_files_into_numpy_matrices_for_training_nn(folder_name, file_name, file_numbers, number_of_replications, 
                                                           image_size):

    total_number = 2*len(file_numbers)*number_of_replications
    images = np.zeros(shape = (total_number, image_size, image_size, 1))
    parameters = np.zeros(shape = (total_number, 2))
    classes = np.zeros(shape = (total_number, 2))

    for i in range(0, len(file_numbers)):
        current_file_name = folder_name + "/" + file_name + "_" + str(file_numbers[i]) + ".json"
        with open(current_file_name) as f:
            current_data = json.load(f)
        for j, arr in enumerate(current_data):
            y = (np.array(arr["y"])).reshape((image_size, image_size, 1))
            images[(2*number_of_replications*(i-1) + j),:,:,:] = y
            parameters[(2*number_of_replications*(i-1)+ j),:] = arr["parameters"]
            if((arr["class"])[0] == 1):
                #first class (dependent class)
                classes[(2*number_of_replications*(i-1) + j),:] = np.array((0,1))
            else:
                #second class (independent class)
                classes[(2*number_of_replications*(i-1) + j),:] = np.array((1,0))
        
    return images, parameters, classes

#in our case, the image size is 25 because the spatial field has 25 by 25 pixels/grid points
image_size = 25
#number of replications of the Gaussian process for each parameter
number_of_replicates = 50
#total number of sampled parameters for training neural network (validation)
parameter_number = 3000
#list of numbers which denote each json file
file_numbers = [i for i in range(1, parameter_number)]
#location of where json files are located
folder_name = "/home/juliatest/Desktop/likelihood_free_inference/neural_likelihood/gaussian_process/nn/models/25_by_25/final_version/data/validation"
#
file_name = "validation_data_expanded_parameter_space_25_by_25_3000_rep_50"

images, parameters, classes = process_json_files_into_numpy_matrices_for_training_nn(folder_name, 
file_name, file_numbers, number_of_replicates, image_size)

#names to use to save numpy matrices
file_name_for_images_matrix = (folder_name + "/validation_images_expanded_parameter_space_25_by_25_3000_reps_50.npy")
file_name_for_parameters_matrix = (folder_name + "validation_parameters_expanded_parameter_space_25_by_25_3000_reps_50.npy")
file_name_for_classes_matrix = (folder_name + "/validation_classes_expanded_parameter_space_25_by_25_3000_reps_50.npy")
np.save(file_name_for_images_matrix, images)
np.save(file_name_for_parameters_matrix, parameters)
np.save(file_name_for_classes_matrix, classes)


