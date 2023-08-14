import json
import numpy as np

#Process json files (one per parameter from the 10 by 10 grid over the parameter space for evaluation) into
#numpy matrices. Specifically, the function outputs three matrices--one each for spatial fields, the corresponding
#parameters which produced the spatial fields, and the log likelihood fields for the spatial fields.
#function parameter: 
    #folder_name: location of the json files
    #file_name: name of json files
    #file_numbers: list of numbers of the json files (the total number of json files is equal to the number of parameters)
    #number_of_parameters: the total parameters to be evaluated (in our case, 100 for 10 by 10 grid over the 
                        #parameter space)
    #number_of_replications: number of realizations of the Gaussian process in each json file (per parameter)
    #image_size: the square root of the number of spatial observations at which the Gaussian Process is observed,
                #the number of pixels in the spatial field is image_size x image_size
    #parameter_length: the square root of the number of grid points (parameter_length by parameter_length) the log
                    #likelihood field is evaluated on (in our case, 40 by 40)
def process_json_files_into_numpy_matrices_for_evaluating_nn(folder_name, file_name, file_numbers, number_of_parameters, 
                                                             number_of_replications, image_size, parameter_length):

    images = np.zeros(shape = (number_of_parameters, number_of_replications, image_size, image_size, 1))
    parameters = np.zeros(shape = (number_of_parameters, number_of_replications, 2))
    log_likelihood_fields = np.zeros(shape = (number_of_parameters, number_of_replications, parameter_length, 
                                              parameter_length))

    for i in range(0, len(file_numbers)):
        current_file_name = folder_name + "/" + file_name + "_" + str(file_numbers[i]) + ".json"
        with open(current_file_name) as f:
            current_data = json.load(f)
        for j, arr in enumerate(current_data):
            y = (np.array(arr["y"])).reshape((image_size, image_size, 1))
            ll = (np.array(arr["log_likelihood_field"])).reshape((parameter_length, parameter_length))

            images[i,j,:,:,:] = y
            parameters[i,j,:] = arr["parameters"]
            log_likelihood_fields[i,j,:,:] = ll
        
    return images, parameters, log_likelihood_fields

#Process json files (one per parameter from the 10 by 10 grid over the parameter space for evaluation in the multiple 
# realization case) into numpy matrices. Specifically, the function outputs three matrices--one each for spatial 
# fields, the corresponding parameters which produced the spatial fields, and the log likelihood fields for the 
# spatial fields.
#function parameter: 
    #folder_name: location of the json files
    #file_name: name of json files
    #file_numbers: list of numbers of the json files (the total number of json files is equal to the number of parameters)
    #number_of_parameters: the total parameters to be evaluated (in our case, 100 for 10 by 10 grid over the 
                        #parameter space)
    #number_of_replications: number of realizations of the Gaussian process in each json file (per parameter)
    #image_size: the square root of the number of spatial observations at which the Gaussian Process is observed,
                #the number of pixels in the spatial field is image_size x image_size
    #parameter_length: the square root of the number of grid points (parameter_length by parameter_length) the log
                    #likelihood field is evaluated on (in our case, 40 by 40)
    #multi_number: the number of realizations/spatial fields (for the multiple realization case)
def process_json_files_into_numpy_matrices_for_evaluating_nn_multiple_realizations(folder_name, file_name, file_numbers, 
                                                                                   number_of_parameters, 
                                                                                   number_of_replications, image_size, 
                                                                                   parameter_length, multi_number):

    images = np.zeros(shape = (number_of_parameters, number_of_replications, multi_number, image_size, image_size, 1))
    parameters = np.zeros(shape = (number_of_parameters, number_of_replications, 2))
    multi_log_likelihood_fields = np.zeros(shape = (number_of_parameters, number_of_replications, parameter_length, 
                                                    parameter_length))

    for i in range(0, len(file_numbers)):
        current_file_name = folder_name + "/" + file_name + "_" + str(file_numbers[i]) + ".json"
        with open(current_file_name) as f:
            current_data = json.load(f)
        for j, arr in enumerate(current_data):
            multi_y = (np.array(arr["multi_y"]))[0:multi_number,:,:]
            multi_y = multi_y.reshape((multi_number, image_size, image_size, 1))
            multi_ll = (np.array(arr["multi_log_likelihood_field"])).reshape((parameter_length, parameter_length))

            images[i,j,:,:,:,:] = multi_y
            parameters[i,j,:] = arr["parameters"]
            multi_log_likelihood_fields[i,j,:,:] = multi_ll
        
    return images, parameters, multi_log_likelihood_fields

number_of_replications = 200
image_size = 25
folder_name = "data/25_by_25/ll/single/reps/" + str(number_of_replications)
file_name = "data_10_by_10_density_25_by_25_image_" + str(number_of_replications)
file_numbers = [i for i in range(1, 101)]
number_of_parameters = 100
parameter_length = 40
images, parameters, log_likelihood_fields = process_json_files_into_numpy_matrices_for_evaluating_nn(folder_name, file_name,
                                                                                                      file_numbers, 
                                                                                                      number_of_parameters,
                                                                                                      number_of_replications,
                                                                                                      image_size, 
                                                                                                      parameter_length)

image_file_name = ("data/25_by_25/ll/single/reps/" + str(number_of_replications) + 
                   "/evaluation_images_10_by_10_density_25_by_25_" + str(number_of_replications) + ".npy")
np.save(image_file_name, images)
log_likelihood_file_name = ("data/25_by_25/ll/single/reps/" + str(number_of_replications) + 
                   "/evaluation_ll_fields_10_by_10_density_25_by_25_" + str(number_of_replications) + ".npy")
np.save(log_likelihood_file_name, log_likelihood_fields)
parameters_file_name = ("data/25_by_25/ll/single/reps/" + str(number_of_replications) + 
                   "/evaluation_parameters_10_by_10_density_25_by_25_" + str(number_of_replications) + ".npy")
np.save(parameters_file_name, parameters)