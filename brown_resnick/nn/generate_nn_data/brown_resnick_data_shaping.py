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


def process_json_files_into_numpy_matrices_for_training_nn(folder_name, file_name, file_numbers, 
                                                           number_of_replications, image_size):

    total_number = 2*len(file_numbers)*number_of_replications
    images = np.zeros(shape = (total_number, image_size, image_size, 1))
    parameters = np.zeros(shape = (total_number, 2))
    classes = np.zeros(shape = (total_number, 2))

    for i in range(0, len(file_numbers)):
        print(file_numbers[i])
        current_file_name = folder_name + "/" + file_name + "_" + str(file_numbers[i]) + ".json"
        with open(current_file_name) as f:
            current_data = json.load(f)
        for j, arr in enumerate(current_data):
            y = (np.array(arr["y"])).reshape((image_size, image_size, 1))
            images[int(2*number_of_replications*(i-1) + j),:,:,:] = y
            parameters[(2*number_of_replications*(i-1)+ j),:] = arr["parameters"]
            if((arr["class"])[0] == 1):
                #first class (dependent class)
                classes[int(2*number_of_replications*(i-1) + j),:] = np.array((0,1))
            else:
                #second class (independent class)
                classes[int(2*number_of_replications*(i-1) + j),:] = np.array((1,0))
        
    return images, parameters, classes


possible_ranges = [.05*i for i in range(1, 41)]
possible_smooths = [.05*i for i in range(1, 41)]
image_size = 25
image_name = str(image_size) + "_by_" + str(image_size)
file_numbers = [i for i in range(1, 3001)]
number_of_replications = 50
number_of_parameters = 3000
version = "final_version"
local_folder = "/home/juliatest/Dropbox/likelihood_free_inference/neural_likelihood/brown_resnick/"
folder_name = (local_folder + "nn/models/" + image_name + "/" + version + "/data/validation")
file_name = ("validation_samples_" + image_name + "_" + str(number_of_parameters) 
             + "_reps_" + str(number_of_replications))

images, params, classes = process_json_files_into_numpy_matrices_for_training_nn(folder_name, file_name,
                                                                                 file_numbers, number_of_replications,
                                                                                 image_size)


#For large datasets, need to split into multiple files for ease of access
#splits = 20
#per_split = number_of_parameters*number_of_replications/splits

#for isplit in range(0,splits):
    #print(isplit)
    #current_split_images = images[int(isplit*per_split):int((isplit+1)*per_split),:,:]
    #current_split_params = params[int(isplit*per_split):int((isplit+1)*per_split),:]
    #current_split_classes = classes[int(isplit*per_split):int((isplit+1)*per_split),:]

    #images_file = (local_folder + "nn/" + image_name + "/" + version + 
                   #"/data/validation/validation_images_" + image_name + "_" + 
                   #str(number_of_parameters) + "_reps_" + 
                   #str(number_of_replications) + "_" + str(isplit) + ".npy")
    #parameters_file = (local_folder + "nn/" + image_name + "/" + version + 
                   #"/data/validation/validation_parameters_" + image_name + "_" + 
                   #str(number_of_parameters) + "_reps_" 
                   #+ str(number_of_replications) + "_" + str(isplit) + ".npy")
    #classes_file = (local_folder + "nn/" + image_name + "/" + version + 
                   #"/data/validation/validation_classes_" + image_name + "_" + 
                   #str(number_of_parameters) + "_reps_" 
                   #+ str(number_of_replications) + "_" + str(isplit) + ".npy")
    #np.save(images_file, current_split_images)
    #np.save(parameters_file, current_split_params)
    #np.save(classes_file, current_split_classes)

#For smaller files, save images, parameters, and classes as one image
images_file = (local_folder + "nn/models/" + image_name + "/" + version + 
                  "/data/validation/validation_images_" + image_name + "_" +
                  str(number_of_parameters) + "_reps_" + str(number_of_replications)
                  + ".npy")
parameters_file = (local_folder + "nn/models/" + image_name + "/" + version + 
                  "/data/validation/validation_parameters_" + image_name + "_" +
                  str(number_of_parameters) + "_reps_" + str(number_of_replications)
                  + ".npy")
classes_file = (local_folder + "nn/models/" + image_name + "/" + version + 
                  "/data/validation/validation_classes_" + image_name + "_" +
                  str(number_of_parameters) + "_reps_" + str(number_of_replications)
                  + ".npy")
np.save(images_file, images)
np.save(parameters_file, params)
np.save(classes_file, classes)
