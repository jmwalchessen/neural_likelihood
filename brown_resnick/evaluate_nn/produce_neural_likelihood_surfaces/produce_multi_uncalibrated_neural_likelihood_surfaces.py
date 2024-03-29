#This script produces uncalibrated neural likelihood surfaces in the multiple realization case.
import numpy as np
import tensorflow
import pickle
from tensorflow import keras

version = "final_version"

#This is the transformation of the classifier into psi (which is proportional to the likelihood)
#function parameters:
    #images: numpy matrix of spatial field
    #parameters: numpy matrix of parameters
def multi_psi(images, parameters):

    classifier_outputs = parameter_classifier.predict([images, parameters])
    psi_values = np.zeros(shape = (classifier_outputs.shape[0], 1))
    for i in range(classifier_outputs.shape[0]):
        output = classifier_outputs[i,:]
        if(output[1] == 1):
            psi_value = (1-output[0])/output[0]
        else:
            psi_value = output[1]/(1-output[1])

        psi_values[i,:] = psi_value

    return psi_values

#Produce the neural likelihood surface for the parameter grid over the parameter space
#function parameters:
    #possible_ranges: range values on the parameter grid
    #possible_smooths: smooth values on the parameter grid
    #image: spatial field (numpy matrix)
    #n: the square root of the number of spatial observations
def produce_psi_field(possible_ranges, possible_smooths, image, n):

    number_of_parameter_pairs = len(possible_ranges)*len(possible_smooths)
    image = image.reshape((1, n, n, 1))
    image_matrix = np.repeat(image, number_of_parameter_pairs, axis  = 0)
    ranges = (np.repeat(np.asarray(possible_ranges), 
                        len(possible_smooths), axis = 0)).reshape((number_of_parameter_pairs, 1))
    smooths = []
    smooths = (np.array(sum([(smooths + possible_smooths) for i in 
                             range(0, len(possible_ranges))], []))).reshape((number_of_parameter_pairs,1))
    parameter_matrix = np.concatenate([ranges, smooths], axis = 1)
    psi_field = (multi_psi(image_matrix, parameter_matrix)).reshape((len(possible_smooths), len(possible_ranges)))

    return psi_field

#Produce the neural likelihood surface in the multiple realization case for the parameter grid over the parameter space
#function parameters:
    #possible_ranges: range values on the parameter grid
    #possible_smooths: smooth values on the parameter grid
    #image_realizations: multiple spatial fields (numpy matrix)
    #n: the square root of the number of spatial observations
    #multi_number: the number of spatial fields
def produce_psi_field_for_multiple_realizations(possible_ranges, possible_smooths, image_realizations, n, 
                                                multi_number):

    multi_psi_field = np.zeros((len(possible_smooths), len(possible_ranges)))

    for i in range(0, multi_number):

        image = image_realizations[i,:,:]
        multi_psi_field = (multi_psi_field + produce_psi_field(possible_ranges, possible_smooths, image, n))

    return multi_psi_field


#Load the evaluation images
n = 25
multi_number = 5
image_size = 25
number_of_reps = 200
image_name = str(image_size) + "_by_" + str(image_size)
local_folder = "/home/juliatest/Dropbox/likelihood_free_inference/neural_likelihood/brown_resnick/"
data_file_name = (local_folder + "evaluate_nn/generate_data/data/" + image_name + 
                  "/multi/" + str(multi_number) + "/reps/" + str(number_of_reps) + 
                  "/evaluation_images_9_by_9_density_" + image_name + "_multi_" + str(multi_number)
                  + "_" + str(number_of_reps) + ".npy")
evaluation_images = np.load(data_file_name)
possible_ranges = [.05*i for i in range(1, 41)]
possible_smooths = [.05*i for i in range(1, 41)]

number_of_parameters = 81


#Load the nn
json_file_name = (local_folder + "nn/models/" + image_name + "/" + version +
"/model/br_" + image_name + "_" + version + "_nn.json")

weights_file_name = (local_folder + "nn/models/" + image_name + "/" + version +
"/model/br_" + image_name + "_" + version + "_nn_weights.h5")

json_file = open(json_file_name, 'r')
loaded_model_json = json_file.read()
json_file.close()
parameter_classifier = keras.models.model_from_json(loaded_model_json)
parameter_classifier.load_weights(weights_file_name)

psi_fields = np.zeros((number_of_parameters, number_of_reps, len(possible_smooths), len(possible_ranges)))

#Produce neural likelihood surfaces and parameter estimates for evaluation data
for i in range(0, number_of_parameters):
    for j in range(0, number_of_reps):

        current_image = evaluation_images[i,j,:,:,:]
        psi_fields[i,j,:,:] = produce_psi_field_for_multiple_realizations(possible_ranges, possible_smooths, 
                                                                          current_image, n, multi_number)


psi_field_file = (local_folder + "/evaluate_nn/produce_neural_likelihood_surfaces/data/" + image_name + 
                  "/" + version + "/uncalibrated/multi/" + str(multi_number) + "/reps/" + str(number_of_reps) +
                   "/uncalibrated_neural_likelihood_surfaces_9_by_9_density_" + image_name + "_multi_" +
                   str(multi_number) + "_" + str(number_of_reps) + ".npy")
np.save(psi_field_file, psi_fields)
