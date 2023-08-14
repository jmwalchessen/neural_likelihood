#This script produces uncalibrated neural likelihood surfaces in the single realization case.
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
    #possible_length_scales: length scales values on the parameter grid
    #possible_variances: variance values on the parameter grid
    #image: spatial field (numpy matrix)
    #n: the square root of the number of spatial observations
def produce_psi_field(possible_length_scales, possible_variances, image, n):

    number_of_parameter_pairs = len(possible_length_scales)*len(possible_variances)
    image = image.reshape((1, n, n, 1))
    image_matrix = np.repeat(image, number_of_parameter_pairs, axis  = 0)
    length_scales = (np.repeat(np.asarray(possible_length_scales), len(possible_variances), axis = 0)).reshape((number_of_parameter_pairs, 1))
    variances = []
    variances = (np.array(sum([(variances + possible_variances) for i in range(0, len(possible_length_scales))], []))).reshape((number_of_parameter_pairs,1))
    parameter_matrix = np.concatenate([length_scales, variances], axis = 1)
    psi_field = (multi_psi(image_matrix, parameter_matrix)).reshape((len(possible_variances), len(possible_length_scales)))

    return psi_field

#Load the evaluation images
n = 25
local_folder = "/home/juliatest/Desktop/likelihood_free_inference/neural_likelihood/gaussian_process/"
data_file_name = (local_folder + "evaluate_nn/generate_data/data/" + str(n) + "_by_" + str(n) + 
                  "/single/reps/200/evaluation_images_10_by_10_density_25_by_25_200.npy")
evaluation_images = np.load(data_file_name)
possible_length_scales = [.05*i for i in range(1, 41)]
possible_variances = [.05*i for i in range(1, 41)]

number_of_parameters = 100
number_of_reps = 200

image_size = 25
image_name = str(image_size) + "_by_" + str(image_size)

#Load the nn
json_file_name = (local_folder + "nn/models/" + image_name + "/" + version +
"/model/gp_" + image_name + "_" + version + "_nn.json")

weights_file_name = (local_folder + "nn/models/" + image_name + "/" + version +
"/model/gp_" + image_name + "_" + version + "_nn_weights.h5")

json_file = open(json_file_name, 'r')
loaded_model_json = json_file.read()
json_file.close()
parameter_classifier = keras.models.model_from_json(loaded_model_json)
parameter_classifier.load_weights(weights_file_name)

psi_fields = np.zeros((number_of_parameters, number_of_reps, 40, 40))

#Produce neural likelihood surfaces and parameter estimates for evaluation images
for i in range(0, number_of_parameters):
    for j in range(0, number_of_reps):

        current_image = evaluation_images[i,j,:,:]
        psi_fields[i,j,:,:] = produce_psi_field(possible_length_scales, possible_variances, current_image, n)


psi_field_file = ("data/" + image_name + "/" + version +
"/uncalibrated/single/reps/200/uncalibrated_neural_likelihood_surfaces_10_by_10_density_" + image_name + "_image_200.npy")
np.save(psi_field_file, psi_fields)
