#This file is for timing how long it takes to produce a neural likelihood surface (both vectorized and unvectorized)
import numpy as np
import time
import json
import tensorflow
from tensorflow import keras
import ray

physical_devices = tensorflow.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
   tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

image_size = 25
image_name = str(image_size) + "_by_" + str(image_size)
version = "final_version"
local_folder = "/home/juliatest/Dropbox/likelihood_free_inference/neural_likelihood/brown_resnick"

json_file_name = (local_folder + "/nn/models/" + image_name + "/" + version + 
                  "/model/br_" + image_name + "_" + version + "_nn.json")

weights_file_name = (local_folder + "/nn/models/" + image_name + "/" + version +
                      "/model/br_" + image_name + "_" + version + "_nn_weights.h5")

json_file = open(json_file_name, 'r')
loaded_model_json = json_file.read()
json_file.close()
parameter_classifier = keras.models.model_from_json(loaded_model_json)
parameter_classifier.load_weights(weights_file_name)

#Apply the classifier to a single image and parameter 
#and transform the classifier output via psi
def psi(image, parameters):
    image = image.reshape((1,25,25,1))
    classifier_output = parameter_classifier.predict([image, parameters])
    psi_values = np.zeros(shape = (classifier_output.shape[0], 1))
    for i in range(classifier_output.shape[0]):
        output = classifier_output[i,:]
        if(output[1] == 1):
            psi_value = (1-output[0])/output[0]
        else:
            psi_value = output[1]/(1-output[1])

    psi_values[i,:] = psi_value
        
    return psi_values

#Apply the classifier to multiple images and parameters 
#and transform the classifier output via psi
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

#Produce the uncalibrated neural likelihood surface using the psi function for
#a single pair of image and parameter
def produce_unvectorized_neural_likelihood_surface(possible_ranges, possible_smooths, image, n):

    neural_likelihood_surface = np.zeros((len(possible_ranges), len(possible_smooths)))

    for i in range(len(possible_ranges)):
        current_length_scale = possible_ranges[i]
        for j in range(len(possible_smooths)):
            current_smooth = possible_smooths[j]
            current_params = (np.array([current_length_scale, current_smooth])).reshape((1,2))
            neural_likelihood_surface[i,j] = multi_psi(image.reshape((1, n, n, 1)), current_params)

    return neural_likelihood_surface

#Produce the uncalibrated neural likelihood surface using the psi function for
#multiple pairs of images and parameters
def produce_vectorized_neural_likelihood_surface(possible_ranges, possible_smooths, image, n):

    number_of_parameter_pairs = len(possible_ranges)*len(possible_smooths)
    image = image.reshape((1, n, n, 1))
    image_matrix = np.repeat(image, number_of_parameter_pairs, axis  = 0)
    ranges = (np.repeat(np.asarray(possible_ranges), len(possible_smooths), 
                        axis = 0)).reshape((number_of_parameter_pairs, 1))
    smooths = []
    smooths = (np.array(sum([(smooths + possible_smooths) for i 
                             in range(0, len(possible_ranges))], []))).reshape((number_of_parameter_pairs,1))
    parameter_matrix = np.concatenate([ranges, smooths], axis = 1)
    psi_field = (multi_psi(image_matrix, parameter_matrix)).reshape((len(possible_smooths), 
                                                                     len(possible_ranges)))

    return psi_field

#Load evaluation data. Timing study 
n = 25
total_number_of_reps = 200
evaluation_data_file_name = (local_folder + "/evaluate_nn/generate_data/data/" + image_name + 
                  "/single/reps/" + str(total_number_of_reps) + "/evaluation_images_10_by_10_density_"
                  + image_name + "_" + str(total_number_of_reps) + ".npy")
evaluation_data = np.load(evaluation_data_file_name)
possible_length_scales = [.05*i for i in range(1, 41)]
possible_variances = [.05*i for i in range(1, 41)]

#Time how long it takes to generate each neural likelihood surface for 50 realizations.
number_of_parameters = 100
number_of_reps = 50
ipred = 33

time_array = np.zeros((number_of_reps))

#Parallelize producing neural likelihood surface (produce neural likelihood
# for each row in the surface on a separate core) using unvectorized method
@ray.remote
def inner_for_loop_for_unvectorized_neural_likelihood_surface(input):

    current_length_scale = input[0]
    possible_variances = input[1]
    image = input[2]

    neural_likelihood_surface_row = np.zeros((len(possible_variances)))
    for j in range(len(possible_variances)):
        current_variance = possible_variances[j]
        current_params = (np.array([current_length_scale, current_variance])).reshape((1,2))
        neural_likelihood_surface_row[j] = psi(image, current_params)

    return neural_likelihood_surface_row


def produce_parallelized_unvectorized_neural_likelihood_surface(operation, inputs):
   return ray.get([operation.remote(input) for input in inputs])

#For each of the fifty realizations, time how long it takes to produce each
#neural likelihood surface using the unvectorized method and all cores on laptop
for irep in range(number_of_reps):

    possible_variances = [.05*i for i in range(1,41)]
    ray.init()
    inputs = [(.05*i, possible_variances, evaluation_data[ipred,irep,:,:]) for i in range(1, 41)]
    start = time.time()
    output = produce_parallelized_unvectorized_neural_likelihood_surface(
        inner_for_loop_for_unvectorized_neural_likelihood_surface, inputs)
    end = time.time()
    time_array[irep] = (end - start)
    ray.shutdown()

unvectorized_time_array_filename = (local_folder + "/evaluate_nn/timing_studies/data/" + image_name + 
                        "/neural/unvectorized_neural_likelihood_surface_time_with_parallelization_on_laptop_"
                        + str(ipred) + ".npy")
np.save(unvectorized_time_array_filename, time_array)


#Parallelize producing mulitple neural likelihood surfaces. Each core
#produces a neural likelihood surface via vectorized method.
def time_vectorized_neural_likelihood_surface(image):

    start = time.time()
    produce_vectorized_neural_likelihood_surface(image)
    end = time.time()
    time_difference = (end - start)
    return time_difference

def produce_parallelized_vectorized_neural_likelihood_surfaces(operation, inputs):
   return ray.get([operation.remote(input) for input in inputs])

irep = 33

ray.init()
inputs = [(evaluation_data[ipred,irep,:,:]) for irep in range(0, number_of_reps)]
output = produce_parallelized_vectorized_neural_likelihood_surfaces(time_vectorized_neural_likelihood_surface, inputs)
ray.shutdown()

output = (np.asarray(output))
vectorized_time_array_filename = (local_folder + "/evaluate_nn/timing_studies/data/"+ image_name + 
                                  "/neural/vectorized_neural_likelihood_surface_time_with_parallelization_on_laptop_33.npy")
np.save(vectorized_time_array_filename, output)
