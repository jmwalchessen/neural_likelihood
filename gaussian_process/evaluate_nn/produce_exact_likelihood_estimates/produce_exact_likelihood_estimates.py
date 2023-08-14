#This script takes the exact log likelihood fields numpy matrix and finds the parameter on the grid over the parameter space
#which maximizes the field for both single and multiple realization case. This parameter becomes the parameter estimate.
import numpy as np

image_size = 25
image_name = str(image_size) + "_by_" + str(image_size)
local_folder = "/home/juliatest/Desktop/likelihood_free_inference/neural_likelihood/gaussian_process/"
ll_file_name = (local_folder + "evaluate_nn/generate_data/data/" + image_name + 
                "/multi/5/reps/200/evaluation_ll_fields_10_by_10_density_" + image_name + "_multi_5_200.npy")
ll_fields = np.load(ll_file_name)

#Find the parameters which maximize the log likelihood field
def produce_max_ll_parameters(possible_length_scales, possible_variances, ll_field):

    max_indices = np.unravel_index(np.argmax(ll_field, axis=None), ll_field.shape)
    max_length_scale = possible_length_scales[max_indices[1]]
    max_variance = possible_variances[max_indices[0]]

    return np.array([max_length_scale, max_variance])

number_of_parameters = 100
number_of_reps = 200
possible_ranges = [.05*i for i in range(1, 41)]
possible_smooths = [.05*i for i in range(1, 41)]
max_params = np.zeros((number_of_parameters, number_of_reps, 2))

for i in range(0, number_of_parameters):

    for j in range(0, number_of_reps):

        max_params[i,j,:] = produce_max_ll_parameters(possible_ranges, possible_smooths, ll_fields[i,j,:,:])

numpy_file_name = ("data/" + image_name + 
                   "/multi/5/reps/200/evaluation_ll_estimators_10_by_10_image_" + image_name + "_multi_5_200.npy")
np.save(numpy_file_name, max_params)
