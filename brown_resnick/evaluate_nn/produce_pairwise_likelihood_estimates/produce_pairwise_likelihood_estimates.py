#This script takes the pairwise likelihood fields numpy matrix and finds the parameter on the grid over the parameter space
#which maximizes the field for both single and multiple realization case. This parameter becomes the parameter estimate.
import numpy as np

image_size = 25
image_name = str(image_size) + "_by_" + str(image_size)
number_of_reps = 200
multi_number = 5
distance_constraint = 1
local_folder = "/home/juliatest/Dropbox/likelihood_free_inference/neural_likelihood/brown_resnick/"
pairwise_likelihood_surfaces_in_single_realization_case_file_name = (local_folder + 
                                                "evaluate_nn/produce_pairwise_likelihood_surfaces/unadjusted/data/"
                                                + image_name + "/dist_" + str(distance_constraint) + 
                                                "/single/reps/" + str(number_of_reps) + 
                                                "/pairwise_likelihood_surfaces_10_by_10_density_" 
                                                + image_name + "_image_" + str(number_of_reps)
                                                + ".npy")
single_pairwise_likelihood_fields = np.load(pairwise_likelihood_surfaces_in_single_realization_case_file_name)
pairwise_likelihood_surfaces_in_multiple_realization_case_file_name = (local_folder + 
                                                "evaluate_nn/produce_pairwise_likelihood_surfaces/unadjusted/data/"
                                                + image_name + "/dist_" + str(distance_constraint) + "/multi/" 
                                                + str(multi_number) + "/reps/" + str(number_of_reps) + 
                                                "/pairwise_likelihood_surfaces_10_by_10_density_" + image_name +
                                                "_image_multi_" + str(multi_number) + "_" + str(number_of_reps)
                                                + ".npy")
multi_pairwise_likelihood_fields = np.load(pairwise_likelihood_surfaces_in_multiple_realization_case_file_name)

#Find the parameters which maximize the pairwise likelihood field
def produce_max_pairwise_likelihood_parameters(possible_ranges, possible_smooths, pairwise_likelihood_field):

    max_indices = np.unravel_index(np.argmax(pairwise_likelihood_field, axis=None), pairwise_likelihood_field.shape)
    max_range = possible_ranges[max_indices[0]]
    max_smooth = possible_smooths[max_indices[1]]

    return np.array([max_range, max_smooth])

number_of_parameters = 100
number_of_reps = 200
possible_ranges = [.05*i for i in range(1, 41)]
possible_smooths = [.05*i for i in range(1, 41)]
single_max_params = np.zeros((number_of_parameters, number_of_reps, 2))
multi_max_params = np.zeros((number_of_parameters, number_of_reps, 2))

for ipred in range(0, number_of_parameters):

    for irep in range(0, number_of_reps):

        single_max_params[ipred,irep,:] = produce_max_pairwise_likelihood_parameters(possible_ranges, possible_smooths, 
                                                                       single_pairwise_likelihood_fields[ipred,irep,:,:])
        multi_max_params[ipred,irep,:] = produce_max_pairwise_likelihood_parameters(possible_ranges, possible_smooths, 
                                                                       multi_pairwise_likelihood_fields[ipred,irep,:,:])

single_max_params_file_name = ("data/" + image_name + "/dist_" + str(distance_constraint) +
                   "/single/reps/" + str(number_of_reps) +
                   "/evaluation_pairwise_likelihood_estimators_10_by_10_image_" + 
                   image_name + "_" + str(number_of_reps) + ".npy")
multi_max_params_file_name = ("data/" + image_name + "/dist_" + str(distance_constraint) + 
                   "/multi/" + str(multi_number) + "/reps/" + str(number_of_reps) +
                   "/evaluation_pairwise_likelihood_estimators_10_by_10_image_" + 
                   image_name + "_multi_" + str(multi_number) + "_" + str(number_of_reps) + ".npy")
np.save(single_max_params_file_name, single_max_params)
np.save(multi_max_params_file_name, multi_max_params)
