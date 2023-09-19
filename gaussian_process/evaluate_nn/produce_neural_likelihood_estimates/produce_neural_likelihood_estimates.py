#This script takes the neural likelihood surfaces numpy matrix in both the single and multiple realization case and 
#finds the parameter on the grid over the parameter space which maximizes the field. This parameter becomes the
#parameter estimate. Note that calibrated and uncalibrated neural likelihood estimators are the same so we simply
#use the uncalibrated neural likelihood surfaces to produce the neural likelihood parameter estimates.
import numpy as np

#Find the parameters which maximize the neural likelihood field
#function parameters:
    #possible_length_scales: length scales values on the parameter grid
    #possible_variances: variance values on the parameter grid
    #psi_field: neural likelihood surface (numpy matrix)
def produce_max_psi_parameters(possible_length_scales, possible_variances, psi_field):

    max_indices = np.unravel_index(np.argmax(psi_field, axis=None), psi_field.shape)
    max_length_scale = possible_length_scales[max_indices[1]]
    max_variance = possible_variances[max_indices[0]]



    return np.array([max_length_scale, max_variance])

number_of_parameters = 81
number_of_reps = 200
parameter_length = 40
possible_variances = [.05*i for i in range(1, 41)]
possible_length_scales = [.05*i for i in range(1, 41)]
image_name = "25_by_25"
version = "final_version"
multi_number = 5

local_folder = "/home/juliatest/Dropbox/likelihood_free_inference/neural_likelihood/gaussian_process/"
multi_psi_fields_file = (local_folder + "evaluate_nn/produce_neural_likelihood_surfaces/data/" + image_name + "/" + 
                         version + "/uncalibrated/multi/" + str(multi_number) + "/reps/" + str(number_of_reps) + 
                         "/uncalibrated_neural_likelihood_surfaces_9_by_9_density_" + image_name +
                         "_multi_" + str(multi_number)+ "_" + str(number_of_reps) + ".npy")
multi_psi_fields = np.load(multi_psi_fields_file)

single_psi_fields_file = (local_folder + "evaluate_nn/produce_neural_likelihood_surfaces/data/" + image_name + 
                          "/" + version + "/uncalibrated/single/reps/" + str(number_of_reps) + 
                          "/uncalibrated_neural_likelihood_surfaces_9_by_9_density_" + image_name +  "_image_" 
                          + str(number_of_reps) + ".npy")
single_psi_fields = np.load(single_psi_fields_file)


single_max_params = np.zeros((number_of_parameters, number_of_reps, 2))

#Produce neural likelihood surfaces and parameter estimates for evaluation images in the single realization case
for i in range(0, number_of_parameters):
    for j in range(0, number_of_reps):

        single_max_params[i,j,:] = produce_max_psi_parameters(possible_length_scales, possible_variances, 
                                                              single_psi_fields[i,j,:,:])


single_max_params_file = (local_folder + "evaluate_nn/produce_neural_likelihood_estimates/data/" + image_name 
                          + "/" + version + "/single/reps/" + str(number_of_reps) + 
                          "/neural_likelihood_estimators_9_by_9_density_" + image_name + "_image_" + 
                          str(number_of_reps) + ".npy")

np.save(single_max_params_file, single_max_params)

multi_max_params = np.zeros((number_of_parameters, number_of_reps, 2))

#Produce neural likelihood surfaces and parameter estimates for evaluation images in the multiple realization case
for i in range(0, number_of_parameters):
    for j in range(0, number_of_reps):

        multi_max_params[i,j,:] = produce_max_psi_parameters(possible_length_scales, possible_variances, 
                                                              multi_psi_fields[i,j,:,:])

multi_max_params_file = (local_folder + "evaluate_nn/produce_neural_likelihood_estimates/data/" + 
                         image_name + "/" + version + "/multi/" + str(multi_number) + "/reps/" + 
                          str(number_of_reps) + "/neural_likelihood_estimators_9_by_9_density_" + image_name 
                          + "_multi_" + str(multi_number) + "_" + str(number_of_reps) + ".npy")
np.save(multi_max_params_file, multi_max_params)
