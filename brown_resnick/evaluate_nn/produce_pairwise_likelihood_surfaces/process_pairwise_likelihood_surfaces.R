#This script is for processing the numpy matrices produce by produce_pairwise_likelihood_surfaces_for_evaluation_data_in_the_x_case.R into
#one numpy matrix.
library(reticulate)

local_folder <- "/home/juliatest/Desktop/likelihood_free_inference/neural_likelihood/brown_resnick/evaluate_nn/produce_pairwise_likelihood_surfaces"
number_of_replications <- 200
image_size <- 25
image_name <- paste(paste(as.character(image_size), "by", sep = "_"), as.character(image_size), sep = "_")
spatial_domain_size <- 20
distance_constraint <- 2
ranges = seq(.2, 2, .2)
smooths = seq(.2, 2, .2)
number_of_parameters <- 100
number_of_multiple_realizations <- 5
parameter_size <- 40

partial_single_pairwise_likelihood_surfaces_file_name <- paste(paste(paste(paste(paste(paste(paste(paste(paste(paste(paste(local_folder, 
                                                "data", sep = "/"), image_name, sep = "/"), "dist", sep = "/"), as.character(distance_constraint), sep = "_"), 
                                                sep = "/"), "single/reps", sep = "/"), as.character(number_of_replications), sep = "/"), 
                                                "pairwise_likelihood_surfaces_10_by_10_density", sep = "/"), image_name, sep = "_"), "image", sep = "_"),
                                                as.character(number_of_replications), sep = "_")
partial_multi_pairwise_likelihood_surfaces_file_name <- paste(paste(paste(paste(paste(paste(paste(paste(paste(paste(paste(paste(
                                                paste(paste(paste(paste(local_folder, "data", sep = "/"), image_name, sep = "/"), "dist", sep = "/"), 
                                                as.character(distance_constraint), sep = "_"), sep = "/"), "multi", sep = "/"), 
                                                as.character(number_of_multiple_realizations), sep = "/"), sep = "/"), "reps", sep = "/"),
                                                as.character(number_of_replications), sep = "/"), "pairwise_likelihood_surfaces_10_by_10_density", sep = "/"),
                                                image_name, sep = "_"), "image", sep = "_"), "multi", sep = "_"), as.character(number_of_multiple_realizations),
                                                sep = "_"), as.character(number_of_replications), sep = "_")
np <- import("numpy")


pairwise_likelihood_surfaces_in_single_realization_case <- array(0, dim = c(number_of_parameters, number_of_replications, parameter_size, parameter_size))
pairwise_likelihood_surfaces_in_multi_realization_case <- array(0, dim = c(number_of_parameters, number_of_replications, number_of_multiple_realizations,
                                                                           parameter_size, parameter_size))
for(ipred in 1:number_of_parameters)
{
  single_pairwise_likelihood_surfaces_file_name_per_parameter <- paste(paste(partial_single_pairwise_likelihood_surfaces_file_name,
                                                                       as.character(ipred), sep = "_"), "npy", sep = ".")
  multi_pairwise_likelihood_surfaces_file_name_per_parameter <- paste(paste(partial_multi_pairwise_likelihood_surfaces_file_name,
                                                                             as.character(ipred), sep = "_"), "npy", sep = ".")
  pairwise_likelihood_surfaces_in_single_realization_case[ipred,,,] <- np$load(single_pairwise_likelihood_surfaces_file_name_per_parameter)
  pairwise_likelihood_surfaces_in_multi_realization_case[ipred,,,,] <- np$load(multi_pairwise_likelihood_surfaces_file_name_per_parameter)
}

pairwise_likelihood_surfaces_in_single_realization_case_file_name <- paste(partial_single_pairwise_likelihood_surfaces_file_name, "npy", sep = ".")
pairwise_likelihood_surfaces_in_multi_realization_case_file_name <- paste(partial_multi_pairwise_likelihood_surfaces_file_name, "npy", sep = ".")
np$save(pairwise_likelihood_surfaces_in_single_realization_case_file_name, pairwise_likelihood_surfaces_in_single_realization_case)
np$save(pairwise_likelihood_surfaces_in_multi_realization_case_file_name, pairwise_likelihood_surfaces_in_multi_realization_case)