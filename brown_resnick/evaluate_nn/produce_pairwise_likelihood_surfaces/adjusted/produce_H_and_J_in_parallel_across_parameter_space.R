#This script computes estimates for H (the expectation of the hessian) and J (the covariance of the gradient) as described in our paper (see appendix) and 
# the paper "Inference for Clustered Data using Independent Loglikelihood" by Chandler and Bate
source("likelihood_adjustment_functions.R")
library(parallel)
library(reticulate)


#Information necessary to load the spatial field realizations for each parameter on the 9 by 9 grid over the parameter space
np <- import("numpy")
folder_name = "/home/juliatest/Dropbox/likelihood_free_inference/neural_likelihood/brown_resnick/evaluate_nn/produce_pairwise_likelihood_surfaces/adjusted/data/simulated_data/25_by_25"
#compile spatial image numpy matrices together
spatial_images_file <- paste(folder_name, "spatial_images_9_by_9_density", sep = "/")
local_folder <- "/home/juliatest/Dropbox/likelihood_free_inference/neural_likelihood/brown_resnick"
n_params <- 81
n_reps <- 5000
image_size <- 25

#Construct a matrix of the parameters on the 10 by 10 grid over the parameter space
range_test = seq(.2, 1.8, .2)
smooth_test = seq(.2, 1.8, .2)

data_y = cbind(expand.grid(range_test, smooth_test)$Var1,
               expand.grid(range_test, smooth_test)$Var2)
parameter_matrix <- matrix(NA, nrow = 81, ncol = 2)
parameter_matrix[,1] <- data_y[,1]
parameter_matrix[,2] <- data_y[,2]

#This function computes estimate of H and J per parameter on the 9 by 9 grid
  #function parameters:
    #ipred: the index of the parameter in the parameter_matrix
    #spatial_images_file_name: name of the file for the numpy matrix of spatial fields for the given parameter
    #theta: the parameter we are evaluating the gradient/hessian at
    #h: the difference used for finite differencing to compute the hessian and gradient
    #dist_constraint: the distance constraint delta used for pairwise likelihood function
    #m: the total number of spatial fields per parameter
compute_H_and_J_per_parameter <- function(ipred, spatial_images_file_name, theta, h, dist_constraint, m)
{
  np <- import("numpy")
  current_spatial_images_name <- paste(paste(spatial_images_file_name, as.character(ipred), sep = "_"), "5000.npy", sep = "_")
  current_spatial_images <- array_reshape(np$load(current_spatial_images_name), dim = c(5000, 25, 25))
  
  H_and_J_list <- compute_H_with_different_deltas_and_J(current_spatial_images, theta, h, dist_constraint, m)
  H_file_name <- paste(paste(paste(local_folder, "evaluate_nn/produce_pairwise_likelihood_surfaces/adjusted/data/matrices/dist_2/h/5e02/H/HI/HI_matrix_dist_2_5000", 
                                   sep = "/"),
                             as.character(ipred), sep = "_"), "npy", sep = ".")
  J_file_name <- paste(paste(paste(local_folder, "evaluate_nn/produce_pairwise_likelihood_surfaces/adjusted/data/matrices/dist_2/h/5e02/J/J_matrix_dist_2_5000", 
                                   sep = "/"), as.character(ipred), sep = "_"), "npy", sep = ".")

  np$save(H_file_name, H_and_J_list[[1]])
  np$save(J_file_name, H_and_J_list[[2]])
}


h <- .05
dist_constraint <- 2
m <- 5000
number_of_parameters <- 81

#Use parallel computing to compute each H and J per parameter across 10 by 10 grid
cores <- (((detectCores(logical = TRUE))))
cluster <- makeCluster(cores)
clusterCall(cluster, function() library(SpatialExtremes))
clusterExport(cluster, c("dist_constraint", "compute_hessian_with_different_deltas", "compute_gradient", "generate_pairwise_likelihood", 
                         "compute_H_with_different_deltas_and_J", "h", "flatten", "compute_H_and_J_per_parameter", "import", "parameter_matrix", 
                         "spatial_images_file", "array_reshape", "local_folder", "compute_hessian", "compute_lower_right_hessian", "m"))


parSapply(cluster, 1:number_of_parameters, function(ipred)
{H_and_J <- compute_H_and_J_per_parameter(ipred, spatial_images_file, c(parameter_matrix[ipred,1], parameter_matrix[ipred,2]), h, dist_constraint, m)})
stopCluster(cluster)