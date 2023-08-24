#This script produces the adjusted H (expectation of the hessian) using the unadjusted H and J as described in the paper "Inference on Clustered Data Using
#Indepent Loglikelihood" by Chandler and Bates
source("likelihood_adjustment_functions.R")
library(reticulate)

h <- .05
h_in_name <- "5e02"
range_test = seq(.2, 1.8, .2)
smooth_test = seq(.2, 1.8, .2)
local_folder = "/home/juliatest/Dropbox/likelihood_free_inference/neural_likelihood/brown_resnick"
dist_constraint <- 2

produce_HA_per_parameter <- function(ipred, local_folder, dist_constraint)
{
  np <- import("numpy")
  HI_folder_name <- paste(paste(paste(paste(paste(paste(local_folder, "evaluate_nn/produce_pairwise_likelihood_surfaces/adjusted/data/matrices/dist", 
                                                                        sep = "/"), as.character(dist_constraint), sep = "_"), "h", sep = "/"), h_in_name, 
                                                      sep = "/"), "H", sep = "/"), "HI", sep = "/")
  HI_file_name <- paste(paste(paste(paste(paste(HI_folder_name, "HI_matrix_dist", sep = "/"), as.character(dist_constraint), sep = "_"), "5000", sep = "_"),
  as.character(ipred), sep = "_"), "npy", sep = ".")
  HI <- np$load(HI_file_name) 
  J_folder_name <- paste(paste(paste(paste(paste(local_folder, "evaluate_nn/produce_pairwise_likelihood_surfaces/adjusted/data/matrices/dist", 
                                                       sep = "/"), as.character(dist_constraint), sep = "_"), "h", sep = "/"), h_in_name, 
                                     sep = "/"), "J", sep = "/")
  J_file_name <- paste(paste(paste(paste(paste(J_folder_name, "J_matrix_dist", sep = "/"), as.character(dist_constraint), sep = "_"), "5000", sep = "_"), 
                       as.character(ipred), sep = "_"), "npy", sep = ".")
  J <- np$load(J_file_name)
  HI_eigenvalues <- eigen(HI)$values
  if((HI_eigenvalues[1] < 0) & (HI_eigenvalues[2] < 0))
  {
    HA <- produce_HA(J, HI)
    HA_folder_name <- paste(paste(paste(paste(paste(paste(local_folder, "evaluate_nn/produce_pairwise_likelihood_surfaces/adjusted/data/matrices/dist", 
                                                        sep = "/"), as.character(dist_constraint), sep = "_"), "h", sep = "/"), h_in_name, 
                                     sep = "/"), "H", sep = "/"), "HA", sep = "/")
    HA_file_name <- paste(paste(paste(paste(paste(HA_folder_name, "HA_matrix_dist", sep = "/"), as.character(dist_constraint), sep = "_"), 
                                      "5000", sep = "_"),
                             as.character(ipred), sep = "_"), "npy", sep = ".")
    np$save(HA_file_name, HA)
  }
}

n_params <- 81

for (ipred in 1:n_params)
  {
    produce_HA_per_parameter(ipred, local_folder, dist_constraint)
  }