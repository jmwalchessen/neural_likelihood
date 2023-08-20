#This script generates the adjusted pairwise likelihood surfaces for the evaluation data for the single realization case using both cholesky and spectral methods.
source("likelihood_adjustment_functions.R")
library(parallel)
library(reticulate)


h <- .05
h_in_name <- "5e02"
dist_constraint <- 2
range_test = seq(.2, 2, .2)
smooth_test = seq(.2, 2, .2)
smooth_test[10] <- 1.99

possible_ranges <- seq(.05, 2, .05)
possible_smooths <- seq(.05, 2, .05)

data_y = cbind(expand.grid(range_test, smooth_test)$Var1,
               expand.grid(range_test, smooth_test)$Var2)
nrep <- 2

local_folder <- "/home/juliatest/Dropbox/likelihood_free_inference/neural_likelihood/brown_resnick"

np <- import("numpy")
#Load the maximum pairwise likelihood estimates for the surfaces to speed up computation
pwl_mle_file <- paste(paste(paste(paste(local_folder, "evaluate_nn/produce_pairwise_likelihood_estimates/data/25_by_25/dist", sep = "/"),
                                                 as.character(dist_constraint), sep = "_"), "single/reps/200", sep = "/"),
                         "evaluation_pairwise_likelihood_estimators_10_by_10_image_25_by_25_200.npy", sep = "/")

pwl_mles <- np$load(pwl_mle_file)

#Load the spatial field realizations for which we will be generating adjusted pairwise surfaces for
evaluation_images_file_name <- paste(paste(local_folder, "evaluate_nn/generate_data/data/25_by_25/single/reps/200", sep = "/"), 
                          "evaluation_images_10_by_10_density_25_by_25_200.npy", sep = "/")
evaluation_images <- np$load(evaluation_images_file_name)


#This function produces the adjusted pairwise likelihood surfaces (both cholesky and spectral) for the given spatial fields specified by the parameter (ipred)
  #function parameters:
    #ipred: the index of the parameter to produce the adjusted pairwise likelihood surfaces (this allows us to parallelize the computation)
    #nrep: the total number of spatial field realizations per parameter
    #spatial_fields: the 
produce_adjusted_pwl_surfaces_per_parameter <- function(ipred, nrep, spatial_fields, pwl_mles, possible_ranges, possible_smooths,
                                                        dist_constraint, h_in_name)
{
  
  HI_file_name <- paste(paste(paste(paste(paste(paste(paste(paste(paste(paste(paste(local_folder,
                                                                                    "evaluate_nn/produce_pairwise_likelihood_surfaces/adjusted/data/matrices/dist", 
                                                                                    sep = "/"), as.character(dist_constraint), sep = "_"), "h", sep = "/"),
                                                                                    h_in_name, sep = "/"), "H", sep = "/"), "HI", sep = "/"), "HI_matrix_dist_2",
                                                                                    sep = "/"), "5000", sep = "_"), as.character(ipred), sep = "_"), "npy",
                                                                                    sep = "."))
  
  HA_file_name <- paste(paste(paste(paste(paste(paste(paste(paste(paste(paste(local_folder,
                                                                              "evaluate_nn/produce_pairwise_likelihood_surfaces/adjusted/data/matrices/dist", 
                                                                        sep = "/"), as.character(dist_constraint), sep = "_"), "h", sep = "/"), h_in_name,
                                                                        sep = "/"), "H", sep = "/"), "HA", sep = "/"), "HA_matrix_dist_2", sep = "/"), "5000",
                                                                        sep = "_"), as.character(ipred), sep = "_"), "npy", sep = ".")
  np <- import("numpy")
  HI <- np$load(HI_file_name)
  HI_eigenvalues <- eigen(HI)$values
  HA <- np$load(HA_file_name)
  
  spectral_adj_pwl_surfaces <- array(0, dim = c(nrep, length(possible_smooths), length(possible_ranges)))
  cholesky_adj_pwl_surfaces <- array(0, dim = c(nrep, length(possible_smooths), length(possible_ranges)))
  
  for(irep in 1:nrep)
  {
    mle <- pwl_mles[ipred,irep,]
    flattened_spatial_field <- t(flatten(spatial_fields[ipred,irep,,]))
    pwl_function <- generate_pairwise_likelihood(flattened_spatial_field, dist_constraint)
    spectralC <- produce_spectral_C(HI, HA)
    spectral_adj_pwl_surfaces[irep,,] <- produce_horizontally_adjusted_pairwise_likelihood_surface(pwl_function, spectralC, mle, possible_smooths,
                                                                                                   possible_ranges)
    choleskyC <- produce_cholesky_C(HI, HA)
    cholesky_adj_pwl_surfaces[irep,,] <- produce_horizontally_adjusted_pairwise_likelihood_surface(pwl_function, choleskyC, mle, possible_smooths,
                                                                                                   possible_ranges)
  }
  
  cholesky_adj_pwl_name <- paste(paste(paste(paste(paste(paste(local_folder, "evaluate_nn/produce_pairwise_likelihood_surfaces/adjusted/data/surfaces/dist",
                                                               sep = "/"), as.character(dist_constraint), sep = "_"),  "cholesky", sep = "/"), 
                                                               "adjusted_pairwise_likelihood_surfaces_200", sep = "/"), as.character(ipred), sep = "_"), "npy",
                                                               sep = ".")
  spectral_adj_pwl_name <- paste(paste(paste(paste(paste(paste(local_folder, "evaluate_nn/produce_pairwise_likelihood_surfaces/adjusted/data/surfaces/dist",
                                                               sep = "/"), as.character(dist_constraint), sep = "_"),  "spectral", sep = "/"), 
                                                               "adjusted_pairwise_likelihood_surfaces_200", sep = "/"), as.character(ipred), sep = "_"), "npy",
                                                               sep = ".")
  np$save(spectral_adj_pwl_name, spectral_adj_pwl_surfaces)
  np$save(cholesky_adj_pwl_name, cholesky_adj_pwl_surfaces)
  
}

ipreds <- c(1:60, 62:70, 72:80, 82:100)
cores <- (((detectCores(logical = TRUE))))
cluster <- makeCluster(cores)
clusterCall(cluster, function() library(SpatialExtremes))
clusterExport(cluster, c("produce_adjusted_pwl_surfaces_per_parameter", "import", "nrep", "pwl_mles", "possible_ranges", "possible_smooths",
                         "dist_constraint", "h_in_name", "local_folder", "h", "produce_MLE", "evaluation_images", "flatten", "generate_pairwise_likelihood",
                         "produce_pairwise_likelihood_surface", "cholesky_adjusted_pairwise_likelihood_function",
                         "spectral_adjusted_pairwise_likelihood_function", "produce_spectral_C", "produce_cholesky_C", 
                         "produce_horizontally_adjusted_pairwise_likelihood_surface", "adjust_theta"))


parSapply(cluster, ipreds, function(ipred)
{adj_pwl <- produce_adjusted_pwl_surfaces_per_parameter(ipred, nrep, evaluation_images, pwl_mles, possible_ranges, possible_smooths, dist_constraint,
                                                        h_in_name)})
stopCluster(cluster)