library("reticulate")

np <- import("numpy")
local_folder <- "/home/juliatest/Dropbox/likelihood_free_inference/neural_likelihood/brown_resnick/evaluate_nn/produce_pairwise_likelihood_surfaces/adjusted"
dist_constraint <- 2
matrix_type <- "spectral"
adjusted_pairwise_likelihood_surfaces_file_name <- paste(paste(paste(paste(local_folder, "data/surfaces/dist", sep = "/"),
                                                                           as.character(dist_constraint), sep = "_"), matrix_type, sep = "/"),
                                                               "adjusted_pairwise_likelihood_surfaces_200", sep = "/")

ipreds <- c(1:60,62:70,72:80,82:100)
adjusted_pairwise_likelihood_surfaces <- array(0, dim = c(100,200,40,40))
for(ipred in ipreds)
  {
    current_adjusted_pairwise_likelihood_surfaces_file_name <- paste(paste(adjusted_pairwise_likelihood_surfaces_file_name, as.character(ipred), sep = "_"),
    "npy", sep = ".")
    adjusted_pairwise_likelihood_surfaces[ipred,,,] <- np$load(current_adjusted_pairwise_likelihood_surfaces_file_name)
}

adjusted_pairwise_likelihood_surfaces_file_name <- paste(adjusted_pairwise_likelihood_surfaces_file_name, "npy", sep = ".")
np$save(adjusted_pairwise_likelihood_surfaces_file_name, adjusted_pairwise_likelihood_surfaces)
