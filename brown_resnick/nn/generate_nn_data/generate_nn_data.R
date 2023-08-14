library(SpatialExtremes)
library(jsonlite)
library(lhs)
library(parallel)
library(reticulate)

np <- import("numpy")
local_folder <- "/home/juliatest/Dropbox/likelihood_free_inference/neural_likelihood/brown_resnick"
nn <- 25
n.beginning <- 0
n.end <- 20
x <- y <- seq(n.beginning, n.end, length = nn)
coord <- expand.grid(x,y)
cov.type <- "brown"
number_of_reps <- 500
version <- "final_version"

number_of_parameters <- 3000

ranges <- 2*parameters[,1]
smooths <- 2*parameters[,2]


data_y <- matrix(0, nrow = number_of_parameters, ncol = 2)
data_y[,1] <- ranges
data_y[,2] <- smooths

image_name <- paste(paste(as.character(nn), "by", sep = "_"), as.character(nn), sep = "_")

false_indices_matrix <- matrix(NA, nrow = number_of_reps, ncol = number_of_parameters)
for (i in 1:number_of_reps)
{
  false_indices_matrix[i,] <- sample(number_of_parameters, number_of_parameters)
}

simu_samp_per_parameter_pair <- function(data_y, false_indices_matrix, observation_number, number_of_reps, nn, coord, 
                                         cov.type)
{
  simu.data <- list()
  y_matrix <- array(rmaxstab(number_of_reps, coord, cov.mod = cov.type, range = data_y[observation_number,1], 
                             smooth = data_y[observation_number, 2]), dim = c(number_of_reps, nn, nn))
  for(i in 1:number_of_reps)
  {
    simu.data[[i]] <- list("y" = y_matrix[i,,], "parameters" = c(data_y[observation_number,1], 
                                                                 data_y[observation_number,2]), "class" = c(1))
    false_ii <- false_indices_matrix[i, observation_number]
    simu.data[[number_of_reps + i]] <- list("y" = y_matrix[i,,], "parameters" = c(data_y[false_ii,1], data_y[false_ii, 2]), 
                                            "class" = c(0))
  }
  
  data_list_name <- paste(paste(paste(paste(paste(paste(local_folder, "nn/models", sep = "/"), 
                                            image_name, sep = "/"),
                          "final_version/data/train/train_samples_25_by_25_3000_reps_500", 
                          sep = "/"), as.character(observation_number), sep = "_"), sep = "_"),
                          "json", sep = ".")
  write_json(simu.data, data_list_name)
  
}

cores <- (detectCores(logical = TRUE))
cluster <- makeCluster(cores)
clusterCall(cluster, function() library(SpatialExtremes))
clusterExport(cluster, c("nn", "coord", "ranges", "smooths", "data_y", "simu_samp_per_parameter_pair", 
                         "number_of_reps", "cov.type", "rmaxstab", "write_json", "false_indices_matrix", "local_folder", "image_name"))

start_time <- Sys.time()
parSapply(cluster, 1:nrow(data_y), function(m)
{simu_samp_per_parameter_pair(data_y, false_indices_matrix, m, number_of_reps, nn, coord, cov.type)})
stopCluster(cluster)
end_time <- Sys.time()
print(end_time - start_time)