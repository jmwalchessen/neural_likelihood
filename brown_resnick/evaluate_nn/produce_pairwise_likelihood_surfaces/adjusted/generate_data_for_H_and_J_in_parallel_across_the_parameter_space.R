#This script is for generating spatial field realizations for estimating H and J (described in Chandler and Bate and our paper). For each parameter on
# a 10 by 10 grid over the parameter space, we generate number_of_reps = 5000 spatial field realizations which we will then use to compute H and J for
# each of the 10 by 10 grid parameters.
library(SpatialExtremes)
library(reticulate)
library(parallel)

#Create the 10 by 10 grid over the parameter space
range_test <- seq(.2, 2, .2)
smooth_test <- seq(.2, 2, .2)
range_test[10] <- 1.99
smooth_test[10] <- 1.99
data_y = cbind(expand.grid(range_test, smooth_test)$Var1,
               expand.grid(range_test, smooth_test)$Var2)
parameter_matrix <- matrix(NA, nrow = 100, ncol = 2)
parameter_matrix[,1] <- data_y[,1]
parameter_matrix[,2] <- data_y[,2]

#Create the spatial grid over the spatial domain (25 by 25)
n <- 25
n.size <- 20
x <- y <- seq(0, n.size, length = n)
coord <- expand.grid(x, y)
#Determine the total number of parameters and the number of realizations generated per parameter
number_of_parameters <- 100
number_of_reps <- 5000

local_folder <- "/home/juliatest/Dropbox/likelihood_free_inference/neural_likelihood/brown_resnick"


#This function generates spatial field realizations for a given parameter in the grid over the parameter space
  #ipred: the index of the parameter of interest in the parameter_matrix
  #parameter_matrix: matrix of parameters on the grid over the parameter space
  #number_of_reps: the number of spatial field realizations to generate per parameter
  #coord: the coordinates for the spatial grid over the spatial domain
  #image_size: the square root of the number of spatial observations (25)
generate_data_per_parameter <- function(ipred, parameter_matrix, number_of_reps, coord, image_size)
{
  m <- number_of_reps/100
  simulated_data <- array(0, dim = c(number_of_reps, image_size*image_size))
  for(i in 1:m)
  {
    beginning <- (i-1)*100+1
    end <- i*100
    simulated_data[beginning:end,] <- rmaxstab(100, coord, cov.mod = "brown", range = parameter_matrix[ipred,1], smooth = parameter_matrix[ipred,2])
  

  }
  np_file <- paste(local_folder, paste(paste("evaluate_nn/produce_pairwise_likelihood_surfaces/adjusted/data/simulated_data/25_by_25/spatial_images_10_by_10_density",
                                             as.character(ipred), sep = "_"), "5000.npy", sep = "_"), sep = "/")
  np <- import("numpy")
  np$save(np_file, simulated_data)
}

#Generate spatial fields per parameter using parallel computing
cores <- ((detectCores(logical = TRUE)))
cluster <- makeCluster(cores)
clusterCall(cluster, function() library(SpatialExtremes))
clusterExport(cluster, c("parameter_matrix", "number_of_parameters", "number_of_reps", "number_of_parameters", "coord", "generate_data_per_parameter",
                         "import", "local_folder", "n"))


parSapply(cluster, 1:number_of_parameters, function(ipred)
{data_list <- generate_data_per_parameter(ipred, parameter_matrix, number_of_reps, coord, n)})
stopCluster(cluster)