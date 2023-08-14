#This file is a script to generate training and test data for calibrating the classifier outputs from the neural network. 
#The data consists of two classes.The two classes consist of pairs of spatial fields and parameters. In the first class, the 
#spatial field is paired with the parameter with which the spatial field was generated and in the second class, the
#spatial field is paired with a permuted parameter so that the paired parameter and spatial field are independent.
library(lhs)
library(mvtnorm)
library(jsonlite)
library(parallel)

#create a matrix of L2 norms between spatial locations on the 
#25 by 25 grid over the spatial domain [-10,10] by [-10,10]
#function parameters: grid_dataframe is a dataframe of the
#spatial locations with two labeled columns (latitude and longitude)
norm_matrix <- function(grid_dataframe)
{
  n <- nrow(grid_dataframe)
  x <- grid_dataframe$longitude
  x <- as.matrix(x)
  x_matrix <- t(matrix(rep(t(x), n), ncol = n, byrow = TRUE))
  
  longitude_squared <- (x_matrix - t(x_matrix))**2
  
  y <- grid_dataframe$latitude
  y <- as.matrix(y)
  y_matrix <- t(matrix(rep(t(y), n), ncol = n, byrow = TRUE))
  
  latitude_squared <- (y_matrix - t(y_matrix))**2
  
  norm_matrix <- sqrt(longitude_squared + latitude_squared)
  return(norm_matrix)
}

#create a covariance matrix using the exponential kernel with
#variance and length scale as the parameters
#function parameters: grid_dataframe is a dataframe of the spatial
#locations with two named columns (latitude and longitude), variance and length_scale are the parameters
#for the exponential kernel
compute_exponential_kernel_fast <- function(grid_dataframe, variance, length_scale)
{
  norm_value <- norm_matrix(grid_dataframe)
  exp_matrix <- variance*exp(-norm_value/length_scale)
  return(exp_matrix)
}

#simulate realizations of a Gaussian Process with n observed locations and
#mean zero with the given covariance matrix
#function parameters: 
  #number_of_replicates: the number of realizations of the Gaussian Process to simulate
  #n: the number of observed locations for the Gaussian Process,
  #covariance matrix: a n by n matrix (created using exponential kernel here)
generate_gaussian_process <- function(number_of_replications, n, covariance_matrix)
{
  z_matrix <- t(rmvnorm(number_of_replicates, mean = rep(0, n), 
                        sigma = diag(n)))
  C <- chol(covariance_matrix)
  y_matrix <- t(C) %*% z_matrix
  
  return(y_matrix)
}

#main function: For each sampled parameter in the user-defined parameter space, simulate a user-specified
#number of realizations of a Gaussian Process with n observed locations. Save first pairs of the simulated
#spatial fields and the parameters and the class (class 1, the dependent class) which generated the spatial fields
#and second pairs of those same spatial fields and permuted parameters and the class (class 0, the independent class). 
#To speed up computation, we use parallel computing.
#function parameters:
  #grid_dataframe: dataframe of spatial locations with two named columns (latitude and longitude)
  #n: number of spatial locations in grid_dataframe and in general the number of spatial locations
                          #where the Gaussian Process is observed
  #parameter_matrix: a matrix of parameter values (variance is the first column and length scale is the second column)
                    #with which to simulate the realizations of the Gaussian Process
  #number_of_replicates: the number of realizations to simulate of the Gaussian process with the specified parameters
  #false_index_matrix: the matrix (dimension: number of parameters by number_of_replicates) containing the permutations
                      #(there are number_of_replicates total permutations) of the indices/rows of the parameter matrix 
                      #to create a pair of spatial field and parameter in which the two are independent for class 0 
                      #(independent class)
  #parameter_index: the index in the parameter matrix which refers to the parameter for which the function generates
                    #realizations of the Gaussian process
  #file location: the file names to save the pairs of spatial field and parameter and corresponding class labels in a
                  #list in a json file. There is one json file per parameter. See gaussian_process_data_shaping.py to
                  #convert these files to numpy matrices (for training the neural network)
generate_nn_data_per_parameter_pair <- function(grid_dataframe, n, parameter_matrix, number_of_replicates, 
                                                false_index_matrix, parameter_index, file_location)
{
  simu.data <- list()

  variance <- parameter_matrix[parameter_index,1]
  length_scale <- parameter_matrix[parameter_index,2]
  kernel <- compute_exponential_kernel_fast(grid_dataframe, variance, length_scale)
  y_matrix <- generate_gaussian_process(number_of_replicates, n, kernel)
    
  for(j in 1:number_of_replicates)
  {
    simu.data[[j]] <- list("y" = y_matrix[,j], "parameters" = c(variance, length_scale), "class" = c(1))
    false_index <- false_indices_matrix[j, parameter_index]
    simu.data[[number_of_replicates + j]] <- list("y" = y_matrix[,j], "parameters" = c(parameter_matrix[false_index,1], 
                                                                                         parameter_matrix[false_index, 2]), 
                                              "class" = c(0))
  }
  
  file_name <- paste(paste(file_location, as.character(parameter_index), sep = "_"), "json", sep = ".")
  write_json(simu.data, file_name)
  
}


#number of spatial observations
n <- 625
#number of realizations of the Gaussian process per parameter
number_of_replicates <- 50
#total number of parameters (number of rows in parameter_matrix)
number_of_parameters <- 3000
#matrix of parameters from extended parameter space (0,2.5) by (0,2.5) using Latin Hypercube sampling
parameter_matrix <- 2*randomLHS(number_of_parameters, 2)
#longitudes and latitudes of grid points in spatial domain [-10,10] by [-10,10]
x <- y <- seq(-10, 10, length = sqrt(n))
#create a dataframe of all spatial locations from the given longitudes and latitudes
coord <- expand.grid(x, y)
#create a dataframe of spatial locations with two named columns (latitude and longitude)
grid_dataframe <- as.data.frame(coord)
names(grid_dataframe) <- c("longitude", "latitude")

#create matrix of permutations (dimension: number of parameters by number of replicates) of the indices for
# the parameter matrix to break the dependency between spatial fields and parameters for class 1 (the dependent class)
#to create class 0 (independent class)
false_indices_matrix <- matrix(NA, nrow = number_of_replicates, ncol = number_of_parameters)
for (i in 1:number_of_replicates)
{
  false_indices_matrix[i,] <- sample(number_of_parameters, number_of_parameters)
}

#name for the created json files (in this case, data for train)
file_location <- "data/25_by_25/final_version/train/train_data_25_by_25_3000_rep_50"

#setting up parallel computing

#collect the cores on machine
cores <- detectCores(logical = TRUE)
cluster <- makeCluster(cores)
#send the required R packages to each of the cores
clusterCall(cluster, function() library(mvtnorm))
#send functions and variables to each of the cores
clusterExport(cluster, c("grid_dataframe", "n", "parameter_matrix", "number_of_replicates", "false_indices_matrix", 
                         "generate_nn_data_per_parameter_pair", "compute_exponential_kernel_fast", "generate_gaussian_process", 
                         "norm_matrix", "write_json", "file_location"))

#parallel computing
parSapply(cluster, 1:nrow(parameter_matrix), function(parameter_index)
{generate_nn_data_per_parameter_pair(grid_dataframe, n, parameter_matrix, number_of_replicates, 
                                     false_indices_matrix, parameter_index, file_location)})
stopCluster(cluster)
