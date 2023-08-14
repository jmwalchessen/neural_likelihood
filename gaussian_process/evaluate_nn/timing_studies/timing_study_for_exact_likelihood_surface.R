#This file is for timing how long it takes to produce an exact likelihood surface (using cholesky factorization and parallelization)
library(mvtnorm)
library(MASS)
library(fields)
library(reticulate)
library(parallel)

#Produce the distances between observations in matrix form (n by n matrix where n is the number of observations)
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

#Produce the covariance matrix using an exponential kernel
#parameters:
  #grid_dataframe: dataframe in which the two columns are the latitude and longitude of the observations
  #variance: parameter for exponential kernel
  #length_scale: parameter for exponential kernel
compute_exponential_kernel_fast <- function(grid_dataframe, variance, length_scale)
{
  norm_value <- norm_matrix(grid_dataframe)
  exp_matrix <- variance*exp(-norm_value/length_scale)
  return(exp_matrix)
}

#Produce covariance matrix without variance. This function speeds up the process
#of computing exact likelihood surface
#parameters:
  #grid_dataframe: dataframe in which the two columns are the latitude and longitude of the observations
  #length_scale: parameter for exponential kernel
compute_exponential_kernel_fast_without_variance <- function(grid_dataframe, length_scale)
{
  norm_value <- norm_matrix(grid_dataframe)
  exp_matrix <- exp(-norm_value/length_scale)
  return(exp_matrix)
}


#Compute the log likelihood for a realization for the Cholesky factorization of the covariance matrix without the variance
#parameters:
  #y: realization of a Gaussian process
  #variance: parameter for covariance matrix
  #n: number of observations
  #Cholesky_without_variance: Cholesky factorization of the covariance matrix without variance
  #diag_c_without_variance: diagonal of Cholesky factorization of the covariance matrix without variance
compute_log_likelihood_with_cholesky <- function(y, variance, n, Cholesky_without_variance, diag_c_without_variance)
{
  Cholesky <- sqrt(variance)*Cholesky_without_variance
  diag_c <- sqrt(variance)*diag_c_without_variance
  D <- backsolve(Cholesky, y, upper.tri = TRUE, transpose = TRUE)
  num <- (-1/2)*t(D) %*% D
  denom_after_log <- (n/2)*log(2*pi) + (1/2)*2*sum(log(diag_c))
  log_likelihood <- num - denom_after_log
  return(log_likelihood)
}

#For a fixed length scale on the parameter grid, compute the Cholesky factorization without the variance then filter through each of the variances
#in the row for the fixed length scale and compute the exact log likelihood using the Cholesky factorization
#parameters:
  #irep: number that refers to the realization
  #ipred: number that refers to the parameter on the 10 by 10 parameter grid
  #data: evaluation data (200 realizations of a GP for 10 by 10 parameter grid)
  #current_length_scale: the fixed lengthscale (points to a row on the 10 by 10 parameter grid)
  #possible_variances: the variances on the 10 by 10 parameter grid (variances of the row for which length scale is fixed)
  #n: the number of observations
  #grid_dataframe: dataframe of the latitude and longitudes for the observations
inner_for_loop_in_log_likelihood_field_with_cholesky <- function(irep, ipred, data, current_length_scale, possible_variances, n, grid_dataframe)
{
  y <- matrix(data[ipred,irep,,,], nrow = n, byrow = TRUE)
  likelihood_vector <- matrix(0, nrow = length(possible_variances))
  current_kernel_without_variance <- compute_exponential_kernel_fast_without_variance(grid_dataframe, current_length_scale)
  cholesky_without_variance <- chol(current_kernel_without_variance)
  diag_c_wihout_variance <- diag(cholesky_without_variance)
  
  for(i in 1:length(possible_variances))
  {
    current_variance <- possible_variances[i]
    likelihood_vector[i] <- compute_log_likelihood_with_cholesky(y, current_variance, n, cholesky_without_variance,
                                                                              diag_c_wihout_variance)
  }
  return(likelihood_vector)
}

#Produce the exact log likelihood surface using Cholesky factorization and parallelization
#parameters:
  #irep: number that refers to the realization
  #ipred: number that refers to the parameter on the 10 by 10 parameter grid
  #data: evaluation data (200 realizations of a GP for 10 by 10 parameter grid)
  #possible_variances: the variances on the 10 by 10 parameter grid
  #possible_length_scales: length scales on the 10 by 10 parameter grid
  #n: the number of observations
  #grid_dataframe: dataframe of the latitude and longitudes for the observations
compute_cholesky_log_likelihood_field_with_parallelization <- function(irep, ipred, data, possible_variances, possible_length_scales,
                                                                                    n, grid_dataframe)
{
  likelihoods <- matrix(0, nrow = length(possible_variances), ncol = length(possible_length_scales))
  outputs <- parSapply(cluster, 1:length(possible_length_scales), function(i)
  {inner_for_loop_in_log_likelihood_field_with_cholesky(irep, ipred, data, possible_length_scales[i], possible_variances, n, grid_dataframe)})
  return(outputs)
}


#Load evaluation data (use 50 realizations of GP with length scale = .8 and variance = .8)
np <- import("numpy")
image_size <- 25
image_name <- paste(paste(as.character(image_size), "by", sep = "_"), as.character(image_size), sep = "_")
local_folder <- "/home/juliatest/Desktop/likelihood_free_inference/neural_likelihood/gaussian_process"
numpy_file_name <- paste(paste(paste(paste(local_folder, "evaluate_nn/generate_data/data", sep = "/"), image_name, sep = "/"), 
                               "single/reps/200", sep = "/"), "evaluation_images_10_by_10_density_25_by_25_200.npy", sep = "/")

evaluation_data <- np$load(numpy_file_name)

n <- 625
x <- y <- seq(-10, 10, length = sqrt(n))
coord <- expand.grid(x, y)
grid_dataframe <- as.data.frame(coord)
names(grid_dataframe) <- c("longitude", "latitude")
possible_length_scales <- seq(.05, 2, .05)
possible_variances <- seq(.05, 2, .05)
number_of_replicates <- 50
ipred <- 34
time_array <- array(0, dim = number_of_replicates)

#compute exact log likelihood surfaces using all cores on laptop
cores <- detectCores(logical = TRUE)
cluster <- makeCluster(cores)
clusterCall(cluster, function() library(mvtnorm))

for(irep in 1:number_of_replicates)
{
  clusterExport(cluster, c("n", "evaluation_data",
                           "number_of_replicates", "norm_matrix", "np",
                           "possible_variances", "possible_length_scales", "grid_dataframe", "time_array",
                           "inner_for_loop_in_log_likelihood_field_with_cholesky", "compute_log_likelihood_with_cholesky",
                           "compute_exponential_kernel_fast_without_variance", "norm_matrix", "ipred", "irep"))
  time_array[irep] <- as.numeric(system.time(compute_cholesky_log_likelihood_field_with_parallelization(irep, ipred, evaluation_data, 
                                                                                                                     possible_variances, possible_length_scales, 
                                                                                                                     n, grid_dataframe))["elapsed"])
}

numpy_file_name <- paste(paste("data", image_name, sep = "/"),
                         "exact_likelihood_surface_time_with_parallelization_on_laptop_33.npy", sep = "/")
np$save(numpy_file_name, time_array)