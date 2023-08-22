#This R file is a script to generate evaluation data for evaluating the neural likelihood surfaces, parameter estimates, and confidence regions. The evaluation
#data consists of spatial fields and the corresponding parameters which generated the spatial fields and the log likelihood field (over the parameter space) for
#the spatial field. The evaluation data is n realizations of GP per m parameters where the m parameters come from a grid over the parameter space.
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

#create a covariance matrix using the exponential kernel with
#variance and length scale as the parameters but without the variance
#as a multiplicative factor to speed up computation of the log likelihood field.
#function parameters: grid_dataframe is a dataframe of the spatial
#locations with two named columns (latitude and longitude), length_scale is a parameter
#for the exponential kernel
compute_exponential_kernel_fast_without_variance <- function(grid_dataframe, length_scale)
{
  norm_value <- norm_matrix(grid_dataframe)
  exp_matrix <- exp(-norm_value/length_scale)
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

#compute the log likelihood for a certain parameter combination (reflected in Cholesky matrix and variance).
#This function will be used to compute the log likelihood field
#function parameters:
#y: realization of the GP
#variance: parameter value for the location on  the parameter space for which we are interested in evaluating the log likelihood
#n: number of observed spatial locations for the realization of the GP
#Cholesky_without_variance: the Cholesky matrix for the covariance matrix without the variance parameter (speeds up computation of log likelihood)
#diag_cholesky_without_variance: the diagonal entries of the Cholesky matrix without the variance parameter
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

#Compute the entire log likelihood field over the parameter space using Cholesky factorization to speed up the computation.
#function parameters:
#y: realization of the GP
#possible_variances: possible variances over the parameter space (denotes the grid density/coarseness over the parameter space)
#possible_length_scales: possible length scales over the parameter space (denotes the grid density/coarseness over the parameter space)
#n: the number of observed spatial locations for the realization of the GP
#grid_dataframe: a dataframe of the spatial locations with two labeled columns (latitude and longitude)
compute_log_likelihood_field_with_cholesky <- function(y, possible_length_scales, possible_variances, n, grid_dataframe)
{
  likelihoods <- matrix(0, nrow = length(possible_length_scales), ncol = length(possible_variances))
  
  for(j in 1:length(possible_length_scales))
  {
    current_length_scale <- possible_length_scales[j]
    current_kernel_without_variance <- compute_exponential_kernel_fast_without_variance(grid_dataframe, current_length_scale)
    cholesky_without_variance <- chol(current_kernel_without_variance)
    diag_c_wihout_variance <- diag(cholesky_without_variance)
    
    for(i in 1:length(possible_variances))
    {
      current_variance <- possible_variances[i]
      likelihood <- compute_log_likelihood_with_cholesky(y, current_variance, n, cholesky_without_variance,
                                                                      diag_c_wihout_variance)
      likelihoods[i,j] <- likelihood
    }
  }
  
  return(likelihoods)
}

#main function: For each parameter grid point over the parameter space, generate n replicates of a GP and the corresponding log likelihood fields.
#function parameters:
#index: the index of the parameter in the parameter matrix to simulate realizations of the GP for
#grid_dataframe: a dataframe of the spatial locations with two labeled columns (latitude and longitude)
#number_of_replicates: number of realizations for GP per parameter
#parameter_matrix: matrix of parameter values (the grid over the parameter space)
#n: the number of observed spatial locations for the realization of the GP
#possible_variances: possible variances over the parameter space (denotes the grid density/coarseness over the parameter space)
#possible_length_scales: possible length scales over the parameter space (denotes the grid density/coarseness over the parameter space)
generate_data_per_parameter_pair <- function(index, grid_dataframe, number_of_replicates, parameter_matrix, n,
                                             possible_variances, possible_length_scales)
{
  
  data_list <- list()
  current_variance <- parameter_matrix[index,1]
  current_length_scale <- parameter_matrix[index,2]
  current_kernel <- compute_exponential_kernel_fast(grid_dataframe, current_variance, current_length_scale)
  y_matrix <- generate_gaussian_process(number_of_replicates, n, current_kernel)
  
  for(j in 1:number_of_replicates)
  {
    y <- matrix(y_matrix[,j], nrow = sqrt(n), byrow = T)
    ll <- compute_log_likelihood_field_with_cholesky(y_matrix[,j], possible_variances, possible_length_scales, n, grid_dataframe)
    data_list[[j]] <- list("y" = y, "log_likelihood_field" = ll,
                           "parameters" = c(current_variance, current_length_scale))
  }
  
  file_name <- paste(paste("data/25_by_25/ll/single/reps/200/data_9_by_9_density_25_by_25_image_200", 
                           as.character(index), sep = "_"), "json", sep = ".")
  write_json(data_list, file_name)
}

n <- 625
x <- y <- seq(-10, 10, length = sqrt(n))
coord <- expand.grid(x, y)
grid_dataframe <- as.data.frame(coord)
names(grid_dataframe) <- c("longitude", "latitude")

possible_variances <- seq(.05, 2, .05)
possible_length_scales <- seq(.05, 2, .05)

number_of_replicates <- 200
number_of_parameters <- 81
length_scale_test <- seq(.2, 1.8, .2)
variance_test <- seq(.2, 1.8, .2)
data_y = cbind(expand.grid(length_scale_test, variance_test)$Var1,
               expand.grid(length_scale_test, variance_test)$Var2)
parameter_matrix <- matrix(NA, nrow = 81, ncol = 2)
parameter_matrix[,1] <- data_y[,2]
parameter_matrix[,2] <- data_y[,1]
image_size <- 25
n.size <- 25


cores <- detectCores(logical = TRUE)
cluster <- makeCluster(cores)
clusterCall(cluster, function() library(mvtnorm))
clusterExport(cluster, c("number_of_parameters", "n", "parameter_matrix", "image_size",
                         "number_of_replicates", "generate_data_per_parameter_pair",
                         "compute_exponential_kernel_fast_without_variance", "compute_log_likelihood_with_cholesky", 
                         "compute_log_likelihood_field_with_cholesky",
                         "generate_gaussian_process", "norm_matrix", "write_json", "possible_variances", 
                         "possible_length_scales", "grid_dataframe", "compute_exponential_kernel_fast"))

parSapply(cluster, 1:nrow(parameter_matrix), function(index)
{generate_data_per_parameter_pair(index, grid_dataframe, number_of_replicates, parameter_matrix, n,
                                  possible_variances, possible_length_scales)})

stopCluster(cluster)

