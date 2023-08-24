#This script is for timing how long it takes to produce pairwise likelihood surfaces for different deltas.
library(SpatialExtremes)
library(parallel)
library(reticulate)
library(ramify)


#This function is for extracting the pairwise likelihood function from the fitmaxstab function (algorithm to find the best parameter
# estimator for the best parameter estimator for the given spatial field using pairwise likelihood).
#parameters:
#coordinates_matrix: the coordinates for the observations on the grid over the spatial domain
#flattened_evaluation_image: the flattened spatial field 
#on the 9 by 9 grid over the parameter space
#weights: the weights (1 or 0) that determine whether to include the pair of observations in the pairwise likelihood
#irep: the replication number for the particular spatial field for which we want to produce a pairwise likelihood surface 
extract_pairwise_likelihood_function_via_fitmaxstab <- function(coordinates_matrix, flattened_evaluation_image, weights)
{
  fit <- fitmaxstab(t(flattened_evaluation_image), coordinates_matrix, "brown",
                    weights = weights, 
                    method = "L-BFGS-B", 
                    #to not run the optimization algorithm (unnecessary to extract pairwise likelihood), se the
                    #number of iterations to 0
                    control = list(pgtol = 100, maxit = 0),
                    start = list(range = 1, smooth = 1))
  
  pwl_function <- function(theta)
  {
    return(-1*fit$nllh(theta))
  }
  return(pwl_function)
}

#This function produces the pairwise likelihood surface for a specific spatial field realization
#generated using a specified parameter on the grid over the parameter space (taken from the evaluation data)
#parameters:
#ipred: the row in the parameter matrix (data_y) which references the parameter of interest
#irep: the number which specifies which spatial field to use
#grid: the grid over the parameter space (density 9 by 9)
#range_grid: the range values on the grid over the parameter space
#smooth_grid: the smooth values on the grid over the parameter space
#flattened_evaluation_images: the evaluation images from the evaluation data that are already flattened from 2 dimensions to one
#coordinates_matrix: the coordinates for the observations on the grid over the spatial domain
#weights: the weights (1 or 0) that determine whether to include the pair of observations in the pairwise likelihood
#distance_constraint: the maximum distance between pairs of observations which have weight 1 and are included in the pairwise likelihood
produce_pairwise_likelihood_surface <- function(ipred, irep, grid, range_grid, smooth_grid,
                                                                flattened_evaluation_images, coordinates_matrix,
                                                                weights, distance_constraint)
{
  
    pwl_function <- extract_pairwise_likelihood_function_via_fitmaxstab(coordinates_matrix = coordinates_matrix,
                                                               flattened_evaluation_image = t(flattened_evaluation_images[ipred,irep,,]),
                                                               weights = weights, 
                                                               irep = irep)
    pairwise_likelihood_surface <- array(0, dim = c(length(range_grid), length(smooth_grid)))
    
    for(i in 1:length(range_grid))
    {
      for(j in 1:length(smooth_grid))
      {
        #range on x axis
        pairwise_likelihood_surface[i,j] <- pwl_function(c(range_grid[i], smooth_grid[j]))
      }
    }
}


#To speed up computation of pairwise likelihood surface, we parallelize the inner for loop in produce_pairwise_likelihood_surface
  #parameters:
    #pwl_function: pairwise likelihood function extracted from fitmaxstab using the spatial field of interest
    #range_value: fixed range value for this row of varying smooth values in the pairwise likelihood surface
    #smooth_grid: smooth values with vary across this row of the pairwise likelihood surface
inner_for_loop_in_produce_pairwise_likelihood_surface <- function(pwl_function, range_value, smooth_grid)
{
  #row for the pairwise likelihood surface
  pairwise_likelihood_surface_vector <- matrix(0, nrow = length(smooth_grid))
  
  for(j in 1:length(smooth_grid))
  {
    pairwise_likelihood_surface_vector[j] <- pwl_function(c(range_value, smooth_grid[j]))
  }
  return(pairwise_likelihood_surface_vector)
}

#This function produces a pairwise likelihood surface using parallelization.
  #parameters:
    #ipred: the row in the parameter matrix (data_y) which references the parameter of interest
    #irep: the number which specifies which spatial field to use
    #range_grid: the range values on the grid over the parameter space
    #smooth_grid: the smooth values on the grid over the parameter space
    #evaluation_images: the evaluation images from the evaluation data 
    #coordinates_matrix: the coordinates for the observations on the grid over the spatial domain
    #weights: the weights (1 or 0) that determine whether to include the pair of observations in the pairwise likelihood
produce_pairwise_likelihood_surface_with_parallelization <- function(flattened_evaluation_image, range_grid, smooth_grid,
                                                                     coordinates_matrix, weights)
{
  pwl_function <- extract_pairwise_likelihood_function_via_fitmaxstab(coordinates_matrix, flattened_evaluation_image, weights)
  
  outputs <- parSapply(cluster, 1:length(range_grid), function(i)
  {inner_for_loop_in_produce_pairwise_likelihood_surface(pwl_function, range_grid[i], smooth_grid)})
  return(outputs)
}

np <- import("numpy")
number_of_replications <- 50
image_size <- 25
image_name <- paste(paste(as.character(image_size), "by", sep = "_"), as.character(image_size), sep = "_")
spatial_domain_size <- 20
distance_constraint <- 1
evaluation_ranges <- seq(.2, 1.8, .2)
evaluation_smooths <- seq(.2, 1.8, .2)

#Latitudes and longitudes for observations on the spatial domain
x <- y <- seq(0, spatial_domain_size, length = image_size)
coordinates <- expand.grid(x, y)
coordinates_matrix = cbind(coordinates$Var1, coordinates$Var2)

parameter_matrix = cbind(expand.grid(evaluation_ranges, evaluation_smooths)$Var1,
               expand.grid(evaluation_ranges, evaluation_smooths)$Var2)

#Load evaluation images
local_folder <- "/home/juliatest/Dropbox/likelihood_free_inference/neural_likelihood/brown_resnick"
evaluation_images_file <- paste(paste(paste(paste(paste(paste(local_folder, "evaluate_nn/generate_data/data", sep = "/"), 
                                      image_name, sep = "/"), "single/reps/200", sep = "/"), "evaluation_images_9_by_9_density",
                                      sep = "/"), image_name, sep = "_"), "200.npy", sep = "_")
evaluation_images <- np$load(evaluation_images_file)
range_grid <- seq(.05, 2, .05)
smooth_grid <- seq(.05, 2, .05)

grid <- cbind(expand.grid(range_grid, smooth_grid)$Var1,
              expand.grid(range_grid, smooth_grid)$Var2)

ipred <- 31
time_array <- matrix(0, nrow = number_of_replications)

weights <- as.numeric(distance(coordinates_matrix) < distance_constraint) 

cores <- detectCores(logical = TRUE)
cluster <- makeCluster(cores)
clusterCall(cluster, function() library(SpatialExtremes))

for(irep in 1:number_of_replications)
{
  flattened_evaluation_image <- flatten(evaluation_images[ipred,irep,,])
  clusterExport(cluster, c("range_grid", "smooth_grid", "weights", "coordinates_matrix", "flattened_evaluation_image",
                           "inner_for_loop_in_produce_pairwise_likelihood_surface", "extract_pairwise_likelihood_function_via_fitmaxstab"))
  
  time_array[irep] <- as.numeric(system.time(produce_pairwise_likelihood_surface_with_parallelization(flattened_evaluation_image,
                                                                                                      range_grid, smooth_grid,
                                                                                                      coordinates_matrix,
                                                                                                      weights))["elapsed"])
    
}

time_file <- paste(paste(paste(paste(paste(paste(paste(local_folder, "evaluate_nn/timing_studies/data", sep = "/"), image_name, sep = "/"), 
                    "pairwise_likelihood/dist", sep = "/"), as.character(distance_constraint), sep = "_"), 
                   "pairwise_likelihood_surface_time_with_parallelization_on_laptop", sep = "/"), as.character((ipred-1)), sep = "_"), 
                   "npy", sep = ".")
np$save(time_file, time_array)

