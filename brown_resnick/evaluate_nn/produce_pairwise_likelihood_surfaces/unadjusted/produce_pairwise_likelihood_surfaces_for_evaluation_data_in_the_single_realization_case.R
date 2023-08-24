#This script is for producing the pairwise likelihood surfaces for the evaluation data in the single realization case. 
#Changing the distance constraint/delta given (see paper for explanation of the distance constraint/delta), will change
#the resulting pairwise likelihood surfaces.
library(SpatialExtremes)
library(parallel)
library(ramify)
library(reticulate)

local_folder <- "/home/juliatest/Dropbox/likelihood_free_inference/neural_likelihood/brown_resnick/evaluate_nn/produce_pairwise_likelihood_surfaces/unadjusted"
number_of_replications <- 200
image_size <- 25
image_name <- paste(paste(as.character(image_size), "by", sep = "_"), as.character(image_size), sep = "_")
spatial_domain_size <- 20
distance_constraint <-2
ranges = seq(.2, 1.8, .2)
smooths = seq(.2, 1.8, .2)
number_of_parameters <- 81

#The longitude and latitudes for the observations on the lattice over the spatial domain
x <- y <- seq(0, spatial_domain_size, length = image_size)
coordinates <- expand.grid(x, y)
coordinates_matrix = cbind(coordinates$Var1, coordinates$Var2)

image_name <- paste(paste(as.character(image_size), "by", sep = "_"), as.character(image_size), sep = "_")
home_folder <- "/home/juliatest/Dropbox/likelihood_free_inference/neural_likelihood/brown_resnick"
np <- import("numpy")
evaluation_images_file_name <- paste(paste(paste(paste(paste(paste(paste(home_folder, "evaluate_nn/generate_data/data", sep = "/"), 
                                     image_name, sep = "/"), "single/reps/200", sep = "/"), 
                         "evaluation_images_9_by_9_density", sep = "/"), image_name, sep = "_"), 
                         as.character(number_of_replications), sep = "_"), "npy", sep = ".")

evaluation_images <- np$load(evaluation_images_file_name)
flattened_evaluation_images <- array(NA, dim = c(number_of_parameters, number_of_replications, (image_size**2)))

for( ipred in 1:number_of_parameters)
{
  for(irep in 1:number_of_replications)
  {
    flattened_evaluation_images[ipred,irep,] <- flatten(evaluation_images[ipred,irep,,])
  }
}

#Weights for which of the pairs of observations are used to compute the pairwise likelihood
weights <- as.numeric(distance(coordinates_matrix) < distance_constraint) 

#Values of the range and smooth parameters 
range_grid <- seq(.05, 2, .05)
smooth_grid <- seq(.05, 2, .05)


#This function is for extracting the pairwise likelihood function from the fitmaxstab function (algorithm to find the best parameter
# estimator for the best parameter estimator for the given spatial field using pairwise likelihood).
  #parameters:
    #coordinates_matrix: the coordinates for the observations on the grid over the spatial domain
    #flattened_evaluation_images_per_parameter: the flattened spatial fields generated using a particular parameter 
      #on the 9 by 9 grid over the parameter space
    #weights: the weights (1 or 0) that determine whether to include the pair of observations in the pairwise likelihood
    #irep: the replication number for the particular spatial field for which we want to produce a pairwise likelihood surface 
extract_pairwise_likelihood_function_via_fitmaxstab <- function(coordinates_matrix, flattened_evaluation_images_per_parameter, weights, irep)
{
  fit <- fitmaxstab(t(flattened_evaluation_images_per_parameter[,irep]), coordinates_matrix, "brown",
                   weights = weights, 
                   method = "L-BFGS-B", 
                   #to not run the optimization algorithm (unnecessary to extract pairwise likelihood), se the
                   #number of iterations to 0
                   control = list(pgtol = 100, maxit = 0),
                   start = list(range = 1, smooth = 1))
  
  return(fit)
}

#This function produces the pairwise likelihood surfaces for all spatial images generated using
#a specified parameter on the grid over the parameter space (taken from the evaluation data)
  #parameters:
    #ipred: the row in the parameter matrix (data_y) which references the parameter of interest
    #grid: the grid over the parameter space (density 9 by 9)
    #range_grid: the range values on the grid over the parameter space
    #smooth_grid: the smooth values on the grid over the parameter space
    #flattened_evaluation_images: the evaluation images from the evaluation data that are already flattened from 2 dimensions to one
    #coordinates_matrix: the coordinates for the observations on the grid over the spatial domain
    #weights: the weights (1 or 0) that determine whether to include the pair of observations in the pairwise likelihood
    #number_of_replications: the total number of realizations per parameter on the lattice over the parameter space
    #distance_constraint: the maximum distance between pairs of observations which have weight 1 and are included in the pairwise likelihood
generate_pairwise_likelihood_surfaces_per_parameter <- function(ipred, grid, range_grid, smooth_grid,
                                                       flattened_evaluation_images, coordinates_matrix,
                                                       weights, number_of_replications, distance_constraint)
{
  np <- import("numpy")
  pairwise_likelihood_surfaces_per_parameter <- array(NA, c(number_of_replications, length(range_grid), length(smooth_grid)))
  
  for(irep in 1:200)
  {
    fit <- extract_pairwise_likelihood_function_via_fitmaxstab(coordinates_matrix = coordinates_matrix,
                                                flattened_evaluation_images_per_parameter = t(flattened_evaluation_images[ipred,,]),
                                                weights = weights, 
                                                irep = irep)
    
    for(i in 1:length(range_grid))
    {
      for(j in 1:length(smooth_grid))
      {
        #range on x axis
        pairwise_likelihood_surfaces_per_parameter[irep,i,j] <- -1*(fit$nllh(c(range_grid[i], smooth_grid[j])))
      }
    }
  }
  
  file_name <- paste(paste(paste(paste(paste(paste(paste(paste(paste(paste(paste(paste(paste(local_folder, "data", sep = "/"), 
                     image_name, sep = "/"), "dist", sep = "/"), as.character(distance_constraint), sep = "_"), sep = "/"), "single/reps", sep = "/"),
                     as.character(number_of_replications), sep = "/"), "pairwise_likelihood_surfaces_9_by_9_density", sep = "/"), image_name, sep = "_"),
                     "image", sep = "_"), as.character(number_of_replications), sep = "_"), as.character(ipred), sep = "_"), "npy", sep = ".")
          
  np$save(file_name, pairwise_likelihood_surfaces_per_parameter)
}

cores <- (detectCores(logical = TRUE))
cluster <- makeCluster(cores)
clusterCall(cluster, function() library(SpatialExtremes))
clusterCall(cluster, function() library(reticulate))
clusterExport(cluster, c("flattened_evaluation_images", "coordinates_matrix", "weights", "grid", "local_folder", "image_name",
                         "fitmaxstab", "smooth_grid", "range_grid", "generate_pairwise_likelihood_surfaces_per_parameter",
                         "number_of_replications", "extract_pairwise_likelihood_function_via_fitmaxstab", "distance_constraint",
                         "number_of_replications"))


start_time <- Sys.time()
parSapply(cluster, 1:number_of_parameters, function(ipred)
{data_list <- generate_pairwise_likelihood_surfaces_per_parameter(ipred, grid, range_grid, smooth_grid,
                                                                  flattened_evaluation_images, coordinates_matrix,
                                                                  weights, number_of_replications, distance_constraint)})
stopCluster(cluster)
end_time <- Sys.time()
print(end_time - start_time)