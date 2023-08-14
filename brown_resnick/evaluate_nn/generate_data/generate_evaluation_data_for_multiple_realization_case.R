#This R file is a script to generate evaluation data in the multiple realization case for evaluating the neural likelihood surfaces, parameter estimates,
#and confidence regions. The evaluation data consists of spatial fields and the corresponding parameters which generated the spatial fields. 
#The evaluation data is n realizations of GP per m parameters where the m parameters come from a grid over the parameter space.
library(SpatialExtremes)
library(reticulate)

#Create parameter matrix over the parameter space. 10 by 10 grid over the parameter space starting at (.2,.2)
#and increasing in both dimensions by increments of .2.
range_test <- seq(.2, 2, .2)
smooth_test <- seq(.2, 2, .2)
range_test[10] <- 1.99
smooth_test[10] <- 1.99
data_y = cbind(expand.grid(range_test, smooth_test)$Var1,
               expand.grid(range_test, smooth_test)$Var2)
parameter_matrix <- matrix(NA, nrow = 100, ncol = 2)
#first column will be range values and second column will be smooth values
parameter_matrix[,1] <- data_y[,1]
parameter_matrix[,2] <- data_y[,2]

n <- 25
n.size <- 20
x <- y <- seq(0, n.size, length = n)
coord <- expand.grid(x, y)
number_of_parameters <- 100
number_of_reps <- 200
multiple_realizations_number <- 5

local_folder <- "/home/juliatest/Desktop/likelihood_free_inference/neural_likelihood/brown_resnick"

data_x_test <- array(NA, dim = c(number_of_parameters, number_of_reps, multiple_realizations_number, n, n))
image_name <- paste(paste(as.character(n), "by", sep = "_"), as.character(n), sep = "_")
data_file_name <- paste(paste(paste(paste(paste(paste(local_folder, "evaluate_nn/generate_data/data", image_name, sep = "/"), "multi/reps/200", sep = "/"), 
                                          "evaluation_images_10_by_10_density1", sep = "/"), image_name, sep = "_"), as.character(number_of_reps), 
                              sep = "_"), "npy", sep = ".")
parameters_file_name <- paste(paste(paste(paste(paste(paste(local_folder, "evaluate_nn/generate_data/data", image_name, sep = "/"), "multi/reps/200", sep = "/"), 
                                                "evaluation_parameters_10_by_10_density1", sep = "/"), image_name, sep = "_"), as.character(number_of_reps), 
                                    sep = "_"), "npy", sep = ".")

for(ipred in 1:nrow(parameter_matrix))
{
  for(irep in 1:multiple_realizations_number)
  {
    data_x_test[ipred,irep,,,] <- rmaxstab(multiple_realizations_number, coord, 
                                    cov.mod = "brown", 
                                    range = parameter_matrix[ipred,1], 
                                    smooth = parameter_matrix[ipred,2])
  }
}

np <- import("numpy")
np$save(data_file_name, data_x_test)
np$save(parameters_file_name, parameter_matrix)