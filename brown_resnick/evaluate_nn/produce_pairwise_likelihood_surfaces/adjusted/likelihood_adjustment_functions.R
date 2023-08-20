#This script contains helper functions for adjusting the pairwise likelihood to get asymptotic guarantees. We adjust the pairwise likelihood in the way
#described in the paper "Inference for Clustered Data using the Independence Loglikelihood" by Chandler and Bate. The adjustment guarantees that the pairwise
#maximum likelihood estimator has the intended asymptotic convergenece to the true parameter with the correct asypmtotic variance (the inverse of the Fisher
#information). There are two methods--spectral and cholesky. This script contains functions to produce the unadjusted, adjusted (cholesky and spectral)
# pairwise surfaces and compute estimators of the H and J matrix described in Chandler and Bate as well as our paper using finite differencing as well
# as other functions.
library(SpatialExtremes)
library(reticulate)
library(ramify)
library(expm)

#This function returns the pairwise likelihood function for a given spatial field and distance constraint delta.
#function parameters:
  #flattened_spatial_field: spatial field of dim (25 by 25) that is flattened to array of 625
  #dist_constraint: distance_constraint or delta used to determine how many pairs of spatial locations to use in computing pairwise likelihood
generate_pairwise_likelihood <- function(flattened_spatial_field, dist_constraint)
{
  n.rep.test <- 1
  nn <- 25
  n.size <- 20
  x <- y <- seq(0, n.size, length = nn)
  coord <- expand.grid(x, y)
  coord_mat = cbind(coord$Var1, coord$Var2)
  weights <- as.numeric(distance(coord_mat) < dist_constraint)
  fit <- fitmaxstab(flattened_spatial_field, coord_mat, "brown",
                    weights = weights, 
                    method = "L-BFGS-B", 
                    control = list(pgtol = 100, maxit = 0),
                    start = list(range = 1, smooth = 1))
  
  pwl <- function(theta)
  {
    return(-1*fit$nllh(theta))
  }
  return(pwl)
}

#This function produces a pairwise likelihood surface given a pairwise likelihood function for a specific spatial field 
# and distance constraint delta
  #pwl_for_spatial_field: pairwise likelihood function for a specific spatial field and distance constraint delta
  #possible_smooths: values for the smoothness parameter for creating the grid over the parameter space to evaluate the surface
  #possible_ranges: values for the range parameter for creating the grid over the parameter space to evaluate the surface
produce_pairwise_likelihood_surface <- function(pwl_for_spatial_field, possible_smooths, possible_ranges)
{
  pwl_surface <- matrix(0, nrow = length(possible_smooths), ncol = length(possible_ranges))
  for(i in 1:length(possible_ranges))
  {
    for(j in 1:length(possible_smooths))
    {
      current_theta <- c(possible_ranges[i], possible_smooths[j])
      #pretty sure it's range first
      pwl_surface[i,j] <- pwl_for_spatial_field(current_theta)
    }
  }
  return(pwl_surface)
}

#This function uses finite differencing (right sided) to compute the gradient
 #function parameters:
  #f: function
  #x: vector to evaluate the gradient at
  #h: the difference used to compute the gradient of f at x
compute_gradient <- function(f, x, h)
{
  n <- length(x)
  gradient <- replicate(n, 0)
  for(i in 1:n)
  {
    ei <- replicate(n,0)
    ei[i] <- 1
    right_side <- f(x+h*ei)
    gradient[i] <- (right_side-f(x))/h
  }
  return(gradient)
}

#This function uses finite differencing (upper right) to compute the hessian
#function parameters:
#f: function
#x: vector to evaluate the hessian at
#h: the difference used to compute the hessian of f at x
compute_hessian <- function(f,x,h)
{
  n <- length(x)
  hessian <- matrix(0, ncol = n, nrow = n)
  for(i in 1:n)
  {
    ei <- replicate(n,0)
    ei[i] <- 1
    for(j in 1:n)
    {
      ej <- replicate(n,0)
      ej[j] <- 1
      upper_right <- f(x+h*ei+h*ej)
      upper <- f(x+h*ei)
      right <- f(x+h*ej)
      hessian[i,j] <- (upper_right - upper - right + f(x))/(h**2)
    }
  }
  return(hessian)
}

#This function uses finite differencing (lower right) to compute the hessian
#function parameters:
#f: function
#x: vector to evaluate the hessian at
#h: the difference used to compute the hessian of f at x
compute_lower_right_hessian <- function(f,x,h)
{
  n <- length(x)
  hessian <- matrix(0, ncol = n, nrow = n)
  for(i in 1:n)
  {
    ei <- replicate(n,0)
    ei[i] <- 1
    #i is horizontal (right +)
    for(j in 1:n)
    {
      ej <- replicate(n,0)
      ej[j] <- 1
      lower_right <- f(x+h*ei-h*ej)
      right <- f(x+h*ei)
      lower <- f(x-h*ej)
      hessian[i,j] <- (lower_right - lower - right + f(x))/(h**2)
    }
  }
  return(hessian)
}

#This function uses finite differencing to compute the hessian and checks whether the 
#resulting hessian is negative definite, if not another finite differencing technique is
#used, the order is upper right, lower left, lower right, upper left
#function parameters:
#f: function
#x: vector to evaluate the hessian at
#h: the difference used to compute the hessian of f at x
compute_hessian_with_different_deltas <- function(f,x,h)
{
  n <- length(x)
  hessian <- compute_hessian(f,x,h)
  hessian_eigenvalues <- (eigen(hessian))$values
  print(hessian_eigenvalues)
  if((hessian_eigenvalues[1] > 0) | (hessian_eigenvalues[2] > 0))
  {
    hessian <- compute_hessian(f,x,-1*h)
    hessian_eigenvalues <- (eigen(hessian))$values
    print(hessian_eigenvalues)
    if((hessian_eigenvalues[1] > 0) | (hessian_eigenvalues[2] > 0))
    {
      hessian <- compute_lower_right_hessian(f,x,h)
      hessian_eigenvalues <- (eigen(hessian))$values
      print(hessian_eigenvalues)
      if((hessian_eigenvalues[1] > 0) | (hessian_eigenvalues[2] > 0))
      {
        hessian <- compute_lower_right_hessian(f,x,-1*h)
        hessian_eigenvalues <- (eigen(hessian))$values
        print(hessian_eigenvalues)
        if((hessian_eigenvalues[1] > 0) | (hessian_eigenvalues[2] > 0))
        {
          return(NULL)
        }
      }
    }
  }
  return(hessian)
}

#This function computes the gradients of all the given spatial fields at theta
#for pairwise likelihood
  #function parameters:
    #spatial_images: the spatial fields used to generate the pairwise likelihood functions
    #theta: the parameter to evaluate the gradient of the pairwise likelihood at
    #h: the difference used to compute gradient via finite differencing
    #dist_constraint: the distance constraint delta used to produce the pairwise likelihood function
compute_gradients <- function(spatial_images, theta, h, dist_constraint)
{
  m <- dim(spatial_images)[1]
  parameter_length <- length(theta)
  gradients <- array(0, dim = c(m, parameter_length))
  for(i in 1:m)
  {
    spatial_image <- spatial_images[i,,]
    flattened_spatial_image <- t(flatten(spatial_image))
    pwl_for_y <- generate_pairwise_likelihood(flattened_spatial_image, dist_constraint)
    gradients[i,] <- compute_gradient(pwl_for_y, theta, h)
  }
  return(gradients)
}

#This function computes the gradients of the pairwise likelihood functions provided
#from the spatial fields and given distance constraint and computes the variance of the
#gradients, the matrix J
  #function parameters:
  #spatial_images: the spatial fields used to generate the pairwise likelihood functions
  #theta: the parameter to evaluate the gradient of the pairwise likelihood at
  #h: the difference used to compute gradient via finite differencing
  #dist_constraint: the distance constraint delta used to produce the pairwise likelihood function 
compute_J <- function(spatial_images, theta, h, dist_constraint)
{
  m <- dim(spatial_images)[1]
  parameter_length <- length(theta)
  gradients <- array(0, dim = c(m, parameter_length))
  for(i in 1:m)
  {
    spatial_image <- spatial_images[i,,]
    flattened_spatial_image <- t(flatten(spatial_image))
    pwl_for_y <- generate_pairwise_likelihood(flattened_spatial_image, dist_constraint)
    gradients[i,] <- compute_gradient(pwl_for_y, theta, h)
  }
  J <- var(gradients)
  return(J)
}


#This function computes the hessians of all the given spatial fields at theta
#for pairwise likelihood
  #function parameters:
    #spatial_images: the spatial fields used to generate the pairwise likelihood functions
    #theta: the parameter to evaluate the hessian of the pairwise likelihood at
    #h: the difference used to compute hessian via finite differencing
    #dist_constraint: the distance constraint delta used to produce the pairwise likelihood function
compute_hessians <- function(spatial_images, theta, h, dist_constraint)
{
  m <- dim(spatial_images)[1]
  parameter_length <- length(theta)
  hessians <- array(0, dim = c(m, parameter_length, parameter_length))
  for(i in 1:m)
  {
    print(i)
    spatial_image <- spatial_images[i,,]
    flattened_spatial_image <- t(flatten(spatial_image))
    pwl_for_y <- generate_pairwise_likelihood(flattened_spatial_image, dist_constraint)
    hessians[i,,] <- compute_hessian(pwl_for_y, theta, h)
  }
  return(hessians)
}

#This function computes the estimates for H and J for the given spatial fields at the given theta and 
#for the given distance constraint
  #function parameters:
    #function parameters:
      #spatial_images: the spatial fields used to generate the pairwise likelihood functions
      #theta: the parameter to evaluate the hessian of the pairwise likelihood at
      #h: the difference used to compute hessian via finite differencing
      #dist_constraint: the distance constraint delta used to produce the pairwise likelihood function
      #m: the total number of spatial fields 
compute_H_with_different_deltas_and_J <- function(spatial_images, theta, h, dist_constraint, m)
{
  parameter_length <- length(theta)
  H <- array(0, dim = c(parameter_length, parameter_length))
  gradients <- array(0, dim = c(m, parameter_length))
  l <- 0
  for(i in 1:m)
  {
    spatial_image <- spatial_images[i,,]
    flattened_spatial_image <- t(flatten(spatial_image))
    pwl_for_y <- generate_pairwise_likelihood(flattened_spatial_image, dist_constraint)
    hessian <- compute_hessian_with_different_deltas(pwl_for_y, theta, h)
    gradients[i,] <- compute_gradient(pwl_for_y, theta, h)
    if(!is.null(hessian))
    {
      l <- l + 1
      H <- H + hessian
    }
  }
  H <- (1/l)*H
  J <- var(gradients)
  return(list(H,J))
}

#This is a function that produces the maximum likelihood estimate of a given field
  #function parameters:
    #pwl_surface: the surface for which we want to find the maximizer
    #possible_smooths: values for the smoothness parameter for creating the grid over the parameter space to evaluate the surface
    #possible_ranges: values for the range parameter for creating the grid over the parameter space to evaluate the surface
produce_MLE <- function(pwl_surface, possible_smooths, possible_ranges)
{
  max_index <- which(pwl_surface == max(pwl_surface), arr.ind = TRUE)
  mle <- c(possible_ranges[max_index[1,1]], possible_smooths[max_index[1,2]])
  return(mle)
}

#Given HI, the original estimated H (i.e. the H computed from the unadjusted pairwise likelihoods), and 
#J, the variance of the gradients, compute the adjusted H (the matrix with the correct curvature of the adjusted pairwise likelihood
#that guarantees the asymptotic properties of the pairwise likelihood estimator)
  #function parameters:
    #J: the matrix estimate of the variance of the gradients
    #HI: the matrix estimate of the unadjusted H matrix (expectation of the hessians)
produce_HA <- function(J, HI)
{
  chol_minus_HI <- chol(-1*HI)
  HIinv <- -1*chol2inv(chol_minus_HI)
  HAinv <- -1*HIinv %*% J %*% HIinv
  chol_minus_HAinv <- chol(-1*HAinv)
  HA <- -1*chol2inv(chol_minus_HAinv)
  return(HA)
}

#This function produces the C matrix described in Chandler and Bate as well as our paper for the horizontal adjustment of 
#the parameters using the cholesky method.
  #function parameters:
    #HI: the matrix estimate of the unadjusted H matrix (expectation of the hessians)
    #HA: the matrix estimate of the adjusted H matrix (expectation of the hessians)
produce_cholesky_C <- function(HI, HA)
{
  chol_minus_HI <- chol(-1*HI)
  MI <- chol_minus_HI
  MA <- chol(-HA)
  C_cholesky <- solve(MI) %*% MA
  return(C_cholesky)
}

#This function produces the adjusted pairwise likelihood function using the C matrix produced via the Cholesky method. See Chandler and Bate for more details.
  #function parameters:
    #pairwise_likelihood_function: function to adjust
    #cholesky_C: the matrix used to transform the parameters (matrix computed using H and J and Cholesky method)
    #mle: the maximizer of the unadjusted pairwise likelihood surface
cholesky_adjusted_pairwise_likelihood_function <- function(pairwise_likelihood_function, cholesky_C, mle)
{
  adj_pairwise_function <- function(theta)
  {
    adjusted_theta <- mle + cholesky_C%*%(theta - mle)
    adj_pairwise_likelihood <- pairwise_likelihood_function(adjusted_theta)
    return(adj_pairwise_likelihood)
  }
  return(adj_pairwise_function)
}

#This function produces the C matrix described in Chandler and Bate as well as our paper for the horizontal adjustment of 
#the parameters using the spectral method.
  #function parameters:
    #HI: the matrix estimate of the unadjusted H matrix (expectation of the hessians)
    #HA: the matrix estimate of the adjusted H matrix (expectation of the hessians)
produce_spectral_C <- function(HI, HA)
{
  z <- eigen(-HI, symmetric = TRUE)
  #eigendecomposition (because HI and HA are symmetric matrices)
  n_pars <- 2
  MI <- z$vectors %*% diag(sqrt(z$values), n_pars, n_pars) %*% t(z$vectors)
  z <- eigen(-HA, symmetric = TRUE)
  MA <- z$vectors %*% diag(sqrt(z$values), n_pars, n_pars) %*% t(z$vectors)
  C_spectral <- solve(MI) %*% MA
}

#This function adjusts the given theta using the matrix C according to the horizontal adjustment method proposed in Chandler and Bate
 #function parameters:
  #theta: the parameter to adjust
  #C: the matrix used to transform theta
  #mle: also involved in horizontal adjustment
adjust_theta <- function(theta, C, mle)
{
  adjusted_theta <- mle + C%*%(theta-mle)
  return(adjusted_theta)
}

#This function produces the adjusted pairwise likelihood function using the C matrix produced via the spectral method. See Chandler and Bate for more details.
  #function parameters:
    #pairwise_likelihood_function: function to adjust
    #spectral_C: the matrix used to transform the parameters (matrix computed using H and J and spectral method)
    #mle: the maximizer of the unadjusted pairwise likelihood surface
spectral_adjusted_pairwise_likelihood_function <- function(pairwise_likelihood_function, spectral_C, mle)
{
  adj_pairwise_likelihood_function <- function(theta)
  {
    adjusted_theta <- mle + spectral_C%*%(theta - mle)
    adj_pairwise_likelihood <- pairwise_likelihood_function(adjusted_theta)
    return(adj_pairwise_likelihod)
  }
  return(adj_pairwise_likelihood_function)
}

#This function produces the adjusted pairwise likelihood surface for the given pairwise likelihood function and the given C matrix
#(either computed using cholesky or spectral method)
  #function parameters:
    #pwl_function: pairwise likelihood function
    #C: C matrix for the horizontal adjustment of the parameters (either cholesky or spectral method)
    #mle: mle of the unadjusted pairwise likelihood surface is needed to do the horizontal adjustment
    #possible_smooths: values for the smoothness parameter for creating the grid over the parameter space to evaluate the surface
    #possible_ranges: values for the range parameter for creating the grid over the parameter space to evaluate the surface
produce_horizontally_adjusted_pairwise_likelihood_surface <- function(pwl_function, C, mle, possible_smooths, possible_ranges)
{
  adjusted_pwl_surface <- matrix(0, nrow = length(possible_smooths), ncol = length(possible_ranges))
  
  for(i in 1:length(possible_ranges))
  {
    for(j in 1:length(possible_smooths))
    {
      current_theta <- c(possible_ranges[i], possible_smooths[j])
      adjusted_theta <- adjust_theta(current_theta, C, mle)
      adjusted_pwl_surface[i,j] <- pwl_function(adjusted_theta)
    }
  }
  return(adjusted_pwl_surface)
}