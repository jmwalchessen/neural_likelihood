# Neural Likelihood for Gaussian Processes
Source code for:

J. Walchessen, A. Lenzi, and M. Kuusela. Neural Likelihood Surfaces for Spatial Processes with Computationally Intensive or Intractable Likelihoods. Preprint arXiv:2305.04634 [stat.ME], 2023. [arxiv preprint](https://arxiv.org/abs/2305.04634)

Contact Julia Walchessen at jwalches@andrew.cmu.edu with any questions.



## Getting Started
This document contains the code for each case study in "Neural Likelihood Surfaces for Spatial Processes with Computationally Intensive or Intractable Likelihoods." Information about each folder and its contents in the repository. To quickly understand what a folder pertains to, please scroll down to find the information. **The package environment for this project is contained in requirements.txt**

## Gaussian Process

### nn
This folder contains code to 1. generate data to train the neural network (*generate_nn_data* folder) and 2. train the neural network (*train_nn* folder) and 3. calibrate the neural network (*calibration* folder) as well as the neural network (models folder).

#### generate_nn_data

##### generate_nn_data.R
The script **generate_nn_data.R** simulates training and validation data to train the neural network. The user needs to specify the parameter space (what the boundaries of the parameter space are), how many parameters to sample from this parameter space, and how many spatial field realizations for each of the sampled parameters should be simulated. The script produces json files for each of the sample parameters (total = number of sampled parameters). Each of the json files is a list of the paired spatial field realization and parameter (permuted or not) and the class of the pair.

##### gaussian_process_data_shaping.py

The script **gaussian_process_data_shaping.py** processes the json files produced by generate_nn_data into three numpy matrices: 1. spatial field realizations 2. and their corresponding paired parameters 3. and the classes which the paired spatial field and parameter belong to.

#### train_nn

##### gp_nn.py
The script **gp_nn.py** constructs the neural network using a specific architecture and trains the neural network with tuning parameters (learning rate schedule and batch size) on a single gpu. The data used to train the neural network comes from the *models/25_by_25/version_x/data/train* folder and the *models/25_by_25/version_x_data/validation* folder. For the sake of keeping this repository small, there is no data in these two folders.

##### gp_nn_with_distributed_training.py

The script **gp_nn_with_distributed_training.py** constructs the neural network using a specific architecture and trains the neural network with tuning parameters (learning rate schedule and batch size) on multiple gpus. The data used to train the neural network comes from the *models/25_by_25/version_x/data/train* folder and the *models/25_by_25/version_x_data/validation* folder. For the sake of keeping this repository small, there is no data in these two folders.

#### models

This folder stores the models produced from **gp_nn.py/gp_nn_with_distributed_training.py** as well as diagnostics for these models. Our final model is stored in the folder *25_by_25/final_version*. Within this folder, the architecture (**gp_25_by_25_final_version_nn.json**) and the weights (**gp_25_by_25_final_version_nn_weights.h5**) are stored in the folder *model*. The folders *accuracy* and *loss* contain visualizations of the training and validation accuracy and loss. The folder *data* contains the training and validation data in the form of numpy matrices of the parameters, spatial fields, and classes which trained the neural network. Note that the models stored in this folder are uncalibrated.

#### calibration

This folder contains scripts to 1. generate training data for calibration and to 2. calibrate the model via Platt scaling (a form of logistic regression).

##### generate_training_data

To train the logistic regression model, we need training data of the following form (uncalibrated class probabilities (i.e. the output of the uncalibrated neural network $\hat{h}(y_{i,j},\theta_{i})$ for pairs of spatial field realizations $y_{i,j}$ and $\theta_{i}$; and the corresponding class $C_{i,j}$ the pair belongs to:

$
\{(\hat{h}(y_{i,j},\theta_{i}), C_{i,j})\}_{i\in [m],j\in [n]}
$

where $n$ is the number of spatial field realizations $y$ per each of the $m$ sampled parameters. What we do to generate this training data is 1. simulate m = 3000 parameters &theta; from the parameter space 2. simulate n = 50 spatial field realizations y for each of the m = 3000 parameters &theta; 3. form data for class one and two using the previously simulated pairs of y and &theta; 4. put this data through the uncalibrated classifier/neural network to get out uncalibrated class probabilities 5. pair each uncalibrated class probability with the true class.

###### produce_training_and_test_data_for_calibration_part_1.R

This script produces the sampled spatial fields y and the samples parameters &theta; (involving steps 1,2) and saves the data in lists in json files. This script is similar to the script **generate_nn_data.R** in the folder *generate_nn_data*.

###### produce_training_and_test_data_for_calibration_part_2.py

This python script is for processing the json files produced from **produce_training_and_test_data_for_calibration_part_2.R** into numpy matrices for calibration. Specifically, three numpy matrices (parameters, spatial fields, and classes) will be saved in the folder *calibration/data/25_by_25/final_version/train* or *calibration/data/25_by_25/final_version/test*. This script is similar to the script **gaussian_process_data_shaping.py** in the folder *generate_nn_data*.

###### produce_training_and_test_data_for_calibration_part_3.py

This script is for obtaining classifier outputs for the training (and test) data for calibration. Calibration requires the classifier output and the true class label. The classifier outputs will be saved as numpy matrices in the folder *calibration/data/25_by_25/final_version/train* (or *calibration/data/25_by_25/final_version/test*).

##### produce_calibration_model.py

This script loads the data from the folder *calibration/data/25_by_25/final_version/train* and uses the data to train a logistic regression model which is then saved in the folder *model/25_by_25/final_version/logistic_regression_model_with_logit_transformation.pkl*. The test data will be used in the folder *evaluate_nn* to produce reliability diagrams which illustrate how effective calibration is in achieving calibrated class probabilities.


### evaluate_nn


#### generate_data

To evaluate our method in the Gaussian process case, we create an evaluation data set described in Section 4.1 of our paper. This evaluation data set consisters of 200 spatial field realizations per each parameter on a $10\times 10$ grid over the parameter space $(0,2)\times (0,2)$. For evaluating neural likelihood, we generally focus on the single realization case (neural likelihood surfaces, point estimates, and confidence regions constructed from a single spatial field realization). Yet, we do evaluate the performance of neural likelihood in producing parameter estimates in the multple iid realization case. As such, there are two R scripts to generate evaluation data in the single and multiple realization cases.

##### generate_evaluation_data_for_single_realization_case.R

This R file is a script to generate evaluation data for evaluating the neural likelihood surfaces, parameter estimates, and confidence regions for the single realization case. The evaluation
data consists of spatial fields and the corresponding parameters which generated the spatial fields and the log likelihood field (over the parameter space) for
the spatial field. The evaluation data is n single realizations per m parameters where the m parameters come from a grid over the parameter space. The evaluation data is saved as json files (the number of json files is equal to the number of parameters on the grid).

##### generate_evaluation_data_for_multi_realization_case.R

This R file is a script to generate evaluation data for evaluating the neural likelihood parameter estimates for the multiple realization case. The evaluation
data consists of spatial fields and the corresponding parameters which generated the spatial fields and the log likelihood field (over the parameter space) for
the spatial field. The evaluation data is n realizations of 5 replications of the spatial field per m parameters where the m parameters come from a grid over the parameter space. The evaluation data is saved as json files (the number of json files is equal to the number of parameters on the grid).

##### gaussian_process_data_shaping.py

This python script processes the json files produced by either **generate_evaluation_data_for_single_realization_case.R** or **generate_evaluation_data_for_multi_realization_case.R** into numpy matrices (images, parameters, log likelihood fields) which are saved in the folders *evaluate_nn/generate_data/data/25_by_25/single/reps/200* or *evaluate_nn/generate_data/data/25_by_25/multi/5/reps/200*.

#### produce_exact_likelihood_estimates

##### produce_exact_likelihood_estimates.py

This python scripts takes the log likelihood fields for the evaluation data stored in the folder *evaluate_nn/generate_data/data/25_by_25/single/reps/200* or evaluate_nn/generate_data/data/25_by_25/multi/5/reps/200* and produces parameter estimates using these log likelihood fields. The parameter estimates for the evaluation data are stored in the folders *produce_exact_likelihood_estimates/data/25_by_25/single/reps/200* and *produce_exact_likelihood_estimates/data/25_by_25/multi/5/reps/200*.


#### produce_neural_likelihood_surfaces

This folder contains scripts to generate neural likelihood surfaces for the evaluation data in the folder *evaluate_nn/generate_data/data/25_by_25/single/reps/200* or evaluate_nn/generate_data/data/25_by_25/multi/5/reps/200*. To evaluate the effect of calibration, we produce both uncalibrated and calibrated neural likelihood surfaces for both the single and multiple realization cases. Hence, we have four different scripts for these four different cases. The neural likelihood surfaces once produced are stored in the folder *evaluate_nn/produce_neural_likelihood_surfaces/data/25_by_25/final_version*.

#### produce_neural_likelihood_estimates

This python scripts takes the neural likelihood fields for the evaluation data stored in the folder *evaluate_nn/produce_neural_likelihood_surfaces/data/25_by_25/final_version* and produces parameter estimates for both the single and multiple realization case. Since calibration does not affect the point estimates, we use the uncalibrated neural likelihood surfaces to produce the point estimates. The parameter estimates for the evaluation data are stored in the folder *evaluate_nn/produce_neural_likelihood_surfaces/data/25_by_25/final_version*.

#### timing_studies

This is a timing study for the time to evaluate an exact or neural likelihood surface on average on the same fixed grid over the parameter space. There are two scripts--one for timing the exact likelihood surface and one for timing the neural likelihood surface. Both the neural and exact likelihood surfaces are computed using the full resources of my laptop which has an Intel Core i7-10875H processor with eight cores, each with two threads, and a NVIDIA GeForce RTX 2080 Super. To use the full resources of my laptop, parallel computing is utilized. The times to produce each of the 50 fields is stored in the folder *timing_studies/data/25_by_25*.

#### visualizations

This folder contains many subfolders for producing the different visualizations that appear in our paper---reliability diagrams to understand the effect of calibration as well as neural and exact likelihood surfaces and approximate confidence regions and point estimates.

##### produce_reliability_diagrams


The notebook **produce_reliability_diagram.ipynb** produces a reliability diagram (empirical class probablity as function of predicted class probability) before and after calibration. The closer the function is to the identity after calibration the better the calibration. The figure configuration comes from https://github.com/hollance/reliability-diagrams but how we compute predicted and empirical class probability is different. The reliability diagrams are stored in the folder *visualizations/produce_reliability_diagrams/diagrams/25_by_25/final_version*.

##### visualize_surfaces

This folder contains jupyter notebooks for visualizing the exact, uncalibrated, and calibrated neural likelihood surfaces for the evaluation data in the single realization case.

##### visualize_approximate_confidence_regions

This folder contains jupyter notebooks for visualizing the exact, uncalibrated, and calibrated neural likelihood surfaces and the corresponding 95% approximate confidence regions for the evaluation data in the single realization case. There is also a jupyter notebook which plots the surfaces and 95% approximate confidence regions for exact, uncalibrated, and calibrated neural likelihood side by side. This visualizations appears in our paper.

##### visualize_empirical_coverage_and_confidence_region_area

This folder contains a jupyter notebook for visualizing the empirical coverage and confidence region area of exact, uncalibrated, and calibrated neural likelihood side by side. These two visualizations (empirical coverage and confidence region area) appear in our paper.

##### visualize_parameter_estimates

To visualize the point estimates across the parameter space $\Theta = (0,2)\times (0,2)$, we display a $4\times 4$ grid of plots in which each plot contains $200$ neural and exact point estimates for the 200 spatial field realizations generated using the parameter indicated by the black/blue star. This figure appears in our paper.


## Brown--Resnick

### nn

##### generate_nn_data.R
The script **generate_nn_data.R** simulates training and validation data to train the neural network. The user needs to specify the parameter space (what the boundaries of the parameter space are), how many parameters to sample from this parameter space, and how many spatial field realizations for each of the sampled parameters should be simulated. The script produces json files for each of the sample parameters (total = number of sampled parameters). Each of the json files is a list of the paired spatial field realization and parameter (permuted or not) and the class of the pair.

##### brown_resnick_data_shaping.py

The script **brown_resnick_data_shaping.py** processes the json files produced by generate_nn_data into three numpy matrices: 1. spatial field realizations 2. and their corresponding paired parameters 3. and the classes which the paired spatial field and parameter belong to.

#### train_nn

##### br_nn.py
The script **br_nn.py** constructs the neural network using a specific architecture and trains the neural network with tuning parameters (learning rate schedule and batch size) on a single gpu. The data used to train the neural network comes from the *models/25_by_25/version_x/data/train* folder and the *models/25_by_25/version_x_data/validation* folder. For the sake of keeping this repository small, there is no data in these two folders.

##### br_nn_with_distributed_training.py

The script **br_nn_with_distributed_training.py** constructs the neural network using a specific architecture and trains the neural network with tuning parameters (learning rate schedule and batch size) on multiple gpus. The data used to train the neural network comes from the *models/25_by_25/version_x/data/train* folder and the *models/25_by_25/version_x_data/validation* folder. For the sake of keeping this repository small, there is no data in these two folders.

#### calibration

This folder contains scripts to 1. generate training data for calibration and to 2. calibrate the model via Platt scaling (a form of logistic regression).

##### generate_training_data

To train the logistic regression model, we need training data of the following form--uncalibrated class probabilities i.e. the output of the uncalibrated neural network $\hat{h}(y_{i,j},\theta_{i})$ for pairs of spatial field realizations $y_{i,j}$ and $\theta_{i}$ and the corresponding class $C_{i,j}$ the pair belongs to:

$\{(\hat{h}(y_{i,j},\theta_{i}), C_{i,j})\}_{i\in [m],j\in [n]}$

where $n$ is the number of spatial field realizations $y$ per each of the $m$ sampled parameters. What we do to generate this training data is 1. simulate m = 3000 parameters &theta; from the parameter space 2. simulate n = 50 spatial field realizations y for each of the m = 3000 parameters &theta; 3. form data for class one and two using the previously simulated pairs of y and &theta; 4. put this data through the uncalibrated classifier/neural network to get out uncalibrated class probabilities 5. pair each uncalibrated class probability with the true class.

###### produce_training_and_test_data_for_calibration_part_1.R

This script produces the sampled spatial fields y and the samples parameters &theta; (involving steps 1,2) and saves the data in lists in json files. This script is similar to the script **generate_nn_data.R** in the folder *generate_nn_data*.

###### produce_training_and_test_data_for_calibration_part_2.py

This python script is for processing the json files produced from **produce_training_and_test_data_for_calibration_part_2.R** into numpy matrices for calibration. Specifically, three numpy matrices (parameters, spatial fields, and classes) will be saved in the folder *calibration/data/25_by_25/final_version/train* or *calibration/data/25_by_25/final_version/test*. This script is similar to the script **brown_resnick_data_shaping.py** in the folder *generate_nn_data*.

###### produce_training_and_test_data_for_calibration_part_3.py

This script is for obtaining classifier outputs for the training (and test) data for calibration. Calibration requires the classifier output and the true class label. The classifier outputs will be saved as numpy matrices in the folder *calibration/data/25_by_25/final_version/train* (or *calibration/data/25_by_25/final_version/test*).

##### produce_calibration_model.py

This script loads the data from the folder *calibration/data/25_by_25/final_version/train* and uses the data to train a logistic regression model which is then saved in the folder *model/25_by_25/final_version/logistic_regression_model_with_logit_transformation.pkl*. The test data will be used in the folder *evaluate_nn* to produce reliability diagrams which illustrate how effective calibration is in achieving calibrated class probabilities.

### evaluate_nn

#### generate_data

To evaluate our method in the Brown--Resnick case, we create an evaluation data set described in Section 4.1 of our paper. This evaluation data set consisters of 200 spatial field realizations per each parameter on a $10\times 10$ grid over the parameter space $(0,2)\times (0,2)$. For evaluating neural likelihood, we generally focus on the single realization case (neural likelihood surfaces, point estimates, and confidence regions constructed from a single spatial field realization). Yet, we do evaluate the performance of neural likelihood in producing parameter estimates in the multple iid realization case. As such, there are two R scripts to generate evaluation data in the single and multiple realization cases.

##### generate_evaluation_data_for_single_realization_case.R

This R file is a script to generate evaluation data for evaluating the neural likelihood surfaces, parameter estimates, and confidence regions for the single realization case. The evaluation
data consists of spatial fields and the corresponding parameters which generated the spatial fields. The evaluation data is n single realizations per m parameters where the m parameters come from a grid over the parameter space. The evaluation data is saved as json files (the number of json files is equal to the number of parameters on the grid).

##### generate_evaluation_data_for_multi_realization_case.R

This R file is a script to generate evaluation data for evaluating the neural likelihood parameter estimates for the multiple realization case. The evaluation data consists of spatial fields and the corresponding parameters which generated the spatial fields. The evaluation data is n realizations of 5 replications of the spatial field per m parameters where the m parameters come from a grid over the parameter space. The evaluation data is saved as json files (the number of json files is equal to the number of parameters on the grid).

##### brown_resnick_data_shaping.py

This python script processes the json files produced by either **generate_evaluation_data_for_single_realization_case.R** or **generate_evaluation_data_for_multi_realization_case.R** into numpy matrices (images and parameters) which are saved in the folders *evaluate_nn/generate_data/data/25_by_25/single/reps/200* or *evaluate_nn/generate_data/data/25_by_25/multi/5/reps/200*.

#### produce_pairwise_likelihood_surfaces

##### produce_pairwise_likelihood_surfaces_for_evaluation_data_in_the_single_realization_case.R

This script is for producing the pairwise likelihood surfaces for the evaluation data in the single realization case. Changing the distance constraint ($\delta$) given (see paper for explanation of the distance constraint $\delta$), will change the resulting pairwise likelihood surfaces. The pairwise likelihood surfaces are stored in numpy matrices (one per each parameter on the $10\times 10$ grid over the parameter space $\Theta$) in the folder *produce_pairwise_likelihood_surfaces/data/25_by_25/dist_value/single/reps/200*.


##### produce_pairwise_likelihood_surfaces_for_evaluation_data_in_the_multi_realization_case.R

This script is for producing the pairwise likelihood surfaces for the evaluation data in the multiple realization case. Changing the distance constraint ($\delta$) given (see paper for explanation of the distance constraint $\delta$), will change the resulting pairwise likelihood surfaces. The pairwise likelihood surfaces are stored in numpy matrices (one per each parameter on the $10\times 10$ grid over the parameter space $\Theta$) in the folder *produce_pairwise_likelihood_surfaces/data/25_by_25/dist_value/multi/5/reps/200*.

##### process_pairwise_likelihood_surfaces.R

This script processes the numpy matrices produced from the scripts **produce_pairwise_likelihood_surfaces_for_evaluation_data_in_the_single_realization_case.R** and **produce_pairwise_likelihood_surfaces_for_evaluation_data_in_the_multi_realization_case.R** into one numpy matrix each for all the pairwise likelihood surfaces and stores the numpy matrices in the folders *produce_pairwise_likelihood_surfaces/data/25_by_25/dist_value/multi/5/reps/200* and *produce_pairwise_likelihood_surfaces/data/25_by_25/dist_value/single/reps/200*

#### produce_pairwise_likelihood_estimates

##### produce_pairwise_likelihood_estimates.py

This python scripts takes the pairwise likelihood fields for the evaluation data stored in the folder *evaluate_nn/pairwise_likelihood_surfaces/data/25_by_25* and produces parameter estimates using these pairwise likelihood surfaces. The parameter estimates for the evaluation data are stored in the folders *produce_pairwise_likelihood_estimates/data/25_by_25*. Note that this is per distance constrain $\delta$ and for both multiple and single realization cases.


#### produce_neural_likelihood_surfaces

This folder contains scripts to generate neural likelihood surfaces for the evaluation data in the folder *evaluate_nn/generate_data/data/25_by_25/single/reps/200* or evaluate_nn/generate_data/data/25_by_25/multi/5/reps/200*. To evaluate the effect of calibration, we produce both uncalibrated and calibrated neural likelihood surfaces for both the single and multiple realization cases. Hence, we have four different scripts for these four different cases. The neural likelihood surfaces once produced are stored in the folder *evaluate_nn/produce_neural_likelihood_surfaces/data/25_by_25/final_version*.

#### produce_neural_likelihood_estimates

This python scripts takes the neural likelihood fields for the evaluation data stored in the folder *evaluate_nn/produce_neural_likelihood_surfaces/data/25_by_25/final_version* and produces parameter estimates for both the single and multiple realization case. Since calibratin, does not affect the point estimates, we use the uncalibrated neural likelihood surfaces to produce the point estimates. The parameter estimates for the evaluation data are stored in the folder *evaluate_nn/produce_neural_likelihood_surfaces/data/25_by_25/final_version*.

#### timing_studies

This is a timing study for the time to evaluate pairwise or neural likelihood surface on average on the same fixed grid over the parameter space. There are two scripts--one for timing the pairwise likelihood surface and one for timing the neural likelihood surface. Both the neural and pairwise likelihood surfaces are computed using the full resources of my laptop which has an Intel Core i7-10875H processor with eight cores, each with two threads, and a NVIDIA GeForce RTX 2080 Super. To use the full resources of my laptop, parallel computing is utilized. The times to produce each of the 50 fields is stored in the folder *timing_studies/data/25_by_25*. Note that the time to produce the pairwise likelihood surface will vary depending on the distance constraint $\delta$ used.

#### visualizations

This folder contains many subfolders for producing the different visualizations that appear in our paper---reliability diagrams to understand the effect of calibration as well as neural and pairwise likelihood surfaces and approximate confidence regions and point estimates.

##### produce_reliability_diagrams


The notebook **produce_reliability_diagram.ipynb** produces a reliability diagram (empirical class probablity as function of predicted class probability) before and after calibration. The closer the function is to the identity after calibration the better the calibration. The figure configuration comes from https://github.com/hollance/reliability-diagrams but how we compute predicted and empirical class probability is different. The reliability diagrams are stored in the folder *visualizations/produce_reliability_diagrams/diagrams/25_by_25/final_version*.

##### visualize_surfaces

This folder contains jupyter notebooks for visualizing the unadjusted and adjusted pairwise likelihoods, uncalibrated, and calibrated neural likelihood surfaces for the evaluation data in the single realization case.

##### visualize_approximate_confidence_regions

This folder contains jupyter notebooks for visualizing the unadjusted and adjusted pairwise likelihood, uncalibrated, and calibrated neural likelihood surfaces and the corresponding 95% approximate confidence regions for the evaluation data in the single realization case. There is also a jupyter notebook which plots the surfaces and 95% approximate confidence regions for exact, uncalibrated, and calibrated neural likelihood side by side. This visualizations appears in our paper.

##### visualize_empirical_coverage_and_confidence_region_area

This folder contains a jupyter notebook for visualizing the empirical coverage and confidence region area of unadjusted and adjusted pairwise likelihood, uncalibrated, and calibrated neural likelihood side by side. These two visualizations (empirical coverage and confidence region area) appear in our paper.

##### visualize_parameter_estimates

To visualize the point estimates across the parameter space $\Theta = (0,2)\times (0,2)$, we display a $4\times 4$ grid of plots in which each plot contains $200$ neural and pairwise point estimates for the 200 spatial field realizations generated using the parameter indicated by the black/blue star. This figure appears in our paper.















