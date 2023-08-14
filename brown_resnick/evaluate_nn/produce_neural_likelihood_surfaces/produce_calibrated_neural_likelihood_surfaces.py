#This script produces calibrated neural likelihood surfaces in the single realization case.
import numpy as np
import tensorflow
import pickle
from tensorflow import keras
import pickle

version = "final_version"

#This is the transformation of the classifier into psi (which is proportional to the likelihood)
#using platt scaling. The function transforms the calibrated classifier output for a pair of spatial
#field and parameters via the psi function
#function parameters:
    #images: numpy matrix of spatial field
    #parameters: numpy matrix of parameters
def multi_calibrated_psi(images, parameters):
    #classifier outputs
    predictions = (parameter_classifier.predict([images, parameters]))
    #logit transformation
    z_scores = (np.log(predictions/(1-predictions)))[:,0]
    #shape for logistic model
    z_scores = z_scores.reshape((-1,1))
    #make sure no infinite values
    z_scores[z_scores == np.inf] = np.amax(z_scores[z_scores != np.inf])
    z_scores[z_scores == np.NaN] = np.amax(z_scores[z_scores != np.inf])
    z_scores[z_scores == -1*np.inf] = np.amin(z_scores[z_scores != -1*np.inf])
    #get logistic model output
    classifier_outputs = (1 - logistic_regression_model.predict_proba(z_scores))
    psi_values = np.zeros(shape = (classifier_outputs.shape[0], 1))
    #apply the psi transformation
    for i in range(classifier_outputs.shape[0]):
        output = classifier_outputs[i,:]
        if(output[1] == 1):
            psi_value = (1-output[0])/output[0]
        else:
            psi_value = output[1]/(1-output[1])

        psi_values[i,:] = psi_value

    return psi_values

#Produce the calibrated neural likelihood surface for the parameter grid over the parameter space
#function parameters:
    #possible_ranges: range values on the parameter grid
    #possible_smooths: smooth values on the parameter grid
    #image: spatial field (numpy matrix)
    #n: the square root of the number of spatial observations
def produce_calibrated_psi_field(possible_ranges, possible_smooths, image, n):

    number_of_parameter_pairs = len(possible_ranges)*len(possible_smooths)
    image = image.reshape((1, n, n, 1))
    image_matrix = np.repeat(image, number_of_parameter_pairs, axis  = 0)
    ranges = (np.repeat(np.asarray(possible_ranges), 
                        len(possible_smooths), axis = 0)).reshape((number_of_parameter_pairs, 1))
    smooths = []
    smooths = (np.array(sum([(smooths + possible_smooths) for i in 
                             range(0, len(possible_ranges))], []))).reshape((number_of_parameter_pairs,1))
    parameter_matrix = np.concatenate([ranges, smooths], axis = 1)
    calibrated_psi_field = (multi_calibrated_psi(image_matrix, parameter_matrix)).reshape((len(possible_smooths), 
                                                                                len(possible_ranges)))

    return calibrated_psi_field

#Load the evaluation images
n = 25
image_name = str(n) + "_by_" + str(n)
number_of_parameters = 100
number_of_reps = 200
local_folder = "/home/juliatest/Desktop/likelihood_free_inference/neural_likelihood/brown_resnick/"
data_file_name = (local_folder + "evaluate_nn/generate_data/data/" + image_name + 
                  "/single/reps/" + str(number_of_reps) + "/evaluation_images_10_by_10_density_" + 
                  image_name + "_" + str(number_of_reps) + ".npy")
evaluation_images = np.load(data_file_name)
possible_ranges = [.05*i for i in range(1, 41)]
possible_smooths = [.05*i for i in range(1, 41)]

#Load the nn
json_file_name = (local_folder + "nn/" + image_name + "/" + version +
"/model/br_" + image_name + "_" + version + "_nn.json")

weights_file_name = (local_folder + "nn/" + image_name + "/" + version +
"/model/br_" + image_name + "_" + version + "_nn_weights.h5")

json_file = open(json_file_name, 'r')
loaded_model_json = json_file.read()
json_file.close()
parameter_classifier = keras.models.model_from_json(loaded_model_json)
parameter_classifier.load_weights(weights_file_name)

#load the logistic regression model
logistic_regression_model_file_name = (local_folder + "evaluate_nn/calibration/model/" 
                                       + image_name + "/" + version + 
                                       "/logistic_regression_model_wtih_logit_transformation.pkl")
with open(logistic_regression_model_file_name, 'rb') as f:
    logistic_regression_model = pickle.load(f)

calibrated_psi_fields = np.zeros((number_of_parameters, number_of_reps, len(possible_smooths), 40))

#Produce calibrated neural likelihood surfaces and parameter estimates for evaluation images
for i in range(0, number_of_parameters):
    for j in range(0, number_of_reps):

        current_image = evaluation_images[i,j,:,:]
        calibrated_psi_fields[i,j,:,:] = produce_calibrated_psi_field(possible_ranges, 
                                                           possible_smooths, current_image, n)


calibrated_psi_field_file = ("data/" + image_name + "/" + version +
"/calibrated/single/reps/" + str(number_of_reps) + "/calibrated_neural_likelihood_surfaces_10_by_10_density_" 
+ image_name + "_image_" + str(number_of_reps) + ".npy")
np.save(calibrated_psi_field_file, calibrated_psi_fields)