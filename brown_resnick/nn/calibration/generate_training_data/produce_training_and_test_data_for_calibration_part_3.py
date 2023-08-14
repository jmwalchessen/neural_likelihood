#This script is for obtaining classifier outputs for the training (and test) data for calibration. Calibration requires the
#classifier output and the true class label.
import numpy as np
import tensorflow 
from tensorflow import keras

local_folder = "/home/juliatest/Dropbox/likelihood_free_inference/neural_likelihood/brown_resnick/"
version = "final_version"
image_size = 25
image_name = str(image_size) + "_by_" + str(image_size)
train_or_test = "test"
number_of_reps = 50
number_of_parameters = 300
#load train/test images and parameters
train_images_file = (local_folder + 
                     "nn/calibration/data/" + image_name + "/" + version + "/" + train_or_test + 
                     "/precalibration_" + train_or_test + "_images_" + str(number_of_parameters)
                     + "_reps_" + str(number_of_reps) + ".npy")
train_images = np.load(train_images_file)
train_parameters_file = (local_folder + 
                         "nn/calibration/data/" + image_name + "/" + version + "/" + train_or_test + "/" + 
                         "precalibration_" + train_or_test + "_parameters_" + str(number_of_parameters)
                           + "_reps_" + str(number_of_reps) + ".npy")
train_parameters = np.load(train_parameters_file)
total_number = train_parameters.shape[0]

#Load the nn

json_file_name = (local_folder + "nn/" + image_name + "/" +  version 
 + "/model/" + "br_" + str(image_name) + "_" + version + "_nn.json")

weights_file_name = (local_folder + "nn/" + image_name + "/" + version
+ "/model/" + "br_" +  image_name + "_" + version + "_nn_weights.h5")

json_file = open(json_file_name, 'r')
loaded_model_json = json_file.read()
json_file.close()
parameter_classifier = keras.models.model_from_json(loaded_model_json)
parameter_classifier.load_weights(weights_file_name)


classifier_outputs = np.zeros((total_number, 2))

for i in range(0, total_number, 1000):
    current_images = (train_images[i:(i+1000),:,:,:]).reshape((1000, image_size, image_size, 1))
    current_parameters = (train_parameters[i:(i+1000),:]).reshape((1000,2))
    classifier_outputs[i:(i+1000),:] = parameter_classifier.predict([current_images, current_parameters])


np.save((local_folder + "nn/calibration/data/" + image_name + "/" + version + "/" + train_or_test + 
         "/precalibration_" + train_or_test + "_classifier_outputs_results_" + str(number_of_parameters)
           + "_reps_" + str(number_of_reps) + ".npy"), classifier_outputs)