#This file is to produce the logistic regression model for calibrating the classifier outputs. We use a logistic regression
#model with a transformation of the classifier outputs via the logit transformation in order to transform the domain of the
#classifier probabilites from [0,1] to the real numbers (-inf,+inf)
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

#First load training data
image_size = 25
local_folder = "/home/juliatest/Dropbox/likelihood_free_inference/neural_likelihood/gaussian_process"
image_name = str(image_size) + "_by_" + str(image_size)
version = "final_version"

#Load the numpy matrix of classifier results for training data (training data for calibration)
train_predictions = np.load((local_folder + "/nn/models/calibration/data/" + 
                             image_name + "/" + version + "/train/precalibration_train_" + version + 
                             "_classifier_results_3000_reps_50.npy"))
#Load the true labels (class 0 is the dependent class and class 1 is the independent class)
train_true_labels = np.load((local_folder + "/nn/models/calibration/data/" + image_name + "/" +
                             version + "/train/precalibration_train_classes_3000_reps_50.npy"))[:,0]
#Load the numpy matrix of classifier results for test data (for the reliability diagram)
test_predictions = np.load((local_folder + "/nn/models/calibration/data/" + image_name +
                             "/" + version + "/test/precalibration_test_" + version + 
                             "_classifier_results_300_reps_50.npy"))
#Load the true labels (class 0 is the dependent class and class 1 is the independent class)
test_true_labels = (np.load(local_folder + "nn/models/calibration/data/" + image_name + "/" +
                            version + "/test/precalibration_test_classes_300_reps_50.npy"))[:,0]

#Apply logit transformation to the classifier results (train)
train_logit = (np.log(train_predictions/(1-train_predictions)))[:,0]
#Reshape the results for the logistic regression model
train_logit = train_logit.reshape((-1,1))
#Apply logit transformation to the the classifier results (test)
test_logit = (np.log(test_predictions/(1-test_predictions)))[:,0]
#Reshape the results for the logistic regression model
test_logit = test_logit.reshape((-1,1))

#Make sure there are no Nan or inf values due to the logit transformation
train_logit[train_logit == np.inf] = np.amax(train_logit[train_logit != np.inf])
train_logit[train_logit == np.NaN] = np.amax(train_logit[train_logit != np.inf])
train_logit[train_logit == -1*np.inf] = np.amin(train_logit[train_logit != -1*np.inf])

test_logit[test_logit == np.inf] = np.amax(test_logit[test_logit != np.inf])
test_logit[test_logit == np.NaN] = np.amax(test_logit[test_logit != np.inf])
test_logit[test_logit == -1*np.inf] = np.amin(test_logit[test_logit != -1*np.inf])

#Train logistic regression model on the training data
logistic_regression_model = LogisticRegression(random_state=0, 
                                               class_weight='balanced').fit(train_logit, train_true_labels)


logistic_regression_filename = (local_folder + "nn/models/calibration/model/" + image_name + "/"
                                + version + "/logistic_regression_model_wtih_logit_transformation.pkl")

with open(logistic_regression_filename, "wb") as file:
    pickle.dump(logistic_regression_model, file)
