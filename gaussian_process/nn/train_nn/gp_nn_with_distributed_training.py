#This python script trains the neural network (CNN) on the classification task for a machine with multiple gpus 
#(cuda devices).
import tensorflow
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers, models, Input
import os
import json

local_folder = "/home/juliatest/Dropbox/likelihood_free_inference/neural_likelihood/gaussian_process/nn/models"
image_size = 25
parameter_dimension = 2
train_total_number_of_parameters = 3000
train_total_number_of_replicates = 500
validation_total_number_of_parameters = 3000
validation_total_number_of_replicates = 50
version = "final_version"
image_name = str(image_size) + "_by_" + str(image_size)
#Numpy matrices for images, parameters, and classes are stored in multiple files for storage reasons on github.
#First, concatenate matrices for training data together.
x_train_image = np.empty((0, image_size, image_size, 1))
x_train_parameters = np.empty((0,parameter_dimension))
y_train = np.empty((0,parameter_dimension))
parts_number = 20

for i in range(0, parts_number):

    current_images_name = (local_folder + "/" + image_name + "/" + version + "/data/train/train_images_expanded_parameter_space_" + image_name
                            + "_" + str(train_total_number_of_parameters) + "_reps_" + str(train_total_number_of_replicates) 
                          + "_" + str(i) + ".npy")
    current_images = np.load(current_images_name)
    x_train_image = np.concatenate((x_train_image, current_images), axis = 0)
    current_parameters_name = (local_folder + "/" + image_name + "/" + version + "/data/train/train_parameters_expanded_parameter_space_" + 
                               image_name + "_" + str(train_total_number_of_parameters) + "_reps_" + 
                               str(train_total_number_of_replicates) + "_" + str(i) + ".npy")
    current_parameters = np.load(current_parameters_name)
    x_train_parameters = np.concatenate((x_train_parameters, current_parameters), axis = 0)
    current_classes_name = (local_folder + "/" + image_name + "/" + version + "/data/train/train_classes_expanded_parameter_space_" + image_name
                             + "_" + str(train_total_number_of_parameters) + "_reps_" + 
                             str(train_total_number_of_replicates) + "_" + str(i) + ".npy")
    current_classes = np.load(current_classes_name)
    y_train = np.concatenate((y_train, current_classes), axis = 0)


validation_images_name = (local_folder + "/" + image_name + "/" + version + "/data/validation/validation_images_expanded_parameter_space_"
                         + image_name + "_" + str(validation_total_number_of_parameters) + "_reps_" + 
                         str(validation_total_number_of_replicates) + ".npy")
x_val_image = np.load(validation_images_name)
validation_parameters_name = (local_folder + "/" + image_name + "/" + version + 
                              "/data/validation/validation_parameters_expanded_parameter_space_" + image_name + "_" 
                              + str(validation_total_number_of_parameters) + "_reps_" + 
                              str(validation_total_number_of_replicates) + ".npy")
x_val_parameters = np.load(validation_parameters_name)
validation_classes_name = (local_folder + "/" + image_name + "/" + version + 
                              "/data/validation/validation_classes_expanded_parameter_space_" + image_name + "_" 
                              + str(validation_total_number_of_parameters) + "_reps_" + 
                              str(validation_total_number_of_replicates) + ".npy")
y_val = np.load(validation_classes_name)


#For distributed training with one machine and multiple gpus, collect cuda devices/gpus and then distribute model and
#parameters to each of the gpus and train synchronously.
strategy = tensorflow.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

#Distribute model to all gpus.
with strategy.scope():

    #Build convolutional part of the nn. The convolutional part processes the spatial image and outputs a flatten vector.
    conv = models.Sequential()
    conv.add(layers.Conv2D(128, (3, 3), activation = 'relu', input_shape = (image_size, image_size, 1)))
    conv.add(layers.MaxPooling2D((2, 2), padding = "same"))
    conv.add(layers.Conv2D(128, (3, 3), activation = 'relu'))
    conv.add(layers.MaxPooling2D((2,2), padding = "same"))
    conv.add(layers.Conv2D(16, (3, 3), activation = 'relu'))
    conv.add(layers.MaxPooling2D((2,2), padding = "same"))
    conv.add(layers.Flatten())

    #Concatenate the parameters (2-d vector) to the flatten vector (64-d vector) produced from the convolutional part of the nn.
    #Concatenation results in a 66 dimension vector
    parameters_input = keras.layers.Input(shape = (parameter_dimension,))
    merged_output = layers.concatenate([conv.output, parameters_input])

    #This is the fully connected part of the nn which processes both the spatial image information and the parameter.
    model_combined = models.Sequential()
    model_combined.add(layers.Dense(64, input_shape = (66,)))
    model_combined.add(layers.Activation('relu'))
    model_combined.add(layers.Dense(16))
    model_combined.add(layers.Activation('relu'))
    model_combined.add(layers.Dense(8))
    model_combined.add(layers.Activation('relu'))
    model_combined.add(layers.Dense(2))
    model_combined.add(layers.Activation('softmax'))

    #This is the combined model (convolutional and full connected parts). The inputs are the spatial image (conv.input) and
    #the parameters (parameters_input) and the output is the output of model_combined.
    final_model = models.Model([conv.input, parameters_input], model_combined(merged_output))

    learning_rate = .001
    #We use Adam for the optimization algorithm and binary crossentropy as the loss.
    final_model.compile(optimizer = keras.optimizers.Adam(learning_rate=learning_rate), 
    loss = 'binary_crossentropy',
    metrics = ['accuracy'])

    #Create a learning rate schedule for training neural network (nn). In this particular schedule, the learning rate
    #is constant for the first five epochs then decreases.
    def scheduler(epoch, learning_rate):
        if epoch < 5:
            return learning_rate
        else:
            return learning_rate * tensorflow.math.exp(-0.1)
  
    callback = tensorflow.keras.callbacks.LearningRateScheduler(scheduler)
  
    epochs = 20
    batch_size = 30000
    #Train model
    history = final_model.fit(x = [x_train_image, x_train_parameters],  y = y_train, epochs = epochs, 
                          validation_data = ([x_val_image, x_val_parameters], y_val),
                          batch_size = batch_size, verbose = 1, callbacks = [callback])
    
    #Save training and validation loss and accuracy
    validation_accuracy_file_name = (local_folder + "/" + image_name + "/" + version + "/accuracy/gp_" + image_name + "_" +
                    version + "_validation_accuracy.json")
    with open(validation_accuracy_file_name, 'w') as f:
      json.dump(history.history['val_accuracy'], f)

    training_accuracy_file_name = (local_folder + "/" + image_name + "/" + version + "/accuracy/gp_" + image_name + "_" +
                    version + "_training_accuracy.json")  
    with open(training_accuracy_file_name, 'w') as f:
      json.dump(history.history['accuracy'], f)

    validation_loss_file_name = (local_folder + "/" + image_name + "/" + version + "/loss/gp_" + image_name + "_" +
                    version + "_validation_loss.json")
    with open(validation_loss_file_name, 'w') as f:
      json.dump(history.history['val_loss'], f)

    training_loss_file_name = (local_folder + "/" + image_name + "/" + version + "/loss/gp_" + image_name + "_" +
                    version + "_training_loss.json")
    with open(training_loss_file_name, 'w') as f:
      json.dump(history.history['loss'], f)
    
    #Plot training and validation loss and accuracy
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    fig_name = (local_folder + "/" + image_name + "/" + version + "/loss/gp_" + image_name + "_" + version + "_loss_plot.png")
    plt.savefig(fig_name)
    plt.clf()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    fig_name = (local_folder + "/" + image_name + "/" + version + "/accuracy/gp_" + image_name + "_" + version + "_accuracy_plot.png")
    plt.savefig(fig_name)
    plt.clf()

    #Save model architecture and model weights.
    json_file = (local_folder + "/" + image_name + "/" + version + "/model/gp_" + image_name + "_" + version + "_nn.json")
    model_json = final_model.to_json()
    with open(json_file, "w") as json_file:
      json_file.write(model_json)

    weights_file = (local_folder + "/" + image_name + "/" + version + "/model/gp_" + image_name + "_" + version + "_nn_weights.h5")
    final_model.save_weights(weights_file)
