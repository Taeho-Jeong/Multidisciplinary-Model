import os
os.environ['PYTHONHASHSEED']=str(1)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.regularizers import L2
    
import tensorflow as tf
import random 

from tensorflow.keras import backend as K


class MyThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None): 
        val_loss = logs["loss"]
        if val_loss <= self.threshold:
            self.model.stop_training = True


# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def reset_random_seeds():
   #os.environ['PYTHONHASHSEED']=str(1)
   tf.random.set_seed(1)
   np.random.seed(1)
   random.seed(1)
    
# ----------------------------------------------------------------------------
#                           Normalization Methods
# ----------------------------------------------------------------------------

def std_scalar(data):
    """
        Method for creating std scalar object.
    """

        
    trans = StandardScaler()
    trans.fit(data)

    return trans

def transform(x, x_transform):
    """
        Method for transforming the input
        based on the std scalar object.
    """
    
    x = x_transform.transform(x)
    
    return x


# ----------------------------------------------------------------------------
#                        Training and prediction Methods
# ----------------------------------------------------------------------------

def train(x, y):
    """
        Method for training the NN for given hyperparameters.
    """
    reset_random_seeds()

    hidden_layer = 4 - 1
    shrinkage = 0.8785811324854128
    first_neuron = 24
    activation = ["tanh"]
    
    # Getting the standard scalar object
    x_transform = std_scalar(x)
    y_transform = std_scalar(y)

    # Normalize training and testing data to zero mean and unit variance
    x = transform(x, x_transform)
    y = transform(y, y_transform)

    # Get learning rate and set the optimizer
    lr = 0.001
    opt = Adam(learning_rate=lr)

    # Get the number of hidden layers and create activation function list
    activations = activation * hidden_layer

    # Get the number of neurons in each layer
    shrinkage_factor = shrinkage
    layers = [first_neuron]
    for idx in range(hidden_layer):
        if idx != 0:
            layers.append(round(layers[idx-1]*shrinkage_factor))

    # Predefined hyperparameters
    regularizer = None
    # regularizer = L2( l2=0.01 )
    epochs = 1000
    initializer = GlorotUniform(seed=10)

    tolerance = 0.0001  # Define your tolerance
    callbacks = MyThresholdCallback(threshold=tolerance)

    # Build the NN structure 
    model = Sequential()

    # Input layer - doesn't have any activation function
    model.add(Input(shape=(x.shape[1],)))

    # Hidden layers
    for i in range(len(layers)):
        model.add(Dense(layers[i], activation=activations[i], activity_regularizer=regularizer, kernel_initializer=initializer))
        
    # Output layer
    model.add(Dense(y.shape[1]))

    # Complile the model
    model.compile(optimizer=opt, loss='mean_squared_error')

    # Train the model
    # history = model.fit(x, y, epochs=epochs, verbose=0, callbacks=[callbacks], use_multiprocessing=True)
    history = model.fit(x, y, epochs=epochs, verbose=0, callbacks=[callbacks])
        
    return model, x_transform, y_transform

def predict(x, x_transform, y_transform, model):
    reset_random_seeds()

    # Reshaping x
    dim = x.ndim
    if dim == 1:
        x = x.reshape(1,-1)

    # Scaling, Prediction, and Rescaling
    x = transform(x, x_transform)
    y = model(x, training=False)
    y = y_transform.inverse_transform(y)

    # Reshaping the y
    # Req for scipy minimize
    if dim == 1:
        y = y.reshape(-1,)

    return y










