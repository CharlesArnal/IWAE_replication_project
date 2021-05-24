# Training a variational autoencoder using the IWAE loss on the fixed binarization MNIST dataset

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import datetime
import math
import pickle
import scipy.io as sio

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

from flexible_IWAE import Flexible_Model

# Download fixed binarization MNIST dataset

x_train, x_test = tfds.load("binarized_mnist", 
                            split=['train', 'test'], 
                            shuffle_files=True,
                            batch_size=-1)  

x_train = tf.convert_to_tensor(x_train["image"].numpy().astype("float32"))
x_test = tf.convert_to_tensor(x_test["image"].numpy().astype("float32"))


# Training parameters
batch_size = 100
optimizer = tf.keras.optimizers.Adam(
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-04,
)


# Creating model


# n_hidden_de/encoder should be a list of length n_latent_encoder (see Encoder or Decoder code)
# 2L
n_hidden_encoder = [200, 100]
n_hidden_decoder = [100, 200]
n_latent_encoder = [100, 50]
n_latent_decoder = [100, 28*28]

# Choose the loss function
loss_function = "IWAE"
k = 50
alpha = 1
p = 1
beta = 0.05

mdl = Flexible_Model(n_hidden_encoder,n_hidden_decoder, n_latent_encoder, 
    n_latent_decoder, dataset_name, loss_function, k, alpha, p, beta)

mdl.compile(optimizer=optimizer)


# Tensorboard set-up 
log_dir = f"tensorboard/test/{mdl.loss_function}-{len(n_hidden_encoder)}L-k_{mdl.k}_starttime" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
print(f"Tensorboard logging directory: {log_dir}")
file_writer = tf.summary.create_file_writer(log_dir)
file_writer.set_as_default()

res2s = []
res1s = []
total_passes = 0 # 1 epoch includes 3**(i-1) passes over the data
for i in range(1, 9): 
    optimizer.learning_rate = 1e-4*round(10.**(1-(i-1)/7.), 1)
    spasses = 3**(i-1)


# Train for 
tf.print(f"Training for {passes} passes over the training data with learning rate = {optimizer.learning_rate.numpy()}")
mdl.fit(x_train, epochs=passes, batch_size=batch_size)
total_passes += passes 

# Get the test statistics

res1, res2 = mdl.get_training_statistics(x_test, k)
print(res1)
print(res2)
res2s.append(res2)
res1s.append(res1)
mdl.tensorboard_log(res1, total_passes)

# Save the model at the end of each epoch (on Colab)
mdl.save_weights(f"{mdl.loss_function}-{len(n_hidden_encoder)}L-k_{mdl.k}-epoch_{i}") # this checkpoint save at the end of each epoch, just in case it stops
with open(f'{mdl.loss_function}-{len(n_hidden_encoder)}L-k_{mdl.k}.res2', 'wb') as file:
  pickle.dump((res1s, res2s), file)
