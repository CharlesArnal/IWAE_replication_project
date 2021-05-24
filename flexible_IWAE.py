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


# Stochastic layer composed of two deterministic layers and a gaussian distribution
# Rmk: could generalize a bit with variable number of layers
class Stochastic_layer(tf.keras.Model):
    def __init__(self, n_hidden, n_latent, **kwargs):
        super(Stochastic_layer, self).__init__(**kwargs)

        self.l1 = tf.keras.layers.Dense(n_hidden, activation=tf.nn.tanh)
        self.l2 = tf.keras.layers.Dense(n_hidden, activation=tf.nn.tanh)
        self.lmu = tf.keras.layers.Dense(n_latent, activation=None)
        self.lstd = tf.keras.layers.Dense(n_latent, activation=tf.exp)

    # outputs a distribution (from which we can then sample or compute likelihoods)
    def call(self, input):
        y1 = self.l1(input)
        y2 = self.l2(y1)
        q_mu = self.lmu(y2)
        q_std = self.lstd(y2)
        qh_given_input = tfd.Normal(q_mu, q_std + 1e-6)
        return qh_given_input


# contains n_stochastic layers. n_hidden and n_latent are vectors of integers of
# length n_stochastics, each coordinate specifying the way to build the corresponding 
class Encoder(tf.keras.Model):
    def __init__(self, n_hidden, n_latent, n_stochastic, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.stochastic_layers=[]
        self.n_stochastic = n_stochastic
        for i in range(n_stochastic):
          self.stochastic_layers.append(Stochastic_layer(n_hidden[i], n_latent[i]))
        
    # Returns a list h = [h1,h2,...,hn_stochastics]),
    # where each hi is of the shape [n_samples,batch_size,n_latent[i-1]]
    # Returns the probability q(h|x) as well, which will be a n_samples x batch_size tensor
    # Doesn't return any distribution at the moment
    # Data x needs to be preemptively flattened   
    def call(self, x, n_samples):
      assert(x.shape[1] == 28*28)
      qh1Ix = self.stochastic_layers[0](x)
      h1 = qh1Ix.sample(n_samples)
      log_qh1Ix = tf.reduce_sum(qh1Ix.log_prob(h1),axis = -1)
      h = [h1]
      distributions=[qh1Ix]
      log_qhip1Ihi = [log_qh1Ix]
      for i in range(1,self.n_stochastic):
        # the distribution q(hi+1|hi), with h0 := x
        qhip1Ihi = self.stochastic_layers[i](h[-1])
        # the corresponding samples
        h.append(qhip1Ihi.sample())
        # the scalar log(q(hi+1|hi))
        log_qhip1Ihi.append(tf.reduce_sum(qhip1Ihi.log_prob(h[-1]), axis=-1))
        distributions.append(qhip1Ihi)
      
      log_qhIx = tf.reduce_sum(log_qhip1Ihi,axis = 0)
      
      return h, log_qhIx, distributions[-1]

class Decoder(tf.keras.Model):
    def __init__(self, n_hidden, n_latent, n_stochastic, dataset_bias="Binarized_MNIST",**kwargs):
        super(Decoder, self).__init__(**kwargs)

        self.stochastic_layers=[]
        self.n_stochastic = n_stochastic
        # the first layer goes from hL to hL-1, second layer from hL-1 to hL-2,
        # ..., last layer from h1 to x  -> stochastic_layers[i] goes from 
        # hL-i to hL-1-i (with h0=x)
        for i in range(n_stochastic-1):
          self.stochastic_layers.append(Stochastic_layer(n_hidden[i], n_latent[i]))

        self.stochastic_layers.append(
            tf.keras.Sequential(
              [
                  tf.keras.layers.Dense(n_hidden[-1], activation=tf.nn.tanh),
                  tf.keras.layers.Dense(n_hidden[-1], activation=tf.nn.tanh),
                  tf.keras.layers.Dense(28*28, activation="sigmoid", bias_initializer=self.get_bias(dataset_bias))
              ]
        ) )
  
    # Outputs the probabilities of being black for each pixel given h=[h1,...,h_n_stochastic]
    # (i.e., in fact, given h1), as well as the corresponding Bernoulli distribution p(x|h)
    def call(self, h):
        probs = self.stochastic_layers[-1](h[0])
        probs = probs*(1-10**(-6)) + 10**(-7)
        pxIh = tfd.Bernoulli(probs=probs)

        return probs, pxIh

    def generate_x(self,h_n_stochastic):
      # h_L of the shape [n_samples,batch_size,D]
      # (in most usages, n_samples should be 1)
      reversed_h = [h_n_stochastic]
      # unlike h in the rest of this code, reversed_h = [h_L,h_L-1,...,h_1]
      for i in range(self.n_stochastic-1):
        phLmim1IhLmi = self.stochastic_layers[i](reversed_h[-1])
        reversed_h.append(phLmim1IhLmi.sample())
      # this turns reversed_h into h
      reversed_h.reverse()
      probs_x_reconstructed, _ = self.call(reversed_h)
      return probs_x_reconstructed


    # log p(x|h): outputs log likelihood of x knowing h=[h1,...,h_n_stochastic]
    # (simply a Bernoulli)
    def get_log_pxIh(self, x, h):
      # probs is a deterministic function of h1 = h[0]
      probs = self.stochastic_layers[-1](h[0])
      probs = probs*(1-10**(-6)) + 10**(-7)
      pxIh = tfd.Bernoulli(probs=probs)
      log_pxIh = tf.reduce_sum(pxIh.log_prob(x), axis=-1)
      return log_pxIh

    # log p(h): outputs log likelihood of h=[h1,h2,...,h_n_stochastic]
    # which is equal to log p(h_n_stochastic) + p(h_n_stochastic-1|h_n_stochastic)...
    # + log p(h1|h2) 
    def get_log_ph(self, h):
      ph_n_stochastic = tfd.Normal(0, 1)
      log_phn_stochastic = tf.reduce_sum(ph_n_stochastic.log_prob(h[-1]), axis=-1)
      log_phiIhip1 = [log_phn_stochastic]
      for i in range(self.n_stochastic-1):
        phiIhip1 = self.stochastic_layers[i](h[self.n_stochastic-1-i])
        log_phiIhip1.append(tf.reduce_sum(phiIhip1.log_prob(h[self.n_stochastic-2-i]), axis=-1))
      get_log_ph = tf.reduce_sum(log_phiIhip1,0)
      return get_log_ph 

    def get_log_pxh(self, x, h):
        return self.get_log_pxIh(x, h) + self.get_log_ph(h)

    def get_bias(self, dataset="Binarized_MNIST"):
      if "binarized_mnist" in dataset.lower():
        tf.print("Setting the bias to the Fixed Binarisation MNIST")
        Xtrain, _ = tfds.load("mnist", 
                                split=['train', 'test'], 
                                shuffle_files=True,
                                batch_size=-1)

        Xtrain = Xtrain["image"].numpy()

      elif "mnist" in dataset.lower():
        tf.print("Setting the bias to the Stochastic Binarisation MNIST")
        # For initializing the bias in the final Bernoulli layer for p(x|z)
        (Xtrain, _), (_, _) = keras.datasets.mnist.load_data()

      elif "omniglot" in dataset.lower():
        tf.print("Setting the bias to the OMNIGLOT")
        d = sio.loadmat("chardata.mat")
        Xtrain = d["data"].transpose().reshape((-1, 28*28))
      else:
        raise Exception("Trying to set the initialisation for the bias, \
                        but the dataset is not recognized")

      Ntrain = Xtrain.shape[0]
      # reshape to vectors
      Xtrain = Xtrain.reshape(Ntrain, -1) / 255
      train_mean = np.mean(Xtrain, axis=0)
      bias = -np.log(1. / np.clip(train_mean, 0.001, 0.999) - 1.)
      return tf.constant_initializer(bias)

class Flexible_Model(keras.Model):
    def __init__(self, n_hidden_encoder, n_hidden_decoder, n_latent_encoder, 
                 n_latent_decoder, dataset_bias="Binarized_MNIST", 
                 loss_function = "VAE", k = 50, p = 1, alpha = 1, beta=0.5, **kwargs):
        """
        Params
        ------
        n_hidden_encoder: list indicating the dimensions of the hidden layers of 
                          each stochastic layer in the encoder. E.g. [100, 200],
                          creates two stochastic layers in the encoder with
                          hidden layers of shape 100 and 200, respectively
        n_hidden_decoder: analogous to n_hidden_encoder, but for the decoder.
                          Note: the layers need to be fed in the order in which
                          the input must be fed. That is, to maintain symmetry,
                          it should be the inverse of n_hidden_encoder
        n_latent_encoder: list indicating the dimensions of the outputs of 
                          each stochastic layer in the encoder. E.g. [100, 50],
                          creates two stochastic layers in the encoder with
                          outputs of shape 100 and 50, respectively
        n_latent_decoder: analogous to n_latent_encoder, but for the decoder.
                          Note: the layers need to be fed in the order in which
                          the input must be fed. That is, to maintain symmetry,
                          it should be the inverse of n_latent_encoder. The last
                          layer needs to be 28*28 (i.e. the size of the input/
                          output)
        """

        super(Flexible_Model, self).__init__(**kwargs)
        
        n_stochastic_encoder = len(n_hidden_encoder)
        n_stochastic_decoder = len(n_hidden_decoder)
        self.encoder = Encoder(n_hidden_encoder, n_latent_encoder, n_stochastic_encoder)
        self.decoder = Decoder(n_hidden_decoder, n_latent_decoder, n_stochastic_decoder,dataset_bias)
        self.n_stochastic_encoder = n_stochastic_encoder
        self.n_stochastic_decoder = n_stochastic_decoder
        self.dataset_bias = dataset_bias
        self.loss_function = loss_function
        self.k = k
        self.p = p
        self.alpha = alpha
        self.beta= beta
        self.epoch = 0

    #@tf.function
    def train_step(self, x):
      # We expect x to be of the shape [batch_size,28,28,1] 
        k = self.k
        p = self.p
        alpha = self.alpha
        beta= self.beta
        with tf.GradientTape() as tape:
          if self.loss_function == "VAE":
             loss = - self.get_L(x,k)
          elif self.loss_function == "IWAE":
             loss = - self.get_L_k(x,k)
          elif self.loss_function == "VAE_V1":
            loss = - self.get_L_V1(x,k)
          elif self.loss_function == "L_alpha":
            loss = - self.get_L_alpha(x,k,alpha)
          elif self.loss_function == "L_power_p":
            loss = - self.get_L_power_p(x,k,p)
          elif self.loss_function == "L_median":
            loss = - self.get_L_median(x,k)
          elif self.loss_function == "CIWAE":
            loss = -self.get_L_CIWAE(x,k,beta)
        res = {self.loss_function: loss}
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.epoch += 1

        return res

    def reconstructed_x_probs(self,x):
      x = keras.layers.Flatten()(x)
      h, _ , _ = self.encoder(x,1)
      reconstructed_x = self.decoder.generate_x(h[-1])

      return reconstructed_x

    def get_reconstruction_loss(self,x):
      probs = self.reconstructed_x_probs(x)
      y = tf.expand_dims(keras.layers.Flatten()(x), axis=0)
      y = tf.expand_dims(y, axis=-1)
      probs = tf.expand_dims(probs, axis=-1)
      reconstruction_loss = tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(y, probs),axis=-1))
      return reconstruction_loss

    def get_levels_of_units_activity(self,x,n_samples):
      # of the shape [batch_size,784]  
      x = keras.layers.Flatten()(x)
      variances = []
      eigen_values=[]
      h_means, _ , _ = self.encoder(x,1)
      for i in range(n_samples - 1):
        h , _ , _ = self.encoder(x,1)
        for index, hi in enumerate(h):
          h_means[index] += hi
      for hi_mean in h_means:
        # take the mean over q(h|x)
        # should now be of dimension [batch_size,n_latent_encoder[i-1]] 
        hi_mean = tf.squeeze(hi_mean/n_samples, axis= 0)
        # should now be of dimension [n_latent_encoder[i-1]]
        variances.append(tf.math.reduce_variance(hi_mean, axis = 0))
        eigen_values.append(self.get_eigenvalues_PCA(hi_mean))
      return variances, eigen_values
     
    # data meant to be of dimension [batch_size,D]
    def get_eigenvalues_PCA(self,data):
        # normalize
        normalized_data = data - tf.reduce_mean(data, axis=0)
        # Empirical covariance matrix
        cov = tf.tensordot(tf.transpose(normalized_data), normalized_data, axes=1) / normalized_data.get_shape()[0]
        # Find eigen_values
        eigen_values, _ = tf.linalg.eigh(cov)
        return eigen_values


    def get_active_units(self,variances,eigen_values, threshold=0.01):
      active_units = []
      number_active_units = []
      number_active_units_PCA = []
      for index, variance in enumerate(variances):
        active_units.append([1 if coordinate_var>threshold else 0 for coordinate_var in variance])
        number_active_units.append(tf.reduce_sum(active_units[-1]))
        number_active_units_PCA.append(tf.reduce_sum([1 if eig>threshold else 0 for eig in eigen_values[index]]))
      return active_units, number_active_units, number_active_units_PCA

    def get_E_qhIx_log_pxIh(self,x,n_samples):

      x = keras.layers.Flatten()(x)
      # encoding
      # h = [h1,...,h_n_stochastic_encoder]
      # h_i of the shape [n_samples,batch_size,n_latent_encoder[i-1]]
      # log_qhIx of the shape [n_samples, batch_size]
      h,log_qhIx, qhLIx = self.encoder(x,n_samples)
      # decoding
      # probs of the shape [n_samples, batch_size,784]
      # pxIh a distribution
      probs, pxIh = self.decoder(h)
      
      y = tf.expand_dims(x, axis=0)
      y = tf.repeat(y, n_samples, axis=0)
      # y of dim [n_samples,batch_size,784]
      # E_q(h|x)[log(p(x|h))]
      y = tf.expand_dims(y, axis=-1)
      probs = tf.expand_dims(probs, axis=-1)
      E_qhIx_log_pxIh = -1*tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(y, probs),axis=-1))

      return E_qhIx_log_pxIh

    def get_log_weights(self,x,n_samples):
      # of the shape [batch_size,784]  
      x = keras.layers.Flatten()(x)

      # encoding
      # h = [h1,...,h_n_stochastic_encoder]
      # h_i of the shape [n_samples,batch_size,n_latent_encoder[i-1]]
      # log_qhIx of the shape [n_samples, batch_size]
      h,log_qhIx, _ = self.encoder(x,n_samples)

      # decoding
      # probs of the shape [n_samples, batch_size,784]
      # pxIh a distribution
      probs, pxIh = self.decoder(h)
      
      # all of the shape [n_samples, batch_size]
      log_pxIh = self.decoder.get_log_pxIh(x,h)
      log_ph = self.decoder.get_log_ph(h)
      log_pxh = log_ph + log_pxIh

      # Computing the weights wi
      # of the shape [n_samples,batch_size] (due to the sum along the batch_size axis)
      log_weights=log_pxh-log_qhIx 

      return log_weights
      
      # L_k = - total loss IWAE
    def get_L_k(self, x, k):
      # log_weights of the shape [n_samples,batch_size]
      log_weights = self.get_log_weights(x,k)
      # L_k(x) computed as in the article (k= n_samples), i.e. MC approximation
      # of E_q(h|x)[log(mean(w1,...,wk))]
      L_k = self.L_k_from_weights(log_weights)

      return L_k

    def L_k_from_weights(self,log_weights):
      # log_weights of the shape [n_samples,batch_size]
      # L_k(x) computed as in the article (k= n_samples), i.e. MC approximation
      # of E_q(h|x)[log(mean(w1,...,wk))]
      # max is of shape [batch_size]
      max = tf.reduce_max(log_weights, axis = 0)  
      L_k = tf.reduce_mean(tf.math.log(tf.reduce_mean(tf.exp(log_weights - max), axis=0)) + max)
      return L_k

    # computes L_median := E_qhIx[log(median(w1,...,wk))] =~ E_qhIx[median(log(w1,...,wk))]
    def get_L_median(self,x,k):
      # log_weights of the shape [n_samples,batch_size]
      log_weights = self.get_log_weights(x,k)
      # of the shape [batch_size]
      median_log_weights = tfp.stats.percentile(log_weights, 50.0, interpolation='midpoint',axis=0)
      L_power_p = tf.reduce_mean(median_log_weights)
      return L_power_p

    # Computes L_CIWAE = beta * LVAE + (1-beta) * LIWAE
    def get_L_CIWAE(self,x,n_samples,beta):
      return beta*self.get_L(x, n_samples) + (1-beta)*self.get_L_k(x,n_samples) 

    # computes L_alpha := E_qhIx_log_pxIh - alpha*Dkl_qhIx_ph
    def get_L_alpha(self,x,n_samples,alpha):
      x = keras.layers.Flatten()(x)
      h,log_qhIx, _ = self.encoder(x,n_samples)
      probs, pxIh = self.decoder(h)
      log_pxIh = self.decoder.get_log_pxIh(x,h)
      log_ph = self.decoder.get_log_ph(h)
      log_pxh = log_ph + log_pxIh
      log_weights=log_pxh-log_qhIx 
      L = self.L_from_weights(log_weights) 
      #---
      y = tf.expand_dims(x, axis=0)
      y = tf.repeat(y, n_samples, axis=0)
      y = tf.expand_dims(y, axis=-1)
      probs = tf.expand_dims(probs, axis=-1)
      E_qhIx_log_pxIh = -1*tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(y, probs),axis=-1))
      L_alpha=(1-alpha)*E_qhIx_log_pxIh + alpha*L
      return L_alpha

    # computes L_power_p = E_q(h|x)[log(mean(w1^p,...,wk^p)^1/p)]
    def get_L_power_p(self,x,k,p):
      log_weights = self.get_log_weights(x,k)
      max = tf.reduce_max(log_weights, axis = 0)  
      L_power_p = tf.reduce_mean(tf.math.log(tf.reduce_mean(tf.exp((log_weights-max)*p), axis=0))/p + max)
      return L_power_p

    def get_Dkl_qhIx_phIx(self,x,k):
      return  -1*(self.get_L(x,k) + self.get_NLL(x))

    def get_Dkl_qhIx_ph(self,x,k):
      return  self.get_E_qhIx_log_pxIh(x,k) -self.get_L(x,k)

    # k measures the degree of precision of the MC approximation
    # L = - total loss VAE
    def get_L(self,x,k=5000):

      # log_weights of the shape [n_samples,batch_size]
      log_weights = self.get_log_weights(x,k)
      # -L(x), where L(x) is computed as in the article,
      # i.e. MC approximation of E_q(h|x)[log(w)]
      # Should be equal to total_loss_VAE_1 up to MC approximation error
      L = self.L_from_weights(log_weights) 
      return L

    def L_from_weights(self,log_weights):
      return tf.reduce_mean(log_weights) 

    # alternative way of computing L (for comparison)
    # only works when there is a single stochastic layer
    def get_L_V1(self,x,n_samples):
      x = keras.layers.Flatten()(x)
      # encoding
      # h = [h1,...,h_n_stochastic_encoder]
      # h_i of the shape [n_samples,batch_size,n_latent_encoder[i-1]]
      # log_qhIx of the shape [n_samples, batch_size]
      
      # distribution qhLIx is only used to compute total_loss_VAE_1 in the case
      # where there is only one stochastic layer
      h,log_qhIx, qhLIx = self.encoder(x,n_samples)

      # decoding
      # probs of the shape [n_samples, batch_size,784]
      # pxIh a distribution
      probs, pxIh = self.decoder(h)
      
      y = tf.expand_dims(x, axis=0)
      y = tf.repeat(y, n_samples, axis=0)
      # y of dim [n_samples,batch_size,784]
      # E_q(h|x)[log(p(x|h))]
      y = tf.expand_dims(y, axis=-1)
      probs = tf.expand_dims(probs, axis=-1)
      E_qhIx_log_pxIh = -1*tf.reduce_mean(tf.reduce_sum(keras.losses.binary_crossentropy(y, probs),axis=-1))
      Dkl_qhIx_ph = -0.5 * (1 + 2*tf.math.log(qhLIx.scale) - tf.math.square(qhLIx.loc)  - tf.math.square(qhLIx.scale))
      Dkl_qhIx_ph = tf.reduce_mean(tf.reduce_sum(Dkl_qhIx_ph, axis=-1))
      L = E_qhIx_log_pxIh-Dkl_qhIx_ph
      return L

    # MC estimation of log(p(x)) (corresponds to computing L_k(x) for very large k)
    def get_NLL(self,x,k=5000):
      return - self.get_L_k(x,k)

    def get_NLL_without_inactive_units(self,x,threshold=0.01,n_samples = 5000):
      variances, eigen_values = self.get_levels_of_units_activity(x,n_samples)
      active_units, _, _ = self.get_active_units(variances,eigen_values,threshold)
      x = keras.layers.Flatten()(x)
      y = x
      assert(y.shape[1] == 28*28)
      qh1Iy = self.encoder.stochastic_layers[0](y)
      h1 = qh1Iy.sample(n_samples)
      modified_h1 = h1*active_units[0]
      log_qh1Iy =  tf.reduce_sum(qh1Iy.log_prob(modified_h1),axis = -1)
      h=[modified_h1]
      distributions = [qh1Iy]
      log_qhip1Ihi = [log_qh1Iy]
      for i in range(1,self.encoder.n_stochastic):
        qhip1Ihi = self.encoder.stochastic_layers[i](h[-1])
        hip1 = qhip1Ihi.sample()
        modified_hip1 = hip1*active_units[i]
        h.append(modified_hip1)
        log_qhip1Ihi.append(tf.reduce_sum(qhip1Ihi.log_prob(h[-1]), axis=-1))
        distributions.append(qhip1Ihi)
      log_qhIx = tf.reduce_sum(log_qhip1Ihi,0)

      probs, pxIh = self.decoder(h)
      log_pxIh = self.decoder.get_log_pxIh(x,h)
      log_ph = self.decoder.get_log_ph(h)
      log_pxh = log_ph + log_pxIh
      log_weights=log_pxh-log_qhIx 

      return - self.L_k_from_weights(log_weights)

    def get_training_statistics(self, x, k, batch_size=10):
      res = {}
      res2 = {}

      batched_x_test = tf.reshape(x, (-1, batch_size, 28, 28, 1))
      n_batches = batched_x_test.shape[0]

      res["VAE"] = 0
      res["IWAE"] = 0
      res["NLL"] = 0
      res["E_q(h|x)[log(p(x|h))]"] = 0
      res["D_kl(q(h|x),p(h))"] = 0
      res["D_kl(q(h|x),p(h|x))"] = 0
      res["reconstruction_loss"] = 0

      tf.print(f"Evaluating the test statistics")
      for batch in batched_x_test:
        res["VAE"] += keras.backend.get_value(self.get_L(batch, k))/n_batches
        res["IWAE"] += keras.backend.get_value(self.get_L_k(batch, k))/n_batches
        res["NLL"] += keras.backend.get_value(self.get_NLL(batch))/n_batches
        res["E_q(h|x)[log(p(x|h))]"] += keras.backend.get_value(self.get_E_qhIx_log_pxIh(batch, k))/n_batches
        res["D_kl(q(h|x),p(h))"] += keras.backend.get_value(self.get_Dkl_qhIx_ph(batch, k))/n_batches
        res["D_kl(q(h|x),p(h|x))"] += keras.backend.get_value(self.get_Dkl_qhIx_phIx(batch, k))/n_batches
        res["reconstruction_loss"] += keras.backend.get_value(self.get_reconstruction_loss(batch))/n_batches
      
      variances, eigen_values = self.get_levels_of_units_activity(x, 1000)
      res2["active_units"], res2["number_of_active_units"], res2["number_of_PCA_active_units"] = self.get_active_units(variances, eigen_values)
      res2["variances"] = variances
      res["LL_pruned"] = self.get_NLL_without_inactive_units(batched_x_test[0])

      return res, res2


    def tensorboard_log(self, res, epoch_n=-1):
      """
          res: a dictionary of the results generated from self.get_training_statistics
          epoch_n: gives the option to override the actual epoch number
      """

      if epoch_n == -1:
        epoch_n = self.epoch

      # tensorboard logging (special chars not allowed)
      tf.summary.scalar("VAE", res["VAE"], step=epoch_n)
      tf.summary.scalar("IWAE", res["IWAE"], step=epoch_n)
      tf.summary.scalar("NLL", res["NLL"], step=epoch_n)
      tf.summary.scalar("E_q(h|x)[log(p(x|h))]", res["E_q(h|x)[log(p(x|h))]"], step=epoch_n)
      tf.summary.scalar("D_kl(q(h|x),p(h))", res["D_kl(q(h|x),p(h))"], step=epoch_n)
      tf.summary.scalar("D_kl(q(h|x),p(h|x))", res["D_kl(q(h|x),p(h|x))"], step=epoch_n)
      tf.summary.scalar("reconstruction_loss", res["reconstruction_loss"], step=epoch_n)