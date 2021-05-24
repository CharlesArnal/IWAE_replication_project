# Importance Weighted Autoencoders
Replication and discussion of experiments from the papers "Importance Weighted Autoencoders" ( https://arxiv.org/abs/1509.00519 ) and "Tighter Variational Bounds are Not Necessarily Better" ( https://arxiv.org/abs/1802.04537 ) on importance weighted variational autoencoders (IWAEs), with a few additional original experiments.
Full description of the project is available in the pdf IWAE_replication.pdf.
This was a joint work with Jamie Lee and Nicholas Pezzotti.

# Setup
The code was run using Tensorflow version 2.4.1 and Numpy version 1.19.5

# Datasets
Our experiments are run on the MNIST, fixed binarization MNIST, Fashion MNIST and Omniglot datasets.

# Running experiments
An example of an experiment (training an autoencoder with the IWAE loss function on the fixed binarization dataset) can be found in experiment_example.py .
We ran our experiments on Google's Colab; a few commands are specific to this environment.

