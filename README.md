# Importance Weighted Autoencoders
Replication and discussion of experiments from the papers "Importance Weighted Autoencoders" ( https://arxiv.org/abs/1509.00519 ) and "Tighter Variational Bounds are Not Necessarily Better" ( https://arxiv.org/abs/1802.04537 ) on importance weighted variational autoencoders (IWAEs), with a few additional original experiments.
Full description of the project is available in the pdf IWAE_replication.pdf.
This was a joint work with Jamie Lee and Nicholas Pezzotti in the context of the University of Cambridge's MPhil in Machine Learning and Machine Intelligence .

# Setup
The code was run using Tensorflow version 2.4.1 and Numpy version 1.19.5

# Datasets
Our experiments are run on the MNIST, fixed binarization MNIST, Fashion MNIST and Omniglot datasets.

# Running experiments
An example of an experiment (training an autoencoder with the IWAE loss function on the fixed binarization dataset) can be found in experiment_example.py .
We ran our experiments on Google's Colab; a few commands are specific to this environment.




<img src="https://user-images.githubusercontent.com/71833961/119830096-d4b69d00-bef3-11eb-9c84-baf21ba39c8f.png" width="400" height="400"> <img src="https://user-images.githubusercontent.com/71833961/119830081-cf595280-bef3-11eb-8679-46268becf1db.png" width="375" height="375"> 


<img src="https://user-images.githubusercontent.com/71833961/119829628-5823be80-bef3-11eb-9b14-29145e136b7c.png" width="800" height="800"> 



