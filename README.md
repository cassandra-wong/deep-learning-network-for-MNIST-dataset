# Deep Learning Network for MNIST dataset

The purpose of this project is to implement a fully-connected deep learning network with 2 hidden layers and a Softmax output layer, and train it to classify two image data sets with 10 classes. 

The two data sets used for this project are MNIST handwritten digits data set, and MNIST fashion images data set. They can both be downloaded [here](http://yann.lecun.com/exdb/mnist/).

Functions that are needed for the neural network is created using only primitive Matlab commands as `phi_ReLU.m`, `phi_Softmax.m`, `jac_ReLU.m`, and `jac_Softmax.m`. The deep learning network with training and testing data is developed as `deepNetwork.m`.

Comparing the digit images test accuracy (95.75%) with fashion images test accuracy (85.45%), using data with maxit=300,000 and training n=60,000, we can see that the digit images have achieved a higher accuracy than fashion images. See `deep-learning-result.pdf` for output details. 
