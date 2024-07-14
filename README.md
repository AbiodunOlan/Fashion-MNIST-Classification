# Fashion MNIST Image Classification with CNN in Python

This project demonstrates how to classify images from the Fashion MNIST dataset using a Convolutional Neural Network (CNN) implemented in Python with TensorFlow and Keras.

## Prerequisites

Before running the code, ensure you have the following dependencies installed:

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Matplotlib

pip install tensorflow keras 


Dataset
The Fashion MNIST dataset consists of 70,000 grayscale images in 10 categories. The images show individual articles of clothing at low resolution (28x28 pixels).

Code Overview
Loading and Preprocessing the Data:

The dataset is loaded and reshaped to include the channel dimension.
Images are normalized to values between 0 and 1.
Building the CNN Model:

A Convolutional Neural Network (CNN) is defined with six layers:
Two convolutional layers followed by max-pooling layers.
A flatten layer to convert the 2D matrix to a vector.
A dense layer with 128 units and ReLU activation.
A dropout layer to prevent overfitting.
A dense output layer with 10 units and softmax activation.
Training the Model:

The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss.
The model is trained for 10 epochs with a validation split.
Evaluating the Model:

The model is evaluated on the test dataset, and accuracy is printed.
Making Predictions:

Predictions are made on the test dataset.
The results for two images from the test dataset are displayed along with the model's predictions and the actual labels.
Running the Code
Clone the repository or download the source code files.

Ensure you have the required dependencies installed.

Run the fashion_mnist_cnn.py script to train the model and make predictions:


# Fashion MNIST Image Classification with CNN in R

This project demonstrates how to classify images from the Fashion MNIST dataset using a Convolutional Neural Network (CNN) implemented in R with TensorFlow and Keras.

## Prerequisites

Before running the code, ensure you have the following dependencies installed:

- R
- TensorFlow
- Keras
- Reticulate

You can install these dependencies using the following R commands:

```r
install.packages("tensorflow")
install.packages("keras")
install.packages("reticulate")
