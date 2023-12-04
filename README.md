# Binary Image Classification Project
This repository contains a project for binary image classification, specifically designed for classifying images of cars and bikes. The project consists of several components, including data preparation, model training, prediction, and result analysis. Below, you'll find an overview of each part.

## Requirements
Before you start, make sure to have the necessary libraries installed. You can easily install the required dependencies using the following command:
###### pip install tensorflow plotly pandas scikit-learn

## Data Preparation
### The first part of the project is dedicated to data preparation. It involves the following steps:

Defining Classes: In the code, two classes are defined, namely "Car" and "Bike." These classes correspond to the types of images the model is trained to classify.

Training and Validation Ratios: The TRAIN_RATIO and VALID_RATIO variables allow you to customize the ratio in which the dataset is divided into training, validation, and test sets. These ratios determine how many images will be allocated to each set.

## Model Architecture
The project employs the LeNet-5 architecture for image classification. The model consists of the following layers:

#### Convolutional Layers: Two convolutional layers with ReLU activation functions and different numbers of filters. These layers extract features from the input images.

#### Max-Pooling Layers: After each convolutional layer, max-pooling layers downsample the feature maps to reduce computational complexity.

#### Flatten Layer: A flatten layer reshapes the 2D feature maps into a 1D vector, which can be fed into fully connected layers.

#### Fully Connected Layers: Two fully connected layers with ReLU activation functions are used for further feature extraction.

#### Output Layer: The output layer has one unit with a sigmoid activation function for binary classification.

You can customize the architecture by changing parameters such as the number of filters, filter sizes, and units in the fully connected layers.

## Model Training
The second part of the project involves training the model. The model is trained using the specified configuration, and the best model is saved.

You can adjust training parameter - the number of epochs.

-e or --epochs: Set to 1 by default. Specifies the number of epochs used to train the model.

Example: python 02_train.py -e 20


After training, an HTML report is generated, which includes accuracy and loss plots. The model is ready to be used for predictions after this training phase.

## Prediction
#### The prediction script allows you to use the trained model to make predictions on a specific dataset. You can provide two arguments to the script:

-d or --dataset: Specifies the type of dataset to predict from (either "train," "valid," or "test").
-m or --model: Optionally, you can provide the path to a pre-trained model to use for predictions.

Example: python 03_predict.py -d test -m output/model_01_11_2023_10_24.hdf5

#### The prediction script performs the following steps:

It loads the specified model.

It makes predictions on the dataset and calculates predicted probabilities and labels.

It creates a CSV file with the prediction results, including predicted probabilities, true labels, predicted labels, and a binary indicator for incorrect predictions.

It prints a confusion matrix, a classification report, and the model accuracy to assess the model's performance.