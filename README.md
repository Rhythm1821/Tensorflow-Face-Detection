# Face Detection Project

## Introduction

Welcome to the Face Detection Project! This project aims to implement a robust system detects the face using computer vision techniques and deep learning. It encompasses various stages, from data collection and preprocessing to model building and deployment. This README provides an overview of the project and its key components.

## Table of Contents

* Project Overview
* Data Collection and Annotation
* Data Preprocessing
* Data augmentation
* Model Building
* Custom Loss Function
* Custom Model Class
* Model Training and Evaluation
* Predictions and Model Deployment
* Real-Time Face Detection
* Cloning and Running

## Project Overview

The Face Detection Project showcases the process of implementing a face detection system using computer vision and deep learning techniques. It highlights the journey from collecting data and annotating it to deploying a real-time face detection system.

## Data Collection and Annotation

The project starts with the collection of image data using OpenCV. These images are then annotated using LabelMe, enabling the creation of labeled data with bounding boxes around the faces.

## Data Preprocessing

The annotated data is organized and stored in separate folders for images and labels. Google Colab is utilized to access the data, and TensorFlow's data pipeline is employed to load and preprocess the images. 

## Data augmentation
Data augmentation is applied using Albumentations, enhancing the model's robustness. The images are scaled to a range of 0-1 for neural network training.

## Model Building

The heart of the project lies in building a deep learning model. A VGG16 model forms the base, with additional Dense and Convolutional layers added for task-specific performance. The model handles both classification and regression tasks concurrently.

## Custom Loss Function

To optimize the model for the regression task of bounding box coordinates, a custom localization loss function is implemented. This loss computes differences in coordinates and sizes, contributing to more accurate regression predictions.

## Custom Model Class

The model is encapsulated within a custom class named FaceTracker, extending TensorFlow's Model class. This class facilitates training and testing steps, compiling the model with loss functions and optimizer, and implementing custom gradient updates during training.

## Model Training and Evaluation

The model is trained for 50 epochs using the training dataset and evaluated on the validation dataset. TensorBoard is used to monitor training progress, visualize losses, and track model performance over epochs.

## Predictions and Model Deployment

Post-training, the model's predictions are tested on the test dataset, exhibiting accurate real-world face detection. The trained model is saved in .h5 format, ready for deployment.

## Real-Time Face Detection

Taking a step further, the project showcases the creation of a real-time face detection system using OpenCV. The trained model is incorporated to detect faces in real-time video streams, illustrating the practical application of the developed model.

## Cloning and Running
To run this project locally, follow these steps:

1. Clone the repository:

"""shell
git clone https://github.com/Rhythm1821/Tensorflow-Face-Detection.git


2. Navigate to the project directory:

cd Tensorflow-Face-Detection

3. Install the required dependencies:

pip install -r requirements.txt

4. Run the real-time face detection Python file:

python3 main.py