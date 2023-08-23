# Face Detection Project

## Cloning and Running
To run this project locally, follow these steps:

1. Clone the repository:
   
<pre>
git clone https://github.com/Rhythm1821/Tensorflow-Face-Detection.git
</pre>

2. Navigate to the project directory:

<pre>
cd Tensorflow-Face-Detection
</pre>

3. Install the required dependencies:
<pre>
pip install -r requirements.txt
</pre>
   
4. Run the real-time face detection Python file:
<pre>
python3 main.py
</pre>

## Introduction

Welcome to the Face Detection Project! This project aims to implement a robust system detects the face using computer vision techniques and deep learning. It encompasses various stages, from data collection and preprocessing to model building and deployment. This README provides an overview of the project and its key components.

## Table of Contents

* Project Overview
* Data Collection and Annotation
* Data Preprocessing
* Data augmentation and and scaling
* Model Building
* Custom Loss Function
* Custom Model Class
* Model Training and Evaluation
* Predictions and Model Deployment
* Real-Time Face Detection

## Project Overview

The Face Detection Project showcases the process of implementing a face detection system using computer vision and deep learning techniques. It highlights the journey from collecting data and annotating it to deploying a real-time face detection system.

## Data Collection and Annotation

The first step is collecting the data. Using OpenCV, a popular computer vision library, images were captured. These images then underwent annotation for face detection using LabelMe, enabling the drawing of bounding boxes around the faces to create labeled data.

## Data Preprocessing

The annotated data was organized and stored in separate folders for images and labels. Google Colab facilitated access to the data by importing and unzipping the data folders from a GitHub repository. TensorFlow's data pipeline was employed, utilizing tf.data.Dataset and a custom data loading function. The images were loaded using TensorFlow's image decoding functions.

## Data augmentation and scaling
To enhance model robustness, data augmentation was applied using Albumentations. This technique involves introducing variations in the images, like rotations, flips, and changes in brightness. The augmented data was stored separately from the original data.

Images were then scaled to the range of 0-1 by dividing by 255, a standard practice in neural network training.

## Model Building

For the core of the project, a deep learning model was developed. The architecture included a pre-trained VGG16 model as a base and additional Dense and Convolutional layers. The model outputs were designed to handle both classification and regression tasks simultaneously.


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
