# CRNN (CNN + RNN) Handwriting Recognition

This repository contains Python code for a CRNN (Convolutional Recurrent Neural Network) model trained for handwriting recognition using the CTC (Connectionist Temporal Classification) loss. The model is designed to recognize text from images of handwritten names.

## Dataset

- **Kaggle Dataset Link**: [Handwriting Recognition Dataset](https://www.kaggle.com/datasets/landlord/handwriting-recognition/data)

## Kaggle Notebook

- **Kaggle Notebook Link**: [Train and Test CRNN Model](https://www.kaggle.com/code/umangmaurya03/traintest)

## Overview of the Dataset

The dataset consists of more than four hundred thousand handwritten names collected through charity projects. Character Recognition utilizes image processing technologies to convert characters on scanned documents into digital forms. It typically performs well in machine-printed fonts. However, it still poses difficult challenges for machines to recognize handwritten characters, because of the huge variation in individual writing styles.

### Dataset Details

- Total number of first names: 206,799
- Total number of surnames: 207,024

The data is divided into the following sets:

- Training set: 331,059
- Testing set: 41,382
- Validation set: 41,382

## Table of Contents

- [Introduction](#introduction)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)

## Introduction

Handwriting recognition is a complex task involving the identification of text within images of handwritten documents. This project showcases the development of a deep learning model for recognizing handwritten text. The model is trained on a dataset of handwritten names and evaluated on a separate test dataset.

## Data Preprocessing

Data preprocessing steps include:

- Loading and visualizing the dataset
- Handling missing values and removing unreadable labels
- Converting lowercase to uppercase for consistency
- Preprocessing the images, including resizing and normalization

## Model Architecture

The deep learning model architecture is based on CRNN:

- Convolutional Neural Network (CNN) for feature extraction
- Bidirectional Long Short-Term Memory (BiLSTM) layers for sequence processing
- Connectionist Temporal Classification (CTC) loss for training

## Training

The model is trained using the CTC loss. The training process involves:

- Splitting data into training and validation sets
- Training the model on the training set
- Evaluating the model on the validation set

## Evaluation

Model performance is evaluated using character-level accuracy on a test dataset:

- Preprocessing test images
- Making predictions on the test images
- Calculating character-level accuracy

## Usage

You can use this code to:

- Train your own CRNN model for handwriting recognition
- Evaluate the model on your own test dataset
- Make predictions on handwritten text images

To predict the text in a specific image, use the provided `predict` function by specifying the image's index.

```python
predict(image_index)
```

Replace `image_index` with the index of the image you want to predict.

For example, to predict the text in the image at index 10261, use:

```python
predict(10261)
```
Output:
![image](https://github.com/Moon-Elf/ML/assets/99338928/dcd016d2-afe9-4fbc-8e82-713a2a3e3f87)
