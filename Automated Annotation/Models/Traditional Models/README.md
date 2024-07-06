# Political issues Stance Classification with ML

This project aims to classify stances on political issues using a ML model. The pipeline involves data preprocessing, balancing the dataset, feature extraction, model training, evaluation, and contradiction resolution.

## Table of Contents

1. Introduction
2. Prerequisites
3. Installation
4. Data Pipeline
   - Loading the Dataset
   - Converting String Representations
   - Combining Vectors
   - Balancing the Data
   - Splitting Resampled Features
   - Converting Numpy Arrays Back to Strings
   - Preparing the Data for Model Training
5. Model Training and Evaluation
   - Defining Fixed Sizes for Vectors
   - Extracting Features and Targets
   - Scaling the Features
   - Evaluating the Model
6. Prediction and Contradiction Resolution
   - Resolving Contradictions
   - Predicting and Resolving Contradictions
7. Conclusion

## Introduction

This README file provides a detailed explanation of the workflow used to classify political issues stances. The process includes balancing the dataset using SMOTE, preprocessing the data, training a ML model, evaluating the model, and resolving contradictory predictions.

## Prerequisites

- Python 3.7 or higher
- Required libraries:
  - pandas
  - numpy
  - imbalanced-learn
  - scikit-learn

## Installation

To install the required libraries, run the following command:

```bash
pip install pandas numpy imbalanced-learn scikit-learn

```

## Data Pipeline

### Loading the Dataset

The dataset is loaded from a CSV file into a pandas DataFrame. This dataset contains preprocessed text and context vectors along with target labels indicating stances on political issues.

### Converting String Representations

The text and context vectors are initially stored as strings representing lists. These strings are converted to numpy arrays for numerical processing. This conversion is handled by a custom function that removes the brackets and splits the string into individual float values.

### Combining Vectors

The text and context vectors are combined into a single feature set for each sample. This combined feature set will be used for model training.

### Balancing the Data

The dataset is often imbalanced, meaning some stances may have fewer samples than others. To address this, SMOTE (Synthetic Minority Over-sampling Technique) is applied. SMOTE generates synthetic samples to balance the dataset, ensuring each class has an equal number of samples.

### Splitting Resampled Features

After applying SMOTE, the combined feature set is split back into separate text and context vectors. This step is necessary for maintaining the structure of the original data.

### Converting Numpy Arrays Back to Strings

For consistency and further processing, the numpy arrays are converted back to string representations. This conversion ensures that the data format remains uniform throughout the pipeline.

### Preparing the Data for Model Training

The resampled dataset is loaded and prepared for model training. Fixed sizes for the text and context vectors are defined to ensure uniformity. Features and targets are extracted, and features are scaled using `StandardScaler` to standardize the data.

## Model Training and Evaluation

### Defining Fixed Sizes for Vectors

Fixed sizes are defined for both text and context vectors to ensure that all vectors have the same length. This standardization is crucial for the model to process the data correctly.

### Extracting Features and Targets

Features are extracted from the dataset by combining the text and context vectors. Targets are extracted based on the stance labels. The extracted features and targets are then prepared for model training.

### Scaling the Features

Features are scaled using `StandardScaler` to normalize the data. This scaling improves the performance of the ML model by ensuring that all features are on the same scale.

### Evaluating the Model

The model is evaluated using cross-validation. The dataset is split into training and testing sets multiple times to assess the model's performance. Metrics such as accuracy, precision, recall, and F1 score are calculated for each stance. The `class_weight='balanced'` parameter is used in the ML model to handle any remaining class imbalance.

## Prediction and Contradiction Resolution

### Resolving Contradictions

A custom function is implemented to resolve contradictory predictions. For example, if the model predicts both pro-Brexit and anti-Brexit stances for the same sample, the function resolves the contradiction based on predefined rules and probabilities.

### Predicting and Resolving Contradictions

The model is used to predict stances for the entire dataset. The predictions are then passed through the contradiction resolution function to ensure consistent and accurate stance classification.

## Conclusion

This project demonstrates a comprehensive workflow for classifying stances on political issues using a ML model. The process involves data balancing, feature extraction, model training, evaluation, and contradiction resolution. By following the steps outlined in this README, you can understand and replicate the entire pipeline to achieve accurate and reliable stance classification.
