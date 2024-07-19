# K-Nearest Neighbors (KNN) Classifier for Predicting Talk Time

This repository contains a Python script that demonstrates data preprocessing, visualization, model training, and evaluation using the K-Nearest Neighbors (KNN) algorithm. The goal is to predict the `talk_time` feature based on other attributes in the dataset.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)
- [Contact](#contact)

## Introduction
This project uses the K-Nearest Neighbors (KNN) algorithm to predict the `talk_time` feature. The script includes:
- Loading and preprocessing the dataset
- Identifying and visualizing binary features
- Handling missing values and zero-width screen issues
- Splitting the dataset into training and testing sets
- Training the KNN model
- Evaluating the model's performance

## Requirements
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Dataset
The dataset used in this project should be available in CSV format. Ensure the dataset is correctly placed in the working directory. Replace the placeholder path in the script with the actual path to your dataset file.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/knn-talk-time-prediction.git
    ```
2. Change the directory:
    ```bash
    cd knn-talk-time-prediction
    ```
3. Install the required Python packages:
    ```bash
    pip install numpy pandas matplotlib scikit-learn
    ```

## Usage
1. Ensure the dataset is placed in the working directory and replace the dataset path in the script with the actual path to your dataset file.

2. Run the script:
    ```bash
    python knn_talk_time_prediction.py
    ```

3. The script will perform the following steps:
    - Display sample records from the dataset
    - Find the number of records and features, and display numeric descriptions
    - Identify and visualize binary features
    - Check for and replace missing values
    - Handle zero-width screen issues
    - Identify the target feature and display class-wise record counts
    - Split the dataset into training and testing sets
    - Train the KNN model
    - Display actual vs. predicted values
    - Display the confusion matrix
    - Display the model's accuracy

## Results
The script will output:
- Sample records from the dataset
- Number of records and features, and numeric description of features
- Binary features and their visualizations
- Class-wise record counts for the target feature
- Actual vs. predicted values
- Confusion matrix
- Model accuracy
