# GameSync-Classifier

Welcome to GameSync-Classifier, a robust solution for video action analysis using advanced deep learning techniques. This project leverages the power of Long Short-Term Memory (LSTM) networks and Recurrent Convolutional Neural Networks (RCNN) to accurately classify actions within video content. By combining convolutional and recurrent neural networks, GameSync-Classifier significantly enhances the accuracy of action detection in sports videos.

## Overview

GameSync-Classifier employs Long-Term Recurrent Convolutional Networks (LRCN) and utilizes the UCF 101 dataset, a widely recognized benchmark in video action recognition, to train and evaluate the model. The primary goal of this project is to advance video analytics by providing a reliable and accurate method for categorizing sports actions in real-world scenarios.

## Features

- **Long-Term Recurrent Convolutional Networks (LRCN)**: Integrates the strengths of convolutional neural networks (CNN) for spatial feature extraction and LSTM networks for temporal sequence learning.
- **UCF 101 Dataset**: Utilizes a comprehensive dataset containing 101 action classes from sports videos, providing a rich and diverse set of action categories for training the model.
- **Enhanced Accuracy**: Combines convolutional and recurrent layers to improve the precision of action recognition within video sequences.
- **Real-World Application**: Focuses on accurately categorizing sports actions, making it highly applicable for use in video analytics, sports analysis, and other real-world scenarios.
- **Data Augmentation**: Implements data augmentation techniques to increase the model's resilience and accuracy in classifying actions under various conditions.

## Installation

To get started with GameSync-Classifier, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/GameSync-Classifier.git
    ```

2. Navigate to the project directory:
    ```bash
    cd GameSync-Classifier-HACKTHEBOX
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the UCF 101 dataset and place it in the designated directory:
    ```bash
    # Follow instructions on the UCF 101 dataset website to download and extract the dataset
    ```

## Usage

To train and evaluate the GameSync-Classifier model, run the following command:

```bash
python train.py --dataset_path /path/to/UCF101 --epochs 50 --batch_size 32
