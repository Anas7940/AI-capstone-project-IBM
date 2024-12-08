# AI Capstone Project: Binary Image Classification with ResNet18

This project demonstrates the use of a pre-trained `ResNet18` model for binary image classification (positive vs. negative samples). The project utilizes PyTorch and its associated libraries to fine-tune and evaluate the model on a custom dataset.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Requirements](#requirements)
4. [Setup Instructions](#setup-instructions)
5. [Model Preparation](#model-preparation)
6. [Training and Evaluation](#training-and-evaluation)
7. [Results](#results)
8. [Acknowledgments](#acknowledgments)

## Project Overview
This project uses a transfer learning approach to classify images into two categories: positive and negative. It involves:
- Downloading and preprocessing the dataset.
- Leveraging a pre-trained `ResNet18` model.
- Fine-tuning the model for binary classification.
- Evaluating the model's performance on a validation set.

## Dataset
The dataset consists of two categories of samples:
- **Positive Samples**: Preprocessed tensors in `.pt` format.
- **Negative Samples**: Preprocessed tensors in `.pt` format.

The dataset is divided into training and validation sets:
- Training set: 30,000 samples
- Validation set: Remaining samples

## Requirements
To run this project, you need the following libraries:
- `torch`
- `torchvision`
- `numpy`
- `pandas`
- `matplotlib`
- `Pillow`
- `h5py`

Install the required libraries using:
```bash
pip install torch torchvision numpy pandas matplotlib Pillow h5py
