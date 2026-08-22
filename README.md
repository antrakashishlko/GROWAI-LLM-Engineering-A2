# GROWAI LLM Engineering – Assignment 2

## Neural Network XOR

This project demonstrates how a simple neural network can learn the XOR logical operation using PyTorch.

## Purpose

The purpose of this assignment is to understand how a neural network learns a non-linear relationship using a hidden layer, activation functions, loss calculation, backpropagation and optimization.

## Features

- Implements the XOR dataset using PyTorch tensors
- Builds a neural network with a hidden layer
- Uses ReLU activation in the hidden layer
- Uses Sigmoid activation for binary classification
- Uses Binary Cross Entropy Loss
- Uses the Adam optimizer
- Trains the model for 5000 epochs
- Displays training loss during training
- Generates and displays final predictions

## Requirements

- Python 3.x
- PyTorch

## Installation

Install the required dependency using:

pip install -r requirements.txt

## How to Run

Run the Python script:

python xor_neural_network.py

The program trains the neural network on the XOR dataset and displays the training loss followed by the final predictions.

## Project Files

- `xor_neural_network.py` – Main Python program
- `requirements.txt` – Required Python dependency

## Real-World Relevance

Although XOR is a simple logical problem, this assignment demonstrates how neural networks learn non-linear patterns. Similar concepts are used in classification, pattern recognition, anomaly detection and decision-making systems.

## Edge Case

The model may fail to learn the XOR pattern if the learning rate is unsuitable, the number of epochs is insufficient, or the network does not have enough capacity. These issues can be addressed by tuning the training configuration or modifying the network architecture.

## Assignment

GROWAI LLM Engineering & Generative AI – Assignment 2
