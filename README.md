# CIFAR-10 Image Classification using PyTorch

A deep learning project to classify images from the CIFAR-10 dataset using a custom-built Convolutional Neural Network (CNN) in PyTorch.

##  Project Highlights
- Built and trained a CNN from scratch in PyTorch
- Achieved **70% test accuracy** after 5 epochs
- Implemented real-time visualization of predictions
- Plotted training loss over epochs
- Evaluated performance with a **confusion matrix**

##  Dataset
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)  
Contains 60,000 32x32 color images across 10 categories:  
`plane, car, bird, cat, deer, dog, frog, horse, ship, truck`

## Technologies Used
- Python
- PyTorch
- Torchvision
- Matplotlib
- scikit-learn (for confusion matrix)
- Google Colab

## Model Architecture
- 2 convolutional layers with ReLU + MaxPooling
- 2 fully connected layers
- Trained using Adam optimizer + CrossEntropy loss

## Sample Predictions
Displays a batch of test images with predicted and actual labels.

## Loss Curve
Visualizes training loss across epochs to track learning.

## Confusion Matrix
Provides a breakdown of correct and incorrect predictions for each class.

## What I Learned
This project helped me understand the end-to-end pipeline of building an image classifier using CNNs. I learned how to load and preprocess image data, define CNN architectures, train using PyTorch, and evaluate model performance using visual and statistical tools.

---

