****CIFAR10-WideResNet****

image classification system using Wide Residual Networks for the CIFAR10 dataset.

An end-to-end deep learning pipeline that classifies 32×32 images into 10 categories with high accuracy, demonstrating the effectiveness of deep residual networks on small-scale image datasets.



**Overview 🎯**

This project implements a WideResNet architecture for multi-class image classification. It achieves 93.99% validation accuracy on the CIFAR10 dataset, showing strong performance and stability over multiple training cycles.



**Architecture 🏗️**

WideResNet (WRN-22-6): 22 layers, widening factor of 6

Residual Blocks: Shortcut connections with Batch Normalization and ReLU activations

Adaptive Average Pooling + Fully Connected Layer: Produces class probabilities



**Dataset 📦
**
CIFAR10: 60,000 32×32 color images in 10 classes (50,000 train, 10,000 test)

Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Normalized using standard CIFAR10 mean and standard deviation



**Training Details 🏋️‍♂️**

Loss Function: Cross-entropy

Optimizer: Adam with weight decay

Learning Rate: Selected via learning rate finder

Batch Size: 256

Total Training: Multiple cycles for best validation accuracy



**Results 📊**

Best Validation Accuracy: 93.99%

Stable training curves across multiple epochs

Model saved in both .pth (weights) and .pkl (full learner with transforms) formats



**Technology Stack 💻**

PyTorch & FastAI: Model implementation, training, and inference

NumPy & Matplotlib: Data handling and visualization

Torchvision: Dataset utilities and transforms
