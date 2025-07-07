# PyTorch Tutorial - Learning the Basics

This folder contains my learning journey through the **PyTorch Blitz** tutorial series, where I explored the fundamentals of PyTorch for deep learning.

## 📚 Tutorial Overview

This repository follows the official PyTorch "Deep Learning with PyTorch: A 60 Minute Blitz" tutorial series, covering the essential concepts needed to get started with PyTorch.

## 📋 Tutorial Contents

### 1. [Tensor Tutorial](tensor_tutorial_01.ipynb) 
**What is PyTorch?**
- Introduction to PyTorch tensors
- Basic tensor operations and manipulations
- Understanding tensor shapes and data types
- Converting between NumPy arrays and PyTorch tensors

### 2. [Autograd Tutorial](autograd_tutorial_02.ipynb)
**Autograd: Automatic Differentiation**
- Understanding automatic differentiation in PyTorch
- Working with gradients and the computational graph
- Learning about `requires_grad` and gradient computation
- Exploring the autograd engine for backpropagation

### 3. [Neural Networks Tutorial](neural_networks_tutorial_03.ipynb)
**Neural Networks**
- Building neural networks with `torch.nn`
- Understanding layers, activation functions, and loss functions
- Forward and backward passes
- Creating custom neural network architectures

### 4. [CIFAR-10 Classification Tutorial](cifar10_tutorial_04.ipynb)
**Training a Classifier**
- Complete end-to-end image classification project
- Loading and preprocessing the CIFAR-10 dataset
- Building a Convolutional Neural Network (CNN)
- Training loop implementation with loss and optimization
- Model evaluation and testing
- GPU acceleration basics

## 🎯 Learning Objectives Achieved

Through these tutorials, I learned:

- **Tensor Operations**: How to create, manipulate, and perform computations with PyTorch tensors
- **Automatic Differentiation**: Understanding how PyTorch computes gradients automatically
- **Neural Network Construction**: Building networks using PyTorch's high-level APIs
- **Complete ML Pipeline**: From data loading to model training and evaluation
- **Computer Vision Basics**: Working with image data and convolutional neural networks
- **GPU Computing**: Basics of accelerating computations with CUDA

## 🛠️ Technologies Used

- **PyTorch**: Deep learning framework
- **torchvision**: Computer vision utilities and datasets
- **NumPy**: Numerical computing
- **Matplotlib**: Data visualization
- **Jupyter Notebooks**: Interactive development environment

## 🚀 Key Concepts Covered

### Core PyTorch Concepts
- Tensors and tensor operations
- Autograd and computational graphs
- Neural network modules (`nn.Module`)
- Optimizers and loss functions
- Data loaders and datasets

### Machine Learning Concepts
- Supervised learning
- Image classification
- Convolutional Neural Networks (CNNs)
- Training and validation loops
- Model evaluation metrics

## 📊 CIFAR-10 Project Highlights

The final tutorial demonstrates a complete image classification pipeline:

- **Dataset**: CIFAR-10 (10 classes, 32x32 color images)
- **Architecture**: Custom CNN with convolutional and fully connected layers
- **Training**: Cross-entropy loss with SGD optimizer
- **Results**: Achieved better than random performance on test set
- **Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## 💡 Next Steps

Having completed the PyTorch Blitz tutorials, potential next learning paths include:

- Advanced neural network architectures (ResNet, Transformer, etc.)
- Generative models (GANs, VAEs)
- Natural Language Processing with PyTorch
- Reinforcement Learning
- Production deployment with TorchScript
- Distributed training techniques

## 📝 Notes

- All tutorials are implemented in Jupyter notebooks for interactive learning
- Code includes detailed comments and explanations
- Examples progress from basic concepts to practical applications
- Each notebook can be run independently but builds upon previous concepts

---

**Tutorial Source**: [PyTorch Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

**Learning Period**: Personal study of PyTorch fundamentals
