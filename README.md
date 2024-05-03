# Optimal Transport in Manifold Learning for Image Classification

This repository explores the efficacy of optimal transport-based manifold learning techniques versus traditional methods in supervised image classification tasks across multiple datasets, including Fashion MNIST, Handwritten MNIST, Dog Breeds Classification, Coil-100, and ImageNet.

## Overview

This project assesses the impact of manifold learning on image classification performance by comparing traditional and optimal transport-based methods. Utilizing datasets with varied complexities, the project aims to evaluate these techniques in terms of visualization quality, stress evaluation, and classification accuracy.

## Summary of Work

### Data Preparation
Data used includes:
- **Fashion MNIST:** Grayscale images of fashion items.
- **Handwritten MNIST:** Grayscale images of handwritten digits.
- **Dog Breeds Classification:** Various dog breed images, adjusted for size and resolution.
- **Coil-100:** Color images of 100 different objects from multiple angles.

All datasets were preprocessed for consistent resolution, pixel normalization, and converted into point clouds using the Dr. Hamm function for embeddings.

### Manifold Learning Techniques
We compared several manifold learning techniques:
- Traditional: PCA, MDS, tSNE, Locally Linear Embedding, and Spectral Embedding.
- Optimal Transport-based: Wassmap.

### Models and Training
Using libraries such as scikit-learn, TensorFlow, and PyTorch, we applied the following models:
- Linear Discriminant Analysis (LDA)
- k-Nearest Neighbors (kNN)
- Support Vector Machine (SVM)
- Random Forest
- Multinomial Logistic Regression

### Performance Comparison
Classification accuracy and error rates were the primary metrics for comparison. Detailed results, including t-test performance evaluations, highlight the potential of optimal transport methods over traditional techniques.

## Conclusions
Initial findings indicate that optimal transport-based methods may offer better embeddings, enhancing classification accuracies. These results suggest promising directions for future research, particularly in integrating these methods with neural networks and expanding to larger datasets.

## Repository Structure
.
├── MDS_ISO_map_embeddings.ipynb       # Notebooks for Multidimensional Scaling and Isomap techniques
├── Wassmap.ipynb                      # Implementation of the Wasserstein map algorithm
└── tSNE_LocallyLinearEmbedding_SpectralEmbedding.ipynb   # Techniques for t-SNE, Locally Linear Embedding, and Spectral Embedding

## Software Setup
Required libraries:
- scikit-learn
- TensorFlow
- PyTorch
- Python Optimal Transport

## Datasets
* Coil-100: https://www.kaggle.com/datasets/jessicali9530/coil100
* Dog Breeds: https://www.kaggle.com/datasets/mohamedchahed/dog-breeds
* Fashion-MNIST: https://www.kaggle.com/datasets/zalando-research/fashionmnist
* Handwritten-MNIST: https://www.kaggle.com/datasets/dillsunnyb11/digit-recognizer

 


