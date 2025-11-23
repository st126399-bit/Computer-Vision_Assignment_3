# Computer-Vision_Assignment_3
Image Segmentation and Generative Modeling Project

Python Environment: 3.12.8

This repository contains implementations of three computer vision tasks: Graph Cut Segmentation, Fully Convolutional Network (FCN) Semantic Segmentation, and Variational Autoencoder (VAE) for image generation. Each task includes a Jupyter notebook with clear code, comments, and visualizations.

Task 1: Graph Cut Segmentation

Objective:
Segment a person in given images using graph-based segmentation.

Instructions:

Input: Two images (asm-1, asm-2).

Generate bounding boxes using any deep learning-based object detector (pretrained).

Apply cv2.grabCut for graph-based segmentation with 1, 3, and 5 iterations.

Visualize results:

Original images
User masks
Segmentation results (foreground only and overlay)

Deliverable:

Jupyter notebook with code and comments.

Visual comparisons of results across different iteration counts.

Task 2: Fully Convolutional Network (FCN)

Objective:
Implement FCN for semantic segmentation and analyze architectural/training choices.

Dataset:

Small subset of Pascal VOC dataset (10–20 images).

Split into train/test.

Preprocess: Resize, normalize, convert masks to class indices.

Model Implementation:

FCN variants: FCN-32s
Backbone: Pretrained VGG (remove final FC layers).
Upsampling methods: 1) Transpose convolution 2) Bilinear interpolation.

Training:

Loss: CrossEntropyLoss
Optimizer: Adam or SGD
Metrics: Mean IoU, Pixel Accuracy
Train for 20 epochs or until convergence, I tried only 6/20 due to configuration

Visualization & Analysis:

Display segmentation results for at least 3 test images.
Compare transpose convolution vs bilinear interpolation with a table and visuals.
Include short analysis of model performance.

Deliverable: Jupyter notebook with clear code, training curves, and results.

Task 3: Variational Autoencoder (VAE)

Objective:
Learn latent representations of images, generate new samples, and analyze latent dimensionality effects.

Dataset: MNIST (preprocessed)

Model Implementation:

Encoder: 3–4 Conv layers → flatten → Linear → output μ and log(σ²)
Latent dimension: 128
Decoder: Transpose convolutions to reconstruct images
Training:
Loss: Reconstruction (MSE or BCE) + KL divergence
Optimizer: Adam, Epochs: 50

Visualization & Analysis:
Reconstruction visualization
Latent space exploration
Sample new images by generating random latent vectors
Interpolation between two latent vectors
Experiment with latent dimension 256 and compare reconstruction quality

Deliverable:Jupyter notebook with code, visualizations, and analysis.

Dependencies
Python 3.12.8
PyTorch
torchvision
OpenCV
NumPy
Matplotlib
Jupyter Notebook
