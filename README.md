# MNIST Neural Network from Scratch (NumPy)

A deep-dive implementation of a Multi-Layer Perceptron (MLP) built entirely from first principles using NumPy. This project demonstrates the mathematical mechanics of backpropagation, matrix optimization, and numerical stability without the use of high-level frameworks like PyTorch or TensorFlow.

## 🚀 Project Overview

The goal of this project was to move beyond the "black box" of modern AI libraries and implement the underlying calculus and linear algebra required to recognize handwritten digits from the MNIST dataset.

Performance Milestone

Final Architecture: 3-Layer Neural Network (Input -> Hidden1 -> Hidden2 -> Output)

Test Accuracy: 90.86%

Dataset: 70,000 images (28x28 pixels) from OpenML mnist_784.

## 🧠 Key Features & Technical Challenges

### Vectorized Matrix Math

To handle 56,000 training images on a machine with 16GB RAM, I utilized NumPy vectorization. By replacing Python loops with matrix operations (SIMD), I avoided memory overflows and achieved a ~100x speedup in training time.

One-Hot Encoding: Implemented using vectorized indexing (np.arange).

Matrix Alignment: Managed complex shape transformations ($784 \rightarrow 10 \rightarrow 10 \rightarrow 10$) ensuring consistent dot products across the chain.

### The Multivariable Chain Rule

Implemented the manual derivation of gradients for a 3-layer network. This involved calculating the "error signal" ($dZ$) as it flows backward through the network, multiplying the weights of the next layer by the derivative of the current activation function.

$$dZ_2 = (W_3^T \cdot dZ_3) \odot \text{ReLU}'(Z_2)$$

### Numerical Stability & Optimization

Solved critical "Exploding Gradient" and overflow issues encountered during the transition to 3 layers:

Stable Softmax: Prevented NaN values by shifting the input vector by its maximum value before exponentiation.

He Initialization: Scaled random weights based on input size ($\sqrt{2/n}$) to keep signal variance consistent across deeper layers.

Activation Functions: Utilized ReLU for hidden layers to mitigate the vanishing gradient problem and Softmax for the output probability distribution.

## 🛠️ Installation & Usage

Prerequisites:

- Python 3.8+

- NumPy

- Matplotlib

- Scikit-learn (only for initial data fetching)

**Running the Model**

The script supports command-line arguments for training or testing:

To Train:

`python nueralnet.py train`

To Test (using saved weights):

`python nueralnet.py test`

Save/Load Functionality

The script includes functionality to save trained weights to a mnist_weights.npz file, allowing for instant inference and visualization without retraining.

### 📊 Visualizing Results

The repository includes a visualization module that selects random images from the test set and plots them alongside the model's prediction using Matplotlib. This is particularly useful for verifying the model's performance on edge cases.

## 🎓 About the Author

I am a Data Science student at the University of Texas at Arlington with a background in Psychology. I am passionate about the intersection of human visual perception and machine learning architecture. This project serves as a cornerstone of my understanding of neural network fundamentals.

---
*Developed as a "First Principles" exploration of Deep Learning.*
