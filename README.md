# img-classification-pista

**By Sarah Benkoussa**

**Evaluating how AI/ML models and neural network architectures can classify pistachio images.**

Data Source: [https://www.muratkoklu.com/datasets/](https://www.muratkoklu.com/datasets/)

## Overview

This project explores and evaluates different Convolutional Neural Network (CNN) architectures for classifying pistachio breeds using image data. The goal is to determine whether increasing model complexity improves classification performance without causing overfitting.

I compare two CNN architectures:

* **Model 1:** A simple baseline CNN
* **Model 2:** A more complex CNN with additional convolutional layers, batch normalization, and linear layers

Both models are trained to classify pistachios into two categories:

* **Kirmizi Pistachio**
* **Siirt Pistachio**

The dataset contains:

* 1,232 Kirmizi breed pistachio images
* 916 Siirt breed pistachio images

Images are resized to **224×224** and normalized before training.

## Technology Used

* Python
* PyTorch
* Torchvision
* NumPy
* Pandas
* Matplotlib
* OpenCV
* Convolutional neural networks

## Installation

1. Clone the repository:

```bash
git clone https://github.com/s-b-0/img-classification-pista
cd pistachio-classification
```

2. Install dependencies:

```bash
pip install torch torchvision numpy pandas matplotlib opencv-python
```

3. Download the dataset from:
   [https://www.muratkoklu.com/datasets/](https://www.muratkoklu.com/datasets/)

4. Place `BothPistachio.zip` in the project root directory and run the notebook.
   The dataset will be automatically extracted.

## Data Preprocessing

Images undergo the following transformations:

* Resize to 224×224
* Convert to tensor
* Normalization

The dataset is then split into:

* 75% training
* 25% testing
* Batch size: 64

## Model Architectures

### Model 1 – Simple CNN (Baseline)

Architecture:

* Conv2D (3 → 32)
* ReLU
* MaxPool2D
* Flatten
* Linear (394272 → 2)

**Loss:** CrossEntropyLoss

**Epochs:** 10

*Model 1 achieved accurate performance with relatively low complexity and minimal overfitting.*

### Model 2 – More Complex CNN

Architecture:

* Conv2D (3 → 32)
* ReLU
* MaxPool2D
* Conv2D (32 → 32)
* ReLU
* BatchNorm2D
* Flatten
* Linear (380192 → 37)
* ReLU
* BatchNorm1D
* Linear (37 → 2)

**Loss:** CrossEntropyLoss

**Epochs:** 10

*While Model 2 improved training accuracy significantly, test accuracy improved only marginally, indicating mild overfitting due to increased complexity.*

## Key Findings

* A simple CNN performs nearly as well as a more complex architecture.
* Increased layers improveed the training accuracy more than test accuracy.
* Model 2 slightly reduces misclassifications but does not substantially outperform Model 1.
* The dataset size may limit gains from deeper architectures.
