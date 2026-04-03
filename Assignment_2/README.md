# Assignment 2 – Image Filtering and CNN Classification

## Overview

This project contains two parts:

- Part 1: Image filtering using convolution kernels (edge detection, blurring, Sobel operator)
- Part 2: Convolutional Neural Network (CNN) for CIFAR-10 image classification

---

## Environment Setup

1. Create the environment:

```
conda create -n ml_venv python=3.10
```

2. Activate the environment:

```
conda activate ml_venv
```

3. Install dependencies:

```
pip install -r requirements.txt
```

---

## Running the Project

1. Start Jupyter Lab:

```
jupyter lab
```

2. Open and run:

- `image_filtering.ipynb`
- `build_cnn.ipynb`

Run all cells from top to bottom.

---

## Results

- Final CNN Test Accuracy: **75%**
- Training Time: ~333 seconds
- Model: 3-layer CNN with dropout

---

## Files

- `image_filtering.ipynb` – Part 1 implementation
- `build_cnn.ipynb` – Part 2 implementation
- `requirements.txt` – dependencies
- `report.pdf` – analysis and findings

---

## Notes

- Best model selected using validation loss
- Overfitting observed after optimal epoch
- Deeper CNN improved feature extraction and accuracy
