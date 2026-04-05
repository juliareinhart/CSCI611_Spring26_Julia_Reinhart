# Assignment 2 – Image Filtering and CNN Classification

## Overview

This project contains two parts:

- Part 1: Image filtering using convolution kernels (edge detection, blurring, Sobel operator)
- Part 2: Convolutional Neural Network (CNN) for CIFAR-10 image classification

---

Repository Structure
Assignment_2/
│
├── build_cnn # Part 2: CNN implementation and training
│ ├── build_cnn.ipynb
│ └── cifar_data.png  
├── image_filtering # Part 1: Image filtering and convolution
│ ├── image_filtering.ipynb
│ └── building2.jpg  
├── CSCI611_Assignment2_Julia_Reinhart.pdf # Final report summarizing results
└── README.md # Project description and instructions

---

## Dataset

This project uses the CIFAR-10 dataset for image classification.

The dataset is automatically downloaded when running the CNN notebook using PyTorch:

```python
train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10('data', train=False, download=True, transform=transform)
```

No manual download is required. If the dataset is not already present, it will be downloaded automatically and stored in a local `data/` directory when the notebook is executed.

For Part 1 (image filtering), the image file used is included in the repository.

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
