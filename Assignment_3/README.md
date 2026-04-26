# CSCI611 Assignment 3: Style Transfer

## Project Overview

This project implements neural style transfer based on the paper **"Image Style Transfer Using Convolutional Neural Networks"** by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge.

The goal of this assignment is to combine the content of one image with the artistic style of another image using a pretrained VGG19 convolutional neural network. The notebook extracts content and style features from selected VGG19 layers, computes Gram matrices for style representation, and optimizes a target image using content and style losses.

## Assignment Structure

This repository contains the work for:

```text
Assignment_3/
```

The folder includes:

```text
Assignment_3/
├── Style_Transfer_Exercise.ipynb
├── README.md
├── requirements.txt
├── houseContent.jpg
├── vanGoghStyle.png
├── outputs/
│   ├── baseline and experiment output images
├── A3_Style_Transfer_Report.pdf
├── sources
│   ├── saved ChatGPT Chat, free version.pdf
```

## Files Included

- `Style_Transfer_Exercise.ipynb` - Jupyter Notebook source code and execution trace
- `README.md` - project setup and usage instructions
- `requirements.txt` - Python dependencies exported from the Conda environment
- `houseContent.jpg` - content image used for style transfer
- `vanGoghStyle.png` - style image used for style transfer
- `outputs/` - generated output images from experiments
- `A3_Style_Transfer_Report.pdf` - final written report

## Main Concepts Implemented

This project follows the neural style transfer method from Gatys et al.

Key ideas include:

- Using pretrained VGG19 convolutional layers as feature extractors
- Freezing VGG19 model parameters
- Extracting content features from deeper CNN layers
- Extracting style features from multiple CNN layers
- Computing Gram matrices to represent style
- Optimizing a target image using gradient descent
- Comparing the effect of different hyperparameters

## Environment Setup

This project was completed using a Conda virtual environment.

### 1. Create Conda Environment

```bash
conda create -n style_transfer python=3.10
```

### 2. Activate Conda Environment

```bash
conda activate style_transfer
```

### 3. Install Dependencies

```bash
pip install torch torchvision matplotlib pillow requests notebook numpy ipykernel
```

### 4. Register the Jupyter Kernel

If Jupyter does not automatically recognize the Conda environment, register it as a kernel:

```bash
python -m ipykernel install --user --name style_transfer --display-name "Python (style_transfer)"
```

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Then open:

```text
Style_Transfer_Exercise.ipynb
```

If needed, select the correct kernel:

```text
Kernel > Change Kernel > Python (style_transfer)
```

## Creating the requirements.txt File

After installing dependencies, generate a `requirements.txt` file from the active Conda environment:

```bash
pip freeze > requirements.txt
```

This saves the exact Python package versions used for the project.

To reinstall dependencies from `requirements.txt`, run:

```bash
pip install -r requirements.txt
```

## How to Run the Notebook

1. Open `Style_Transfer_Exercise.ipynb` in Jupyter Notebook.
2. Run the import cells.
3. Load the pretrained VGG19 model.
4. Load the content and style images.
5. Run the feature extraction function.
6. Run the Gram matrix function.
7. Run the baseline style transfer experiment.
8. Run additional hyperparameter experiments.
9. Save generated images into the `outputs/` folder.
10. Use the generated images in the final PDF report.

## Images Used

### Content Image

The content image is:

```text
houseContent.jpg
```

This image provides the structure and layout that the generated image should preserve.

### Style Image

The style image is:

```text
vanGoghStyle.png
```

This image provides the artistic texture, brush stroke patterns, and color style.

## Baseline Settings

The baseline experiment used:

```python
content_weight = 1
style_weight = 1e6
steps = 2000
learning_rate = 0.003
content_layer = "conv4_2"
```

The baseline result was saved in the `outputs/` folder.

## Hyperparameter Experiments

Several experiments were performed to compare the effect of different settings.

### Style Weight Experiments

The style weight controls how strongly the generated image follows the style image.

Examples tested:

```python
style_weight = 1e4
style_weight = 1e6
style_weight = 1e9
```

Lower style weights preserve more of the original content image. Higher style weights increase the influence of the artistic style, sometimes at the cost of content clarity.

### Step Count Experiments

The number of optimization steps controls how long the target image is updated.

Examples tested:

```python
steps = 300
steps = 1000
steps = 2000
```

Lower step counts produce faster but less refined results. Higher step counts produce more polished and complete style transfer results.

### Content Layer Experiments

The content layer controls what level of image structure is preserved.

Examples tested:

```python
content_layer = "conv2_1"
content_layer = "conv3_1"
content_layer = "conv4_2"
content_layer = "conv5_1"
```

Shallower layers preserve more edges and fine detail. Deeper layers preserve higher-level image structure and allow more stylization.

## Saving Output Images

Generated images were saved using:

```python
plt.imsave("outputs/experiment_name.png", im_convert(target))
```

The `outputs/` folder contains the images used for comparison in the report.

## Displaying Saved Output Images

Saved images can be displayed in the notebook using:

```python
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("outputs/experiment_name.png")
plt.imshow(img)
plt.axis("off")
plt.show()
```

Multiple images can also be displayed side-by-side for comparison:

```python
from PIL import Image
import matplotlib.pyplot as plt

img1 = Image.open("outputs/experiment1.png")
img2 = Image.open("outputs/experiment2.png")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

axes[0].imshow(img1)
axes[0].set_title("Experiment 1")
axes[0].axis("off")

axes[1].imshow(img2)
axes[1].set_title("Experiment 2")
axes[1].axis("off")

plt.show()
```

## Important Implementation Details

### VGG19 Feature Mapping

The notebook maps PyTorch VGG19 layers to the layer names used in the Gatys paper:

```python
layers = {
    "0": "conv1_1",
    "5": "conv2_1",
    "10": "conv3_1",
    "19": "conv4_1",
    "21": "conv4_2",
    "28": "conv5_1"
}
```

### Gram Matrix

The Gram matrix is used to represent style by measuring correlations between feature maps:

```python
def gram_matrix(tensor):
    batch_size, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram
```

### Content Loss

The default content loss uses layer `conv4_2`:

```python
content_loss = torch.mean(
    (target_features["conv4_2"] - content_features["conv4_2"]) ** 2
)
```

### Style Loss

The style loss compares the Gram matrices of the target image and style image:

```python
target_gram = gram_matrix(target_feature)
style_gram = style_grams[layer]

layer_style_loss = style_weights[layer] * torch.mean(
    (target_gram - style_gram) ** 2
)
```

### Total Loss

The target image is optimized using a weighted combination of content and style losses:

```python
total_loss = content_weight * content_loss + style_weight * style_loss
```

## Notes on Runtime

Neural style transfer is computationally expensive because each optimization step passes the target image through VGG19 and updates the image pixels using backpropagation.

Runtime depends on:

- Image size
- Number of steps
- CPU vs GPU availability
- Number of experiments

For faster experimentation, lower step counts such as `300` can be used. For higher-quality final results, `2000` steps gives more refined outputs.

## Final Report

The final PDF report summarizes:

- The Gatys style transfer method
- Implementation details
- Important code snippets
- Hyperparameter experiments
- Visual comparisons
- Findings and conclusions

## Author

Julia Reinhart

CSCI611 Spring 2026

Assignment 3: Style Transfer
