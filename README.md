# Image Detection / Classification (Directory: FINAL)

This folder contains the training notebook for an image classification task using a **self‑collected dataset**. The notebook performs **data augmentation**, builds an **AlexNet‑like CNN** in TensorFlow/Keras, trains the model, and saves the trained weights for inference.The accuricy is 99%

## Contents
- `ImageDItection.ipynb` — end-to-end pipeline:
  - dataset loading
  - offline augmentation (to balance classes)
  - train/validation split
  - AlexNet-like model definition
  - training + callbacks
  - saving model + class mapping
  - sample prediction on a single image

## Dataset (Self-Collected)
- The dataset used in this project was **self collected**.
- The notebook expects a directory structure like:

```text
Dataset/
  class_1/
    img1.jpg
    img2.jpg
    ...
  class_2/
    ...
  ...
```

Where each subfolder name is the **class label** (in this project they look like IDs such as `22-49791-3`, etc.).

## Augmentation Output
The notebook creates an augmented dataset directory and expands each class up to a fixed target (200 images per class):

```text
augmented/
  class_1/
    original + augmented images...
  class_2/
    ...
```

Augmentations include small rotations, shifts, shear, zoom, and horizontal flips.

## Training Details
Key default settings used in the notebook:
- Image size: **224 × 224**
- Batch size: **32**
- Validation split: **0.15**
- Epochs: **25**
- Optimizer: **Adam (lr = 1e-4)**
- Loss: **sparse_categorical_crossentropy**
- Metric: **accuracy**

### Model
An **AlexNet-like** architecture is used (Conv → BN → ReLU blocks + pooling, then dense layers and softmax).

## Outputs / Saved Files
During/after training the notebook saves:
- `best_alexnet.h5` — best checkpoint by validation accuracy
- `alexnet_attendance.h5` — final saved model
- `class_indices.json` — mapping of class index → class name
- `epoch_plot.png` — training curves saved via callback (loss/accuracy)

> Note: Keras prints a warning that `.h5` is a legacy format. It still works, but you can also save in the newer `.keras` format if desired.

## How to Run
1. Install requirements:
   - Python 3.x
   - TensorFlow, Keras, NumPy, Matplotlib, etc.

2. Update paths inside the notebook if needed:
   - `DATA_DIR` (original dataset path)
   - `AUG_DIR` (augmented output path)

3. Run all cells in `ImageDItection.ipynb`.

## Inference (Single Image Prediction)
The notebook includes a helper:

- Loads an image
- Resizes to `224×224`
- Normalizes to `[0,1]`
- Predicts with the trained model
- Returns predicted class name and probability
# CVPR — Coursework Notebooks

This repository contains course assignments organized by directory (e.g., **MID**, **FINAL**).

---

## MID / Assignment 1 — NumPy Neural Network (5‑Class Classification)

In this assignment, I implemented a **fully connected neural network from scratch using NumPy** to perform **5-class classification** on a **synthetic 2D dataset**.

### Model Architecture
- **Input**: 2D features (synthetic dataset)
- **Preprocessing**:
  - **Z-score normalization** of features
  - **One-hot encoding** of class labels
- **Network**:
  - **3 hidden layers** with **ReLU** activation
  - **Softmax** output layer (5 classes)
- **Loss**: **Categorical Cross-Entropy**
- **Training**: Gradient descent with backpropagation for **5000 epochs**

### Results
The model showed strong convergence and achieved:
- **Training accuracy**: 100%
- **Test accuracy**: 100%

Performance was verified using:
- overall accuracy metrics
- per-class precision/recall/F1 scores
- **confusion matrix**
- **decision boundary plot** (showing clear separation among the five classes)

### Learning Rate Experiments
I experimented with multiple learning rates to observe training behavior:
- too small → learning is slow
- too large → training becomes unstable
- **0.01 performed best** (fast and stable convergence)

### Challenges & Solutions
Key implementation challenges included:
- Correct **backpropagation across multiple layers**
- **Numerical stability** in softmax and cross-entropy computations
- Careful handling of **array shapes** throughout forward/backward passes

These were addressed using:
- careful gradient derivations and validation
- stability tricks in softmax/loss computation
- clipping / safe computations to avoid overflow/underflow
- consistent shape management for matrix operations

### Future Improvements
Possible extensions to improve performance and robustness:
- better optimizers (Momentum, Adam, RMSProp)
- regularization (L2, dropout)
- **learning rate decay**
- increasing network depth/width
- testing on more complex and realistic datasets

---

## Author
Created by the repository author.

Example usage is included near the end of the notebook.



## Author
Self-collected dataset and training notebook prepared by the repository author.
