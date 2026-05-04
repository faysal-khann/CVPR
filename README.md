# CVPR — Coursework Notebooks

This repository contains course assignments organized by directory (e.g., **MID**, **FINAL**).

---

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
# Mid Assignment 1 — k-NN on 32×32 Grayscale Animal Images

This assignment implements a **k-Nearest Neighbors (k-NN)** classifier for **multi-class animal image classification** using a simple **raw pixel** representation. The goal is to compare **Euclidean (L2)** vs **Manhattan (L1)** distances and select the best configuration using **5-fold cross-validation**.

## Task Summary
- Input images are **animal images**
- Each image is:
  - converted to **grayscale**
  - resized to **32×32**
  - flattened into a **1D feature vector** (32×32 = 1024 features)
- A **k-NN classifier** is trained and evaluated using:
  - multiple values of **k**
  - two distance metrics: **L1** and **L2**
  - **5-fold cross-validation** for model selection

## Method
### 1) Preprocessing
- Convert RGB images to **grayscale**
- Resize to **32×32**
- Flatten pixel grid into a vector

This produces a basic pixel-based feature space (no feature extraction).

### 2) Model Selection (5-Fold Cross-Validation)
We evaluated different values of **k** using:
- **Euclidean distance (L2)**
- **Manhattan distance (L1)**

## Results
### Cross-Validation
Best configuration:
- **Distance metric:** Manhattan (L1)
- **k:** 9
- **Mean CV accuracy:** **0.4500**

Interpretation: For this grayscale pixel representation, **L1 distance** (sum of absolute differences) measures similarity more effectively than L2 (squared differences).

### Final Evaluation (Train/Test Split)
Using the best setting (**L1, k=9**):
- Train on **80%** of the data
- Test on **20%** of the data
- **Test accuracy:** **0.4667**

### Confusion Matrix Notes
The confusion matrix indicates:
- many correct classifications occur
- there is still **noticeable confusion between certain animal classes**
- this likely happens due to:
  - visual similarity between animals at low resolution
  - limitations of using **raw pixel values** (sensitive to pose/lighting/shift)

## Conclusion
k-NN with **Manhattan distance (L1)** achieved **moderate performance** on this dataset. Performance could likely be improved by using a stronger feature representation, such as:
- feature extraction (HOG / CNN embeddings)
- dimensionality reduction (e.g., PCA) before k-NN
- improved normalization and preprocessing

## How to Run (General)
1. Ensure you have Python installed.
2. Install common dependencies (example):
   ```bash
   pip install numpy matplotlib scikit-learn opencv-python
   ```
3. Run the notebook/script for preprocessing, cross-validation, and final testing.

## Key Takeaway
**Distance metric choice matters.** In this experiment, **L1 + k=9** performed best under grayscale pixel-based features.

## MID / Assignment 2 — NumPy Neural Network (5‑Class Classification)

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




