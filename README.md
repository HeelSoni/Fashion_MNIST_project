# Fashion-MNIST Deep Learning Project

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg?logo=keras&logoColor=white)](https://keras.io/)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c.svg)](https://matplotlib.org/)

An end-to-end deep learning project classifying apparel items from the **Fashion-MNIST** dataset. This project implements, benchmarks, and evaluates both a fully-connected **Dense Baseline Neural Network** and an optimized multi-stage **Convolutional Neural Network (CNN)** equipped with Batch Normalization, Dropout regularization, and Early Stopping.

---

## 📌 Project Overview

The objective of this project is to develop and evaluate deep neural networks capable of categorizing 28×28 grayscale images of fashion articles into 10 distinct categories. While standard MNIST handwritten digits can easily reach >98% accuracy with simple models, Fashion-MNIST poses a significantly more realistic and challenging computer vision benchmark due to complex textures, varying silhouettes, and subtle intra-class variations (such as distinguishing shirts from t-shirts or coats).

### Key Highlights
- **Comparative Modeling**: Directly benchmarks a Multi-Layer Perceptron (MLP/Dense) baseline against a 3-stage Deep Convolutional Neural Network (CNN).
- **Advanced Regularization**: Incorporates `BatchNormalization`, spatial `Dropout` (0.25) and dense `Dropout` (0.40), alongside `EarlyStopping` restoring optimal model weights.
- **In-Depth Diagnostic Analytics**: Features loss and accuracy learning curves, full 10×10 confusion matrix heatmap, per-class accuracy breakdown, and misclassification error analysis.
- **Model Serialization**: Saves the trained production-ready model as `fashion_cnn.keras`.

---

## 📊 Dataset

The project uses the **Fashion-MNIST** dataset (Zalando Research), accessed directly through `keras.datasets.fashion_mnist`:

- **Total Images**: 70,000 grayscale images
  - **Training Set**: 60,000 images (`x_tr`, shape: `(60000, 28, 28, 1)`)
  - **Test Set**: 10,000 images (`x_te`, shape: `(10000, 28, 28, 1)`)
- **Image Dimensions**: 28 × 28 pixels (single channel / grayscale)
- **Preprocessing**: Pixel intensities normalized from `[0, 255]` to `[0.0, 1.0]` as `float32`, and expanded with a trailing channel dimension.
- **Class Labels (10 Categories)**:

| Class Index | Label Name | Description / Category |
|:-----------:|:-----------|:-----------------------|
| 0 | `T-shirt/top` | Casual upper-body wear |
| 1 | `Trouser` | Pants and trousers |
| 2 | `Pullover` | Jumpers / sweaters |
| 3 | `Dress` | Full-length dresses |
| 4 | `Coat` | Outerwear / coats |
| 5 | `Sandal` | Open-toe summer footwear |
| 6 | `Shirt` | Button-down collared shirts |
| 7 | `Sneaker` | Athletic closed footwear |
| 8 | `Bag` | Handbags and luggage |
| 9 | `Ankle boot` | High-top footwear / boots |

---

## 🏗️ Model Architectures

### 1. Dense Baseline Model (MLP)
A fully connected feedforward baseline architecture to establish standard performance:
- **Input**: `(28, 28, 1)`
- **Flatten Layer**: Unrolls 2D image into a 784-dimensional vector.
- **Dense Layer 1**: 256 units, ReLU activation.
- **Dropout Layer**: Rate = 0.30 (mitigates co-adaptation).
- **Dense Layer 2**: 128 units, ReLU activation.
- **Output Layer**: 10 units, Softmax activation.
- **Parameters / Training**: 10 epochs, batch size 128, Adam optimizer, `sparse_categorical_crossentropy` loss.

### 2. Convolutional Neural Network (CNN)
A custom 3-stage feature extractor engineered to capture spatial hierarchies and fine textures:

```text
Input (28 × 28 × 1)
  │
  ├── [Conv2D: 32 filters, 3×3, ReLU, Same Padding]
  ├── [BatchNormalization]
  ├── [MaxPooling2D: 2×2]
  ├── [Dropout: 0.25]
  │
  ├── [Conv2D: 64 filters, 3×3, ReLU, Same Padding]
  ├── [BatchNormalization]
  ├── [MaxPooling2D: 2×2]
  ├── [Dropout: 0.25]
  │
  ├── [Conv2D: 128 filters, 3×3, ReLU, Same Padding]
  ├── [MaxPooling2D: 2×2]
  ├── [Flatten]
  ├── [Dropout: 0.40]
  │
  ├── [Dense: 128 units, ReLU]
  └── [Dense: 10 units, Softmax]
```

---

## ⚙️ Training Process & Regularization

- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy (`sparse_categorical_crossentropy`)
- **Batch Size**: 128
- **Validation Split**: 10% (54,000 training samples, 6,000 validation samples)
- **Early Stopping**: Monitored `val_loss` with `patience=3` and `restore_best_weights=True` to halt training when validation loss stops improving and prevent overfitting.
- **Convergence**: The CNN converged efficiently within 14 epochs, maintaining stability through batch normalization and dropout stages.

---

## 📈 Experimental Results & Performance

### Overall Model Comparison

| Architecture | Parameters & Regularization | Test Accuracy | Test Loss |
|:---|:---|:---:|:---:|
| **Dense Baseline (MLP)** | Flatten + Dense(256) + Dropout(0.3) + Dense(128) | **88.76%** | ~0.32 |
| **Convolutional Neural Network (CNN)** | 3× Conv Blocks + BatchNorm + Dropout(0.25/0.4) + Dense(128) | **91.28%** | **~0.22** |

> **Performance Gain**: The CNN provides a **+2.52% absolute increase** in generalization accuracy over the dense baseline on the unseen 10,000-sample test set.

### Per-Class Accuracy Breakdown (CNN)

| Class | Label | Test Accuracy | Status |
|:---:|:---|:---:|:---|
| 5 | **Sandal** | **98.7%** | 🟢 Exceptional separation |
| 8 | **Bag** | **98.6%** | 🟢 Exceptional separation |
| 1 | **Trouser** | **97.9%** | 🟢 Exceptional separation |
| 9 | **Ankle boot** | **97.1%** | 🟢 High precision |
| 7 | **Sneaker** | **95.6%** | 🟢 High precision |
| 3 | **Dress** | **93.4%** | 🟢 High precision |
| 2 | **Pullover** | **89.1%** | 🟡 Solid accuracy |
| 0 | **T-shirt/top** | **86.5%** | 🟡 Solid accuracy |
| 4 | **Coat** | **84.2%** | 🟡 Moderate confusion |
| 6 | **Shirt** | **71.7%** | 🔴 Most challenging category |

### Diagnostic & Error Analysis
1. **Footwear & Accessories**: Sandals (98.7%), Bags (98.6%), Trousers (97.9%), and Ankle boots (97.1%) are classified with near-perfect reliability due to distinct structural silhouettes.
2. **Upper Apparel Overlap**: Shirts (71.7%) represent the most ambiguous category, occasionally being predicted as T-shirts or Coats. At 28×28 grayscale resolution, fine collar lines and button rows are blurred, leading to natural visual overlap.
3. **Artifacts & Plots**: All training trajectories (`cnn_curves.png`, `dense_curves.png`), confusion matrix heatmap (`confusion_matrix.png`), and error samples (`wrong_predictions.png`) are archived in the `results/` directory.

---

## 📁 Repository Structure

```text
Fashion_MNIST_project/
├── fashion_mnist_project.ipynb   # Main Jupyter notebook with model training & analysis
├── fashion_cnn.keras             # Serialized best CNN model weights
├── results/                      # Evaluation metrics, diagnostic plots & reports
│   ├── cnn_curves.png            # CNN training/validation accuracy & loss curves
│   ├── dense_curves.png          # Dense baseline training/validation curves
│   ├── confusion_matrix.png      # 10x10 confusion matrix heatmap
│   ├── wrong_predictions.png     # Visual grid of misclassified samples
│   ├── fashion_cnn.keras         # Saved CNN model copy
│   └── results.txt               # Raw numerical results and per-class metrics
└── README.md                     # Comprehensive project documentation
```

---

## 🚀 How to Run the Notebook

### 1. Clone the Repository
```bash
git clone https://github.com/HeelSoni/Fashion_MNIST_project.git
cd Fashion_MNIST_project
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install tensorflow numpy matplotlib jupyterlab
```

### 4. Launch Jupyter Notebook
```bash
jupyter notebook fashion_mnist_project.ipynb
```
or launch JupyterLab:
```bash
jupyter lab
```

### 5. Execute Cells
Run all cells from top to bottom (`Cell` -> `Run All`). The notebook will:
1. Automatically download the Fashion-MNIST dataset via `keras.datasets`.
2. Train the Dense baseline and the Deep CNN.
3. Compute evaluation metrics, plot curves, and render confusion matrices.
4. Save the trained model to `fashion_cnn.keras`.

---

## 🛠️ Technologies Used

- **Language**: Python 3.10+
- **Deep Learning Framework**: TensorFlow 2.x & Keras (Sequential API, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Dense, EarlyStopping)
- **Scientific Computing**: NumPy
- **Data Visualization**: Matplotlib
- **Interactive Development**: Jupyter Notebook

---

## 👤 Author

**Heel Soni**
- GitHub: [@HeelSoni](https://github.com/HeelSoni)
- Portfolio: [https://github.com/HeelSoni/My_Portfolio](https://github.com/HeelSoni/My_Portfolio)
