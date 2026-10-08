
# 🚗 Traffic Vehicle Classification

A deep learning project for **traffic vehicle image classification** using **PyTorch**.

The main goal of this project is to classify traffic vehicle images into **8 different classes** while systematically investigating the effects of data cleaning, augmentation, model architecture, loss functions, regularization, class balancing, and learning-rate scheduling.

---

## 🚘 Vehicle Classes

The dataset contains **8 vehicle categories**:

| #  | Class       |
| -- | ----------- |
| 🚑 | `ambulance` |
| 🚌 | `autobus`   |
| 🚛 | `kamyun`    |
| 🚐 | `kamyunet`  |
| 🚎 | `minibus`   |
| 🚗 | `savari`    |
| 🚕 | `taxi`      |
| 🚙 | `vanet`     |

---

## 📂 Project Structure

```text
project/
│
├── 📁 dataset/
│   ├── train/
│   ├── test/
│   ├── unclean/
│   ├── merged_train/
│   └── ...
│
├── 📁 models/
│   └── trained models and model implementations
│
├── 📁 experiments/
│   └── experiments and evaluation results
│
├── 📁 scratch/
│   └── draft codes and temporary experiments
│
├── 📁 src/
│   ├── models/
│   ├── datasets/
│   └── training/
│
├── 📁 reports/
│   └── project reports and figures
│
└── 📄 README.md
```

---

## 📝 Folder Overview

### 🧪 `scratch/`

Contains **draft and temporary code** created during development.

This folder includes:

* Initial implementations
* Quick tests
* Debugging scripts
* Experimental code
* Alternative approaches
* Temporary notebooks and scripts

The code in this folder is mainly used during development and experimentation.

---

### 🧠 `models/`

Contains the **model architectures and trained model files** used throughout the project.

Different CNN configurations and trained networks were developed and evaluated during the project.

---

### 🔬 `experiments/`

Contains the main experimental work of the project.

Different aspects of the machine-learning pipeline were investigated, including:

* 🧹 Dataset cleaning
* ⚖️ Class balancing
* 🖼️ Data augmentation
* 🧠 Model architecture
* 📉 Loss functions
* 🛡️ Regularization
* 📐 Weight decay
* 📈 Learning-rate scheduling
* 🎲 Random seeds
* 📊 Model evaluation
* 🤝 Ensemble experiments

The purpose of this folder is to keep the experimental process organized and make it easier to compare different approaches.

---

## 🧹 Dataset Preparation

Before training, the dataset was carefully inspected and processed.

The preprocessing pipeline included:

* 🔍 Detecting corrupted images
* 🗑️ Identifying unsuitable samples
* ♻️ Detecting duplicate images
* 🖼️ Verifying image validity
* 📐 Checking image dimensions and formats
* 🏷️ Checking class labels
* 🧹 Creating cleaned datasets
* 🔀 Creating merged training data
* ✂️ Splitting the data into training and validation sets

Multiple dataset versions were created and evaluated to investigate how data quality affects model performance.

---

## 🖼️ Data Augmentation

Several augmentation strategies were tested to improve model generalization.

The experiments included:

* 🔄 Random horizontal flipping
* 🔃 Random rotation
* 🌈 Color transformations
* ✂️ Random cropping
* 📏 Resizing
* 📡 Noise
* 🌫️ Blurring
* 🎭 Random erasing

Different augmentation configurations were compared experimentally to determine which transformations were actually beneficial for the dataset.

---

## 🧠 Model Development

Multiple neural-network configurations were implemented and evaluated.

The experiments included:

* 🧠 Custom CNN architectures
* 🏗️ Deeper convolutional architectures
* 🔢 Different numbers of convolutional channels
* 🎯 Dropout
* 🛡️ Regularization
* ⚖️ Weight decay
* 📈 Learning-rate scheduling
* 🔧 Fine-tuning

Models were compared based on their validation and test performance rather than training accuracy alone.

---

## 📉 Loss Function Experiments

Different loss-function strategies were investigated.

The main comparison included:

* 🔵 Cross-Entropy Loss
* 🟠 BCE-based classification

These experiments were used to investigate how the choice of loss function affects performance in the **8-class classification problem**.

---

## ⚖️ Class Balancing

The effect of class imbalance was also investigated.

Experiments included:

* 📊 Simulated class imbalance
* ⚖️ Balanced training configurations
* 🔄 Balanced batch sampling
* 📈 Comparison between balanced and imbalanced training

This experiment helped evaluate the sensitivity of the models to different class distributions.

---

## 📈 Learning Rate & Scheduling

Different optimization strategies were tested to improve convergence and generalization.

Experiments included:

* 🎯 Different learning rates
* ⚖️ Weight decay
* 📈 Learning-rate schedulers
* ⏹️ Early stopping
* 🔧 Fine-tuning strategies

The goal was to identify a stable training configuration with strong validation performance.

---

## 📊 Evaluation

Models were evaluated using several classification metrics:

* 🎯 Accuracy
* 🟢 Precision
* 🔵 Recall
* 🟣 F1-score
* 📊 Macro Precision
* 📊 Macro Recall
* 📊 Macro F1-score
* 🔲 Confusion Matrix

Performance was also analyzed separately for each vehicle class to identify difficult classes and common sources of error.

---

## 🔍 Error Analysis

Misclassified samples were analyzed to better understand model weaknesses.

The analysis focused on:

* 🚗 Visually similar vehicle classes
* ⚠️ Low-confidence predictions
* 🖼️ Difficult images
* 📊 Class-specific errors
* 🔄 Confusion between similar categories

This analysis was used to guide further experiments and improve the final model.

---

## 🤝 Ensemble Experiments

Multiple trained models were also evaluated as an ensemble.

The ensemble experiments investigated whether combining predictions from independently trained models could improve:

* 🎯 Classification accuracy
* 🛡️ Robustness
* 📊 Prediction consistency
* 💪 Generalization

Confidence-based prediction strategies were also considered when individual models disagreed.

---

## 🎲 Reproducibility

Random seeds were controlled during training whenever possible.

The training pipeline initializes seeds for:

* 🐍 Python
* 🔢 NumPy
* 🔥 PyTorch
* 🎮 CUDA

Deterministic CuDNN settings were also considered when reproducibility was required.

> ⚠️ GPU-based training can still introduce small numerical differences between runs. Therefore, repeated training runs may produce slightly different model weights and evaluation results even when the same random seed is used.

---

## 🛠️ Technologies

| Technology      | Purpose                                   |
| --------------- | ----------------------------------------- |
| 🐍 Python       | Programming language                      |
| 🔥 PyTorch      | Deep learning framework                   |
| 🖼️ TorchVision | Computer vision utilities                 |
| 🔢 NumPy        | Numerical computation                     |
| 📊 Scikit-learn | Evaluation and machine learning utilities |
| 🖼️ PIL         | Image processing                          |
| 📈 Matplotlib   | Visualization                             |
| 🎮 CUDA         | GPU acceleration                          |

---

## 🔄 Project Workflow

```text
                 📦 Raw Dataset
                       │
                       ▼
                🔍 Data Verification
                       │
                       ▼
                   🧹 Cleaning
                       │
                       ▼
               📂 Dataset Preparation
                       │
                       ▼
             ✂️ Train / Validation Split
                       │
                       ▼
               🖼️ Data Augmentation
                       │
                       ▼
                 🧠 Model Training
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       🧠 CNN       📉 Loss       ⚖️ Balance
     Experiments  Experiments    Experiments
          │            │            │
          └────────────┼────────────┘
                       ▼
             🛡️ Regularization
                       │
                       ▼
               📈 Scheduler
                       │
                       ▼
                📊 Evaluation
                       │
                       ▼
                 🔍 Error Analysis
                       │
                       ▼
               🤝 Model Comparison
                       │
                       ▼
                 🏆 Final Model
```

---

## 🎯 Project Objective

The final objective is to develop a reliable deep learning model capable of classifying traffic vehicle images into **8 different categories**:

```text
ambulance
autobus
kamyun
kamyunet
minibus
savari
taxi
vanet
```

The project follows a systematic experimental approach rather than relying on a single training configuration.

Each major component of the pipeline was independently investigated, compared, and evaluated before selecting the final configuration.

---

## 🚀 Project Status

**Status: ✅ Completed**

The project includes:

* ✅ Dataset inspection and cleaning
* ✅ Dataset preparation
* ✅ Data augmentation experiments
* ✅ CNN model development
* ✅ Class-balancing experiments
* ✅ Loss-function comparison
* ✅ Regularization experiments
* ✅ Learning-rate scheduling
* ✅ Multiple training configurations
* ✅ Model evaluation
* ✅ Error analysis
* ✅ Ensemble experiments
* ✅ Final model selection

---

### 👨‍💻 Author

**Amirhosein Ranjbar**

*Deep Learning & Computer Vision Project* 🚗🤖
