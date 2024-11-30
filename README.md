# **Advanced Human Activity Recognition Using the WISDM Dataset**

## **Overview**

This project extends our previous work on human activity recognition using the **WISDM (Wireless Sensor Data Mining)** dataset. We explore advanced data augmentation techniques specifically designed for time-series data and experiment with different deep learning architectures to enhance model performance. The goal is to address overfitting, improve generalization, and achieve higher accuracy in classifying human activities based on accelerometer data.

---

## **Table of Contents**

- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Advanced Data Augmentation](#advanced-data-augmentation)
- [Model Development](#model-development)
- [Results](#results)
- [Conclusion](#conclusion)
- [Acknowledgments](#acknowledgments)

---

## **Dataset**

We use the **WISDM v1.1** dataset, which contains accelerometer data collected from smartphones carried by 36 users performing six activities:

- Walking
- Jogging
- Upstairs
- Downstairs
- Sitting
- Standing

**Note**: Download the dataset from the [WISDM Lab Website](https://www.cis.fordham.edu/wisdm/dataset.php) and place it in the `data/` directory.

---

## **Project Structure**

```
HAR-WISDM-Advanced/
├── code/
│   ├── code.py
│   
├── data/
│   └── WISDM_ar_v1.1_raw.txt
├── README.md
└── requirements.txt
```

- **code/**: Contains Python scripts for data processing, augmentation, model training, and evaluation.
- **data/**: Contains the dataset files.
- **README.md**: Project description and instructions.
- **requirements.txt**: Python dependencies.

**Note**: The project structure is currently simple, with only `code/` and `data/` directories. Further organization may be implemented in the future.

---

## **Installation**

- Python 3.6 or higher
- Virtual environment tool (recommended)


## **Advanced Data Augmentation**

To enhance the diversity of the training data and address overfitting, we implemented advanced data augmentation techniques tailored for time-series data:

1. **Time Warping**: Randomly stretches or compresses sections of the time series to simulate variations in activity speed.

2. **Window Slicing**: Randomly selects a sub-window from the time series, introducing variability in the sequence length.

3. **Window Warping**: Applies time warping to a random window within the time series.

4. **Magnitude Warping**: Scales the magnitude of the data over time by applying a smooth, random curve.

5. **Permutation**: Randomly permutes segments of the time series, maintaining local temporal information but altering the global sequence.

6. **Random Sampling with Interpolation**: Randomly samples points from the time series and interpolates to create a new sequence.

**Implementation Highlights:**

- Augmentation functions are applied to underrepresented classes to balance the dataset.
- Augmented data is combined with the original training data and shuffled.
- Class weights are recomputed to reflect the new class distribution.

---

## **Model Development**

We experimented with several deep learning architectures to improve performance:

### **1. CNN-Transformer Model**

- **Architecture**:
  - Convolutional layers for feature extraction.
  - Transformer encoder layer to capture temporal dependencies with self-attention mechanisms.
  - Global average pooling and dense layers for classification.

- **Key Features**:
  - Incorporates attention mechanisms to focus on important time steps.
  - Utilizes multi-head attention for capturing complex patterns.

### **2. CNN-Bidirectional LSTM Model**

- **Architecture**:
  - Convolutional layers for initial feature extraction.
  - Bidirectional LSTM layers to capture sequential patterns in both forward and backward directions.
  - Dropout layers to prevent overfitting.
  - Dense output layer with softmax activation.

- **Key Features**:
  - Effectively captures temporal dependencies in the data.
  - Bidirectional layers enhance context understanding.

### **3. Enhanced CNN-Bidirectional LSTM Model**

- **Architecture**:
  - Similar to the CNN-BiLSTM model with added enhancements:
    - **Attention Mechanism**: Improves the model's focus on relevant parts of the sequence.
    - **Residual Connections**: Helps in training deeper networks by mitigating the vanishing gradient problem.
    - **Layer Normalization**: Stabilizes and accelerates training.
    - **L2 Regularization and Increased Dropout**: Reduces overfitting.
    - **Learning Rate Scheduler**: Adjusts the learning rate during training for optimal convergence.

- **Key Features**:
  - Combines the strengths of CNNs, Bidirectional LSTMs, and attention mechanisms.
  - Enhanced regularization techniques for better generalization.

---

## **Results**

We evaluated the performance of each model using the test dataset.

### **Model Performance**

| Model                               | Test Accuracy |
|-------------------------------------|---------------|
| CNN-GRU-Attention (Baseline)        | 99.07%        |
| CNN-Transformer                     | 98.22%        |
| CNN-Bidirectional LSTM              | 98.98%        |
| **Enhanced CNN-Bidirectional LSTM** | **99.30%**    |

### **Observations**

- **Enhanced CNN-Bidirectional LSTM Model** achieved the highest test accuracy.
- Advanced data augmentation contributed to better model generalization.
- The attention mechanism and residual connections significantly improved performance.

### **Visualization Examples**

- **Learning Curves**: Showed improved training stability and reduced overfitting in the enhanced model.
- **Confusion Matrix**: Demonstrated high precision and recall across all activity classes.
- **Precision-Recall Curves**: Indicated strong model performance, especially in minority classes.

---

## **Conclusion**

By implementing advanced data augmentation techniques and experimenting with different deep learning architectures, we significantly improved the performance of our human activity recognition model. The Enhanced CNN-Bidirectional LSTM model, in particular, demonstrated superior accuracy and generalization capabilities. This project highlights the importance of tailored data augmentation and model enhancements in time-series classification tasks.

---

## **Acknowledgments**

- **WISDM Lab**: For providing the dataset.
- **TensorFlow and Keras**: For the deep learning framework used in this project.


---
