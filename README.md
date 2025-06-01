# **LSTM-Based CCTV Crime Prediction using Deep Learning**

A deep learning-based solution that leverages Long Short-Term Memory (LSTM) networks to analyze surveillance footage features and predict potential criminal activities.

---

## **Table of Contents**

1. [Introduction](#introduction)  
2. [Features](#features)  
3. [Setup and Installation](#setup-and-installation)  
4. [Usage](#usage)  
5. [Dataset](#dataset)  
6. [Project Workflow](#project-workflow)  
7. [Results](#results)  
8. [Future Enhancements](#future-enhancements)  
9. [Contributing](#contributing)  
10. [License](#license)  

---

## **Introduction**

This project focuses on enhancing public safety by predicting criminal activity using CCTV video data. It applies **LSTM neural networks**, well-suited for sequence data, to detect suspicious or criminal patterns based on temporal features extracted from video footage.

---

## **Features**

- 📹 **Video Frame Extraction**: Preprocesses CCTV footage into sequences of frames.  
- 🧠 **Deep Learning Model**: Uses LSTM for sequential pattern recognition in surveillance data.  
- 🧪 **Anomaly Detection**: Identifies unusual activity based on temporal motion patterns.  
- 📊 **Visualization**: Displays model predictions and behavior timelines.  

---

## **Setup and Installation**

### **Prerequisites**

- Python 3.8 or higher  
- Libraries:
  - `numpy`
  - `pandas`
  - `opencv-python`
  - `tensorflow` / `keras`
  - `matplotlib`
  - `scikit-learn`

### **Installation**

1. Clone the repository:

    ```bash
    git clone https://github.com/ryl-tushar04/cctv-crime-prediction.git
    cd cctv-crime-prediction
    ```

2. Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install required libraries:

    ```bash
    pip install -r requirements.txt
    ```

---

## **Usage**

### **Run the Prediction Pipeline**

1. **Preprocess the video**  
   Extract frames and convert them into a numerical dataset using `video_preprocessing.py`.

2. **Train the model**  
   Use `train_lstm.py` to train the LSTM model on labeled sequences.

3. **Make predictions**  
   Run `predict_crime.py` on new video sequences to predict crime likelihood.

4. **Visualize results**  
   `visualize_results.py` shows prediction scores and flag events on a timeline.

---

## **Dataset**

The dataset used for training consists of labeled surveillance footage containing both normal and criminal activity samples. You can download the dataset from the following Google Drive folder:

🔗 **[Download Dataset](https://drive.google.com/drive/folders/1gRDI_5anXdtJu_rH6r0tpbl_lW5MJX7a?usp=sharing)**

---

## **Project Workflow**

### **1. Data Collection & Preprocessing**

- Extract frames using OpenCV  
- Normalize and shape frame sequences for LSTM input  

### **2. Feature Engineering**

- Use motion vectors, bounding box shifts, or CNN embeddings  
- Create sequences from fixed-length windows

### **3. Model Architecture**

- LSTM with time-distributed dense layers  
- Binary classification (Normal vs Suspicious/Crime)

### **4. Training**

- Optimized with Adam  
- Binary cross-entropy loss  
- Model evaluation using accuracy, precision, recall, F1-score

---

## **Results**

### **Model Performance**

| Metric       | Value  |
|--------------|--------|
| Accuracy     | 88%    |
| Precision    | 86%    |
| Recall       | 84%    |
| F1 Score     | 85%    |

### **Visualization**

1. **Prediction Timeline**  
   Highlights suspicious intervals in video sequences.

2. **Confusion Matrix**  
   Shows model classification performance.

---

## **Future Enhancements**

- Integrate real-time video streaming for live predictions  
- Extend classification into multiple crime categories (e.g., theft, violence)  
- Combine LSTM with CNN or Transformer architectures for spatial-temporal learning  
- Deploy on edge devices for real-time monitoring in smart cities  

---

## **Contributing**

Contributions are welcome! Please follow these steps:

1. Fork the repository  
2. Create a new branch:

    ```bash
    git checkout -b feature-branch
    ```

3. Commit your changes:

    ```bash
    git commit -m "Add your message here"
    ```

4. Push to the branch:

    ```bash
    git push origin feature-branch
    ```

5. Create a pull request  

---

### **Author**

**Tushar Saxena**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/tushar-saxena0410/)
