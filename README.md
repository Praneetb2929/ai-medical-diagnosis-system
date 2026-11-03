# 🏥 AI Medical Diagnosis System

# Dataset: Chest X-Ray Images (Pneumonia) by Paul Mooney, download from Kaggle, place in chest_xray/ folder.


### Pneumonia Detection from Chest X-Ray Images Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-orange?logo=pytorch)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?logo=fastapi)
![Computer Vision](https://img.shields.io/badge/Domain-Computer%20Vision-purple)
![License](https://img.shields.io/badge/Project-Academic%20Research-green)

## 📌 Project Overview

The **AI Medical Diagnosis System** is a Deep Learning-based web application designed to classify chest X-ray images into two categories: **NORMAL** and **PNEUMONIA**.

The project uses Artificial Intelligence, Machine Learning, Deep Learning and Computer Vision techniques to analyse chest X-ray images and assist in identifying visual patterns associated with pneumonia.

Two Deep Learning models are implemented and compared:

* **Baseline CNN (Convolutional Neural Network)**
* **DenseNet121 (Pretrained Deep Learning Model)**

The system also integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)**, an Explainable AI technique that generates heatmaps to visualise the image regions influencing the model's predictions.

An interactive web interface allows users to upload chest X-ray images, view predictions, compare model outputs and generate PDF reports.

> **Disclaimer:** This project is developed for academic and research purposes. It is not a clinically validated diagnostic tool and must not replace professional medical examination or diagnosis.

---

## 🎯 Project Objectives

* Develop an AI-based system for pneumonia classification using chest X-ray images.
* Implement and compare Baseline CNN and DenseNet121 architectures.
* Apply Transfer Learning to improve image classification performance.
* Perform image preprocessing and augmentation for better model training.
* Integrate Explainable AI using Grad-CAM.
* Develop a user-friendly web application for image uploading and prediction.
* Evaluate model performance using standard classification metrics.
* Generate downloadable PDF reports containing prediction results.

---

## 🧠 Domains Used

This project combines multiple domains of Computer Science and Healthcare Technology.

| Domain                  | Application                                  |
| ----------------------- | -------------------------------------------- |
| Artificial Intelligence | Intelligent medical image analysis           |
| Machine Learning        | Learning patterns from labelled datasets     |
| Deep Learning           | Neural network-based image classification    |
| Computer Vision         | Understanding and analysing X-ray images     |
| Medical Imaging         | Processing chest radiographs                 |
| Explainable AI (XAI)    | Visualising predictions using Grad-CAM       |
| Image Processing        | Image resizing, conversion and normalisation |
| Data Science            | Dataset analysis and model evaluation        |
| Healthcare AI           | AI-assisted pneumonia screening              |
| Web Development         | Interactive user interface                   |
| Backend Development     | API and prediction processing                |
| Software Engineering    | Integration of project components            |

---

## ✨ Key Features

### 1. Dual Model Architecture

* Implements a Baseline CNN trained from scratch.
* Uses pretrained DenseNet121 through Transfer Learning.
* Compares predictions from both models.

### 2. Chest X-Ray Classification

* Accepts chest X-ray images.
* Classifies images into NORMAL or PNEUMONIA.
* Displays prediction probabilities.

### 3. Explainable AI Using Grad-CAM

* Generates visual heatmaps.
* Highlights image regions that influenced model predictions.
* Helps users understand model behaviour.

### 4. Model Comparison

* Displays results from both models.
* Allows comparison of prediction classes and confidence scores.

### 5. Uncertainty Flagging

* Uses a confidence threshold of 0.65.
* Displays a warning when the maximum prediction probability is below the threshold.
* Encourages specialist review for uncertain predictions.

### 6. Performance Evaluation

* Accuracy
* Precision
* Recall
* Sensitivity
* Specificity
* F1-Score
* AUC-ROC
* Confusion Matrix

### 7. PDF Report Generation

* Generates downloadable PDF reports.
* Includes model predictions and confidence scores.
* Supports visualisation of prediction results.

---

## 🗂️ Dataset

The project uses the **Kermany Chest X-Ray Images (Pneumonia) Dataset**, obtained from Kaggle.

The dataset contains two classes:

* NORMAL
* PNEUMONIA

### Dataset Distribution

| Dataset    | Number of Images | Purpose                      |
| ---------- | ---------------: | ---------------------------- |
| Training   |            5,236 | Training the models          |
| Validation |               16 | Monitoring model training    |
| Testing    |              624 | Evaluating model performance |
| **Total**  |        **5,876** |                              |

### Testing Dataset Distribution

| Class     |  Images |
| --------- | ------: |
| NORMAL    |     234 |
| PNEUMONIA |     390 |
| **Total** | **624** |

**Dataset Source:** [Kaggle – Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

---

## ⚙️ Technologies and Tools Used

| Category                | Technology            |
| ----------------------- | --------------------- |
| Programming Language    | Python                |
| Frontend                | HTML, CSS, JavaScript |
| Backend Framework       | FastAPI               |
| Deep Learning Framework | PyTorch               |
| AI Architectures        | CNN, DenseNet121      |
| Image Processing        | Pillow                |
| Explainable AI          | Grad-CAM              |
| Model Training          | Google Colab          |
| GPU                     | NVIDIA T4             |
| Backend Server          | Uvicorn               |
| PDF Generation          | jsPDF                 |
| Version Control         | Git                   |
| Code Hosting            | GitHub                |
| Dataset Source          | Kaggle                |

---

## 🏗️ System Architecture

The project follows a client-server architecture.

```text
             USER
               |
               |
       Web Application
               |
       HTML / CSS / JS
               |
               |
         Image Upload
               |
               |
         FastAPI Backend
               |
               |
       Image Preprocessing
               |
         224 x 224
               |
        +------+------+
        |             |
        |             |
    Baseline CNN   DenseNet121
        |             |
        |             |
   Prediction     Prediction
        |             |
        +------+------+
               |
        Model Comparison
               |
               |
           Grad-CAM
               |
               |
       Confidence Analysis
               |
               |
       Results Generation
               |
        +------+------+
        |             |
   Web Results     PDF Report
        |             |
        +------+------+
               |
              USER
```

---

## 🔬 Deep Learning Models

### 1. Baseline CNN

A custom Convolutional Neural Network trained from scratch using the chest X-ray dataset.

**Main components:**

* Convolutional Layers
* ReLU Activation
* Max Pooling
* Flatten Layer
* Fully Connected Layers
* Dropout
* Classification Layer

The model learns visual patterns from the training images and predicts NORMAL or PNEUMONIA.

### 2. DenseNet121

DenseNet121 is a pretrained Convolutional Neural Network architecture used through Transfer Learning.

The model initially learns general image features from ImageNet.

For this project:

* Pretrained ImageNet weights are used.
* Existing convolutional layers are frozen.
* The final classification layer is replaced.
* A new classification layer is trained for two output classes.

The final classification layer is:

`Linear(1024, 2)`

### Model Comparison

| Feature           | Baseline CNN      | DenseNet121         |
| ----------------- | ----------------- | ------------------- |
| Architecture      | Custom CNN        | DenseNet121         |
| Initialisation    | Random            | ImageNet pretrained |
| Training Approach | From scratch      | Feature extraction  |
| Classification    | 2 classes         | 2 classes           |
| Class Weighting   | Not applied       | Applied             |
| Reported Accuracy | 69% in second run | 87%                 |

---

## 📊 Model Performance

The DenseNet121 model achieved the following results on the 624-image testing dataset.

### Evaluation Metrics

| Metric               | Result |
| -------------------- | -----: |
| Accuracy             |    87% |
| Sensitivity (Recall) |    94% |
| Specificity          |    76% |
| AUC-ROC              | 0.9405 |

### Confusion Matrix

| Actual / Predicted | NORMAL | PNEUMONIA |
| ------------------ | -----: | --------: |
| NORMAL             |    178 |        56 |
| PNEUMONIA          |     24 |       366 |

### Understanding the Results

* **Accuracy:** Measures the overall percentage of correct predictions.
* **Sensitivity:** Measures how effectively the model identifies actual pneumonia cases.
* **Specificity:** Measures how effectively the model identifies normal cases.
* **AUC-ROC:** Measures the model's ability to distinguish between the two classes.

DenseNet121 demonstrated better classification performance than the baseline CNN in the reported experiment.

---

## 🔍 Explainable AI – Grad-CAM

Grad-CAM stands for **Gradient-weighted Class Activation Mapping**.

It is an Explainable AI technique used to understand which regions of an image influenced a neural network's prediction.

In this project, Grad-CAM generates heatmaps over chest X-ray images.

The heatmap uses different colours to represent the relative influence of image regions on the prediction.

This provides a visual explanation of model behaviour.

**Note:** Grad-CAM visualisations do not prove the presence of pneumonia or establish clinical reasoning.

---

## 🔄 Image Preprocessing

Before feeding an X-ray image into the models, preprocessing is performed.

The main steps include:

1. Loading the uploaded image.
2. Converting the image into RGB format.
3. Resizing the image to 224 × 224 pixels.
4. Converting the image into a tensor.
5. Normalising image values.
6. Passing the processed image to the Deep Learning models.

Training images also undergo augmentation to improve model generalisation.

---

## 🌐 Web Application and API

The project uses FastAPI as the backend framework.

The frontend communicates with the backend through API requests.

### API Endpoints

| Method | Endpoint   | Description                      |
| ------ | ---------- | -------------------------------- |
| GET    | `/`        | Health check                     |
| POST   | `/predict` | Predicts using both models       |
| GET    | `/metrics` | Returns model evaluation metrics |
| GET    | `/static`  | Serves static resources          |

### Prediction Workflow

1. User uploads a chest X-ray image.
2. Frontend sends the image to the FastAPI backend.
3. Backend preprocesses the image.
4. Both Deep Learning models process the image.
5. Predictions and confidence scores are generated.
6. Grad-CAM heatmaps are created.
7. Backend returns the results.
8. Frontend displays predictions and visualisations.

---

## 🛠️ Model Training Configuration

| Parameter               | Value        |
| ----------------------- | ------------ |
| Input Image Size        | 224 × 224    |
| Batch Size              | 32           |
| Number of Epochs        | 5            |
| Optimizer               | Adam         |
| Learning Rate           | 0.001        |
| Deep Learning Framework | PyTorch      |
| Training Platform       | Google Colab |
| GPU                     | NVIDIA T4    |
| Number of Classes       | 2            |

---

## 📁 Project Structure

The repository is organised into different components.

```text
ai-medical-diagnosis-system/
│
├── app/
│   ├── Backend application
│   ├── Frontend resources
│   └── API-related files
│
├── data-pipeline/
│   ├── Dataset preparation
│   ├── Image preprocessing
│   └── Data processing scripts
│
├── models/
│   ├── Baseline CNN
│   ├── DenseNet121
│   ├── Model training
│   └── Model evaluation
│
├── .gitignore
├── LICENSE
└── README.md
```

*Note: This is a high-level representation of the repository. Individual filenames and subdirectories may differ.*

---

## 🚀 Installation and Setup

### Prerequisites

Make sure you have the following installed:

* Python 3.x
* Git
* Visual Studio Code
* Google Chrome or another modern browser
* Python package manager (pip)

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/ai-medical-diagnosis-system.git
```

### Step 2: Navigate to the Project

```bash
cd ai-medical-diagnosis-system
```

### Step 3: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### Step 4: Install Dependencies

Install the required Python libraries according to the project's dependency file.

```bash
pip install -r requirements.txt
```

If the repository does not contain a root-level `requirements.txt`, install the dependencies specified by the application and model-training components.

### Step 5: Configure the Models

Ensure the trained model weights are available at the locations expected by the backend application.

The required model files must be generated during training or obtained from the project's authorised model storage.

### Step 6: Start the Backend

Navigate to the backend application directory and run the FastAPI application using Uvicorn.

```bash
uvicorn <module_name>:app --reload --port 8000
```

Replace `<module_name>` with the actual Python module containing the FastAPI application.

The backend will be available at:

`http://localhost:8000`

FastAPI documentation:

`http://localhost:8000/docs`

### Step 7: Start the Frontend

Open the frontend HTML file using Visual Studio Code Live Server.

The frontend will generally be available at:

`http://localhost:5500`

Make sure the frontend API configuration points to the running backend.

---

## 📈 Advantages

* Automates part of the chest X-ray classification process.
* Implements two different Deep Learning architectures.
* Uses Transfer Learning.
* Provides visual explanations through Grad-CAM.
* Offers an interactive web interface.
* Supports model performance comparison.
* Provides downloadable prediction reports.
* Demonstrates the practical application of AI in healthcare research.

---

## ⚠️ Limitations

* The model is trained specifically for NORMAL and PNEUMONIA classification.
* The dataset is limited to the images available in the selected dataset.
* The original validation set contains only 16 images.
* Model performance may vary with images from different hospitals, equipment and populations.
* The model may produce false-positive and false-negative predictions.
* Grad-CAM does not guarantee that the model is focusing on medically relevant regions.
* The system has not been clinically validated.
* The reported results should not be interpreted as real-world clinical accuracy.

---

## 🔮 Future Enhancements

The project can be improved in several ways:

* Increase the size and diversity of the dataset.
* Use a larger and more representative validation dataset.
* Evaluate additional Deep Learning architectures.
* Improve model generalisation across different medical imaging sources.
* Explore lung segmentation before classification.
* Improve Grad-CAM visualisation and explainability.
* Perform external validation with independent datasets.
* Deploy the application using cloud infrastructure.
* Introduce secure user authentication and patient data protection.
* Conduct clinical evaluation before considering real-world medical use.

---

## 👥 Collaborators

This project was collaboratively developed by:

* **Ayush Kumar**
* **Praneet Biswal**
* **Arnav Naman**

---

## 🎓 Academic Purpose

This project demonstrates the practical application of Artificial Intelligence, Machine Learning, Deep Learning, Computer Vision and Explainable AI in medical image analysis.

It was developed as an academic project to explore how Deep Learning techniques can be applied to chest X-ray classification and how AI-based predictions can be presented through an interactive web application.

---

## 📜 License

This project is intended for academic and educational purposes.

Please refer to the repository's `LICENSE` file for licensing information.

---

## ⭐ Acknowledgements

We acknowledge:

* The creators of the Kermany Chest X-Ray Images dataset.
* The PyTorch development community.
* The FastAPI development community.
* Google Colab for providing the model training environment.
* The open-source research community working on medical image analysis and Explainable AI.

---

**Made with ❤️ for Artificial Intelligence and Healthcare Research.**
