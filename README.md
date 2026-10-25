# Heart-disease

# ❤️ Heart Disease Prediction Using Machine Learning

## 📌 Project Overview

This project focuses on predicting the possibility of heart disease using Machine Learning classification algorithms. I developed a machine learning application that accepts patient health information and generates a prediction through an interactive Streamlit web interface.

The project also involves comparing multiple classification algorithms to identify the best-performing model based on evaluation metrics, including accuracy.

**Objective:** To build a machine learning model that predicts heart disease outcomes and compare classification algorithms to determine which performs best on the selected dataset.

## 🎯 Project Objectives

- Analyze patient health data for heart disease prediction.
- Preprocess the dataset and prepare features for model training.
- Implement and compare multiple classification algorithms.
- Evaluate model performance using accuracy and other classification metrics.
- Identify the best-performing algorithm based on experimental results.
- Deploy the selected model using Streamlit.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Data Visualization:** Matplotlib, Seaborn (if used)
- **Web Application:** Streamlit
- **Model Persistence:** Joblib
- **Development Environment:** Jupyter Notebook / VS Code

## 📊 Dataset and Features

The project uses patient health information to predict heart disease outcomes.

The application accepts the following inputs:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise-Induced Angina
- Oldpeak
- ST Slope

The input features are transformed into the expected format before being passed to the trained model.

*Note: Add your dataset source, dataset size, and target-variable details here.*

## 🤖 Machine Learning Algorithms

I evaluated multiple classification algorithms to compare their predictive performance.

Add the algorithms you actually implemented in your notebook or training code. For example:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree Classifier
4. Random Forest Classifier
5. Support Vector Machine (SVM)
6. Naive Bayes
7. Gradient Boosting Classifier

## 📈 Model Evaluation and Accuracy Comparison

The classification models are compared using evaluation metrics to identify the best-performing algorithm.

| Algorithm | Accuracy |
|---|---:|
| Logistic Regression | 87% |
| KNN | 88% |
| Decision Tree | 86% |
| Random Forest | 76% |
| SVM | 86% |

**Best-performing algorithm:** [Insert the algorithm with the highest measured accuracy]

**Best accuracy:** [Insert your actual accuracy percentage]

The final model selected for the Streamlit application is K-Nearest Neighbors (KNN). The model is loaded from a saved file and used to generate predictions for user-provided patient information.

*The final model selection and highest accuracy should be verified against your actual evaluation results.*

## ⚙️ Project Workflow

1. Load and explore the dataset.
2. Clean and preprocess the data.
3. Encode categorical variables.
4. Split the data into training and testing sets.
5. Scale the features where required.
6. Train multiple classification algorithms.
7. Evaluate and compare model performance.
8. Select the final model based on evaluation results.
9. Save the trained model, scaler, and expected feature columns.
10. Build an interactive Streamlit application for prediction.

## 🖥️ Streamlit Application

The application allows users to enter patient information and obtain a machine learning prediction.

**Main features:**

- Interactive patient-information form.
- Input validation through Streamlit controls.
- Automatic feature-column alignment.
- Feature scaling using the saved scaler.
- Prediction using the trained KNN model.
- Display of positive or negative prediction results.
- Error handling for missing model files and prediction failures.

## 📁 Project Structure

```text
heart-disease-prediction/
│
├── app.py
├── KNN_heart.pkl
├── scaler.pkl
├── columns.pkl
├── requirements.txt
├── README.md
│
├── dataset/
│   └── heart.csv
│
└── notebooks/
    └── model_training.ipynb
```

*This is a suggested structure. Include only the files and folders that actually exist in your repository.*

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project folder

```bash
cd heart-disease-prediction
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

If you do not have a requirements file yet, install the core packages:

```bash
pip install streamlit pandas scikit-learn joblib
```

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser. Enter the patient details and click **Predict** to view the result.

## 📚 Key Learnings

Through this project, I gained practical experience in:

- Building classification models using Scikit-learn.
- Comparing multiple machine learning algorithms.
- Evaluating model performance using classification metrics.
- Encoding categorical variables and scaling numerical features.
- Saving and loading trained machine learning models.
- Integrating a machine learning model with Streamlit.
- Building an interactive end-to-end machine learning application.

## ⚠️ Disclaimer

This project is intended for educational and research purposes only. Its predictions are not medical diagnoses and should not be used to make medical decisions. Consult a qualified healthcare professional for medical evaluation.

## 👨‍💻 Author

**Your Name**

- GitHub: [Your GitHub Profile URL](https://github.com/swapneshpanda500-a11y/Heart-disease/edit/main/README.md)

