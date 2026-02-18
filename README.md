# Heart Disease Prediction Model using Logistic Regression

A machine learning project that predicts the presence or absence of heart disease in patients using medical diagnostic features and logistic regression classification.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Model](#model)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project develops a machine learning model to predict heart disease using logistic regression. The model is designed to serve as a clinical decision support tool that helps identify patients at risk of heart disease based on various medical diagnostic measurements. The project emphasizes achieving a balance between accuracy, precision, and recall for reliable healthcare applications.

**Key Highlights:**
- ✅ Achieves **86% accuracy** on test data
- ✅ Excellent **AUC of 0.92** demonstrating strong discriminative ability
- ✅ Uses hyperparameter tuning for optimal performance
- ✅ Implements feature standardization for better model convergence
- ✅ Provides comprehensive model evaluation metrics

## 📊 Dataset

The project uses a preprocessed heart disease dataset containing **303 patient records**. The data is split using:
- **Training set:** 70% of the data (stratified sampling)
- **Testing set:** 30% of the data (stratified sampling)

Stratified sampling ensures that the class distribution (disease vs. no disease) is maintained in both training and testing sets.

## 🔍 Features

The model uses **13 predictor variables** to make predictions:

| Feature | Description | Type |
|---------|-------------|------|
| **Age** | Patient's age in years | Continuous |
| **Sex** | Gender (1 = male, 0 = female) | Binary |
| **Chest Pain Type** | Type of chest pain (1-4) | Categorical |
| **BP** | Resting blood pressure (mm Hg) | Continuous |
| **Cholesterol** | Serum cholesterol level (mg/dl) | Continuous |
| **FBS over 120** | Fasting blood sugar > 120 mg/dl | Binary |
| **EKG Results** | Resting electrocardiographic results | Categorical |
| **Max HR** | Maximum heart rate achieved | Continuous |
| **Exercise Angina** | Exercise-induced angina | Binary |
| **ST Depression** | ST depression induced by exercise | Continuous |
| **Slope of ST** | Slope of the peak exercise ST segment | Categorical |
| **Number of Vessels Fluro** | Number of major vessels colored by fluoroscopy (0-3) | Categorical |
| **Thallium** | Thallium stress test results | Categorical |

**Target Variable:** Presence (1) or Absence (0) of heart disease

## 🤖 Model

### Algorithm: Logistic Regression

The project implements logistic regression with several optimization techniques:

1. **Feature Standardization:** StandardScaler is applied to normalize all features, ensuring they have zero mean and unit variance
2. **Hyperparameter Tuning:** GridSearchCV with 5-fold cross-validation to find optimal parameters
3. **Optimization Metric:** F1-score (balances precision and recall)
4. **Model Evaluation:** Comprehensive metrics including confusion matrix, classification report, ROC curve, and AUC

### Model Pipeline
```
Raw Data → Feature Scaling → Train-Test Split → Logistic Regression → Hyperparameter Tuning → Final Model
```

## 📈 Results

### Best Model Performance (Hyperparameter-Tuned)

| Metric | Score |
|--------|-------|
| **Accuracy** | 86% |
| **Precision (Disease)** | 87% |
| **Recall (Disease)** | 79% |
| **F1-Score (Disease)** | 83% |
| **AUC-ROC** | 0.92 |

### Model Comparison

| Model | Accuracy | Recall | Precision | F1-Score |
|-------|----------|--------|-----------|----------|
| Baseline | 84% | 76% | 87% | 81% |
| **Tuned (Final)** | **86%** | **79%** | **87%** | **83%** |

The tuned model shows improvement across all key metrics while maintaining excellent discriminative ability (AUC = 0.92), making it suitable for clinical decision support.

## 🚀 Installation

### Prerequisites
- Python 3.8+ (Python 3.9 or higher recommended)
- Google Colab (optional, for running the notebook)
- Jupyter Notebook (for local execution)

### Required Libraries
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

Or install with specific tested versions from requirements.txt:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Running in Google Colab
1. Open the notebook: `Heart Disease Prediction Model using Logistic Regression.ipynb`
2. Mount Google Drive (if your dataset is stored there)
3. Update the `file_path` variable to point to your preprocessed dataset
4. Run all cells sequentially

### Running Locally
1. Clone this repository:
```bash
git clone https://github.com/DataDarling/Heart-Disease-Prediction-Model.git
cd Heart-Disease-Prediction-Model
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook "Heart Disease Prediction Model using Logistic Regression.ipynb"
```

4. Update the dataset path in the notebook and run all cells

### Using the Model for Predictions

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load your trained model
# ... (train the model as shown in the notebook)

# Prepare new patient data
new_patient = [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]

# Scale the features
new_patient_scaled = scaler.transform(new_patient)

# Make prediction
prediction = model.predict(new_patient_scaled)
probability = model.predict_proba(new_patient_scaled)

print(f"Prediction: {'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'}")
print(f"Probability: {probability[0][1]:.2%}")
```

## 📁 Project Structure

```
Heart-Disease-Prediction-Model/
│
├── Heart Disease Prediction Model using Logistic Regression.ipynb
│   └── Main notebook with complete analysis and model development
│
└── README.md
    └── Project documentation (this file)
```

## 🛠️ Technologies Used

- **Python 3.8+** - Primary programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning library
  - LogisticRegression
  - StandardScaler
  - GridSearchCV
  - train_test_split
  - Classification metrics
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical data visualization
- **Google Colab** - Development environment

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/improvement`)
3. **Commit** your changes (`git commit -m 'Add some improvement'`)
4. **Push** to the branch (`git push origin feature/improvement`)
5. **Open** a Pull Request

### Ideas for Contribution
- Try different algorithms (Random Forest, SVM, Neural Networks)
- Add feature engineering techniques
- Implement cross-validation strategies
- Create a web interface for predictions
- Add more visualizations
- Improve documentation

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 📞 Contact

For questions or feedback, please open an issue in the GitHub repository.

---

**Note:** This model is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.