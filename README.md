# ML-Diabetes-Predictive-Analysis

This project analyzes a medical dataset to predict whether a person is diabetic or not based on diagnostic attributes. 

It includes **exploratory data analysis (EDA)**, **feature engineering**, **data preprocessing**, multiple classification models, and **model comparison using performance metrics**.

The dataset includes key health indicators such as glucose levels, BMI, age, and more. We apply a range of machine learning techniques to build a reliable predictive model.

---

## Dataset Overview

The dataset includes the following columns:

| Feature | Description |
|---------|-------------|
| `Pregnancies` | Number of times pregnant |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | Triceps skin fold thickness (mm) |
| `Insulin` | 2-Hour serum insulin (mu U/ml) |
| `BMI` | Body mass index (weight in kg/(height in m)^2) |
| `DiabetesPedigreeFunction` | A function that scores the likelihood of diabetes based on family history |
| `Age` | Age (in years) |
| `Outcome` | Binary class: 1 (diabetic), 0 (non-diabetic) |

---

## Project Workflow

### 1. 🧼 Exploratory Data Analysis (EDA)
- Analyzed the distribution of all features.
- Identified missing or zero values in `Glucose`, `BloodPressure`, `Insulin`, `BMI`, etc.
- Visualized relationships between each feature and `Outcome`.
- Assessed **correlation** to determine which factors are most indicative of diabetes.

### 2. ⚙️ Data Preprocessing
- Replaced zeroes in physiologically invalid columns with NaNs (if needed).
- Scaled the data using **StandardScaler** for optimal model performance.

### 3. 🤖 Model Training
Trained the following models on the processed dataset:
- **Gaussian Naive Bayes (`GaussianNB`)**
- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**

### 4. 📊 Model Evaluation
Evaluated each model using:

| Metric | Description |
|--------|-------------|
| `Accuracy` | Overall correct predictions |
| `Precision` | TP / (TP + FP) – How many selected items are relevant |
| `Recall` | TP / (TP + FN) – How many relevant items were selected |
| `F1 Score` | Harmonic mean of precision and recall |
| `Support` | Number of actual instances for each class |
| `Macro Avg` | Unweighted average across classes |
| `Weighted Avg` | Average considering class imbalance |

Each model's performance was visualized using a **comparison plot** showing:
- Accuracy
- Precision
- Recall
- F1-Score

---

## 📈 Model Comparison Results

At the end of the notebook, a **bar chart** is plotted to compare the performance of all four models side-by-side, making it easier to choose the best one for deployment or further tuning.

---

## How to Run
```bash
git clone https://github.com/brendanros31/ML-Diabetes-Predictive-Analysis.git

cd ML-Diabetes-Predictive-Analysis

pip install -r requirements.txt
jupyter notebook main.ipynb
```

## Project Structure
```
data/
  raw/
    diabetes.csv

src/
  data_loader.py
  evaluate.py
  features.py
  model.py
  utils.py

EDA.ipynb
main.ipynb
config/config.yaml
```
