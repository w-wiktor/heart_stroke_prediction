# ❤️ Heart Stroke Prediction – Machine Learning Project

## 🎯 Project goal
The goal of this project was to build a machine learning model that predicts the risk of stroke based on patient medical data.

---

## 📊 Dataset
The dataset contains medical information about patients, including:
- age
- BMI
- glucose level
- heart disease
- smoking status
- other health-related features

The target variable is the occurrence of stroke.

---

## ⚠️ Problem challenge
The dataset is highly imbalanced – stroke cases are much rarer than healthy cases.  
This makes standard accuracy misleading and requires a different evaluation approach.

---

## 🛠 Methods used

- Data cleaning (handling missing values)
- One-hot encoding of categorical variables
- Train/test split with stratification
- Handling imbalanced data using oversampling
- Machine learning models:
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - XGBoost
- Hyperparameter tuning using GridSearchCV
- Model evaluation using recall and confusion matrix

---

## 📈 Results

### Best model: XGBoost (optimized)

- Recall: **0.79**
- Accuracy: **0.70**

---

## 🧠 Key insights

- The dataset was highly imbalanced, which negatively affected the detection of stroke cases
- Using oversampling significantly improved model performance
- The best results were achieved using the XGBoost model after hyperparameter tuning
- Adjusting the classification threshold improved recall from low values to ~79%
- Increasing recall was prioritized over accuracy to better detect patients at risk
- In a medical context, it is more important to detect potential stroke cases than to minimize false alarms

---

## ⚙️ Model improvement

A custom classification threshold (0.35) was applied to improve sensitivity:

- Higher recall (more detected stroke cases)
- Lower accuracy (more false positives)

This trade-off was intentional and aligned with the problem context.

---

## 📊 Feature importance analysis

The feature importance ranking obtained from the XGBoost model shows that:

- **age** is by far the most important predictor of stroke risk
- Other important features include:
  - average glucose level
  - BMI
  - smoking status
- Medical and lifestyle factors such as hypertension and heart disease also contribute to predictions, but to a lesser extent
- Some categorical features (e.g. gender, work type) have relatively low importance in the final model

<br><br>
<img width="1022" height="599" alt="image" src="https://github.com/user-attachments/assets/5421971b-e170-4eb7-bb98-04c8d5ebdef5" />
<br><br>


### 🧠 Key insight
The model confirms that age and metabolic health indicators are the strongest predictors of stroke risk in this dataset.

---

## 🚀 Deployment example

A function was created to simulate real-world usage of the model:

```python
predict_stroke_risk(new_data, model, threshold=0.35)
