# Telco Customer Churn Prediction & Analysis

## 🎯 Project Overview
This project aims to analyze **user behavioral and account data** to predict whether a customer will churn (leave the company). We combined **detailed Exploratory Data Analysis (EDA)** with a **predictive classification model** to uncover key churn drivers and build an actionable predictive system.

***

## Project Workflow

### 1️⃣ Data Cleaning & Preprocessing
- **Loaded Dataset:** The project used a dataset containing 19 behavioral and account features and a target variable, **`Churn`**.
- **Data Cleaning:** Missing values in the `TotalCharges` column were handled, and the column was converted to a numeric data type. Irrelevant columns like `customerID` were dropped.
- **Encoding:** Categorical features were converted into a numeric form using **One-Hot Encoding**.
- **Scaling:** Numerical features were standardized using **`StandardScaler`** for optimal model performance.

---

### 2️⃣ Exploratory Data Analysis (EDA)
- Sessions with **Month-to-month contracts** showed a significantly higher churn rate.
- Customers with **Fiber optic** internet service had a higher likelihood of churning.
- **Tenure** was a strong predictor — shorter tenure correlated with increased churn intent, indicating a high churn risk during the early stages of a customer's life cycle.

---

### 3️⃣ Model Development — Baseline Models
- Trained two baseline classifiers, a **Logistic Regression** and a **Random Forest**, to predict customer churn.
- **Top Predictive Features:**
  1. `Tenure`
  2. `Contract_Month-to-month`
  3. `TotalCharges`
  4. `InternetService_Fiber optic`
- **Performance:**
  - The **Logistic Regression** model achieved a perfect **Recall of 1.00**, indicating it was highly effective at identifying all churned customers.

---

### 4️⃣ Key Insights & Recommendations

#### **1. Prioritize Month-to-Month Contracts**
- The **Contract Type** is the most important predictor of churn.
- **Action:** Offer incentives to customers on month-to-month contracts to encourage them to switch to one- or two-year plans.

#### **2. Focus on Early-Stage Customer Experience**
- Short tenure is a strong indicator of churn.
- **Action:** Implement a targeted welcome and engagement program for new customers, particularly within the first six months, to improve their initial experience and build loyalty.

#### **3. Reduce Dissatisfaction with Fiber Optic Service**
- Customers with Fiber optic service have a high churn rate despite being a high-value segment.
- **Action:** Investigate and resolve service quality issues for this segment to reduce dissatisfaction and prevent churn.

#### **4. Promote Service Add-ons**
- The absence of add-ons like Online Security and Tech Support is correlated with a higher churn risk.
- **Action:** Offer these add-ons as part of a high-value bundle to at-risk customers to increase service stickiness.

---

## 📊 Technologies Used
- **Python** (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)
- **Jupyter Notebook** for EDA and modeling
- **`class_weight='balanced'`** for handling class imbalance

---

## 📌 Dataset Reference
- The dataset used for this project is the **IBM Telco Customer Churn Dataset**.