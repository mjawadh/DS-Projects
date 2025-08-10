# Online Shoppers Purchase Intentions

**Objective:**  
This project aims to analyze **user browsing behavior** to predict whether a user will complete a purchase during an online shopping session.  
We combined **detailed Exploratory Data Analysis (EDA)** with a **powerful XGBoost model** to uncover insights and build an accurate predictive system.

---

## Project Workflow

### 1️⃣ Data Cleaning & Preprocessing
- **Loaded Dataset:** Containing 18 behavioral features and a target variable **`Revenue`**.
- **Encoding:** Converted categorical features (`Month`, `VisitorType`, `Weekend`, etc.) into numeric form.
- **Scaling:** Applied **`StandardScaler`** to numerical features for optimal model performance.

---

### 2️⃣ Exploratory Data Analysis (EDA)
- Sessions with purchases had **higher total pages viewed**, especially product-related pages.
- **Returning Visitors** were most common and had higher purchase likelihood.
- **`PageValues`** was a strong predictor — higher values correlated with increased purchase intent.

---

### 3️⃣ Model Development — XGBoost
- Trained an **XGBoost Classifier** to predict purchase intent.
- Optimized hyperparameters using **`GridSearchCV`**.
- **Top Predictive Features:**
  1. `PageValues`
  2. Month of **November**
  3. `BounceRates`
  4. `ExitRates`
- **Performance:**
  - Achieved high **Accuracy** and strong **ROC-AUC**, indicating excellent discrimination between purchasing and non-purchasing sessions.

---

### 4️⃣ Key Insights & Recommendations

#### **1. Prioritize High-Value Pages**
- `PageValues` is the **most important predictor** of purchase intent.  
- **Action:** Optimize content, design, and UX for these pages to encourage conversions.

#### **2. Focus on Returning Visitors**
- Returning visitors are more likely to buy.  
- **Action:** Use personalized marketing, targeted offers, and a smoother checkout experience.

#### **3. Reduce Bounce & Exit Rates**
- High bounce/exit rates signal lost sales.  
- **Action:** Improve navigation, page speed, and simplify UI on high-exit pages.

#### **4. Leverage Seasonal Trends**
- November shows increased purchase intent (holiday season).  
- **Action:** Plan promotions, ads, and inventory for this period.

---

## 📊 Technologies Used
- **Python** (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost)
- **Jupyter Notebook** for EDA and modeling
- **GridSearchCV** for hyperparameter tuning

---

## 📌 Dataset Reference
- The dataset used for this project is publicly available on the **UCI Machine Learning Repository**:  
  [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/ml/datasets/online+shoppers+purchasing+intention+dataset)


