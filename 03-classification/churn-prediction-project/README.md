# Customer Churn Prediction Project

A comprehensive machine learning project to predict customer churn using logistic regression. This project demonstrates the complete workflow of a binary classification problem, from feature importance analysis to model deployment.

## 📋 Project Overview

This project implements a **customer churn prediction model** for a telecommunications company. The goal is to predict whether a customer will churn (leave the service) based on their demographics, account information, and service usage patterns.

## 🎯 Learning Objectives

This project is designed to help you understand:

- **Binary Classification**: Predicting categorical outcomes (churn vs. no churn)
- **Feature Importance Analysis**: Using risk ratios, mutual information, and correlation
- **One-Hot Encoding**: Converting categorical features for machine learning
- **Logistic Regression**: Understanding the sigmoid function and probability predictions
- **Model Interpretation**: Understanding feature weights and their impact
- **Model Evaluation**: Using accuracy and probability thresholds

## 📊 Dataset

**Source**: [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)

The dataset contains customer information with the following features:

### Numerical Features
- `tenure`: Number of months the customer has stayed
- `monthlycharges`: Monthly subscription charges
- `totalcharges`: Total amount charged to the customer

### Categorical Features
- **Demographics**: gender, seniorcitizen, partner, dependents
- **Services**: phoneservice, multiplelines, internetservice, onlinesecurity, onlinebackup, deviceprotection, techsupport, streamingtv, streamingmovies
- **Account**: contract, paperlessbilling, paymentmethod

### Target Variable
- `churn`: Whether the customer churned (Yes/No)

## 🔧 Project Structure

```
churn-prediction-project/
│
├── notebook.ipynb       # Main notebook with detailed explanations
├── data-week-3.csv      # Dataset (downloaded during execution)
└── README.md            # Project documentation
```

## 🚀 Key Steps

### 1. Data Preparation
- Download and load the dataset
- Standardize column names and categorical values
- Convert categorical target to binary (0/1)
- Handle missing values in numerical features

### 2. Validation Framework
- Split data into train (60%), validation (20%), and test (20%) sets
- Use Scikit-Learn's `train_test_split` for consistency
- Separate features (X) from target variable (y)

### 3. Exploratory Data Analysis (EDA)
- Analyze churn rate (~27% in the dataset)
- Check for missing values
- Examine distributions of numerical and categorical features

### 4. Feature Importance Analysis

#### Churn Rate & Risk Ratio
- Calculate churn rate for each category
- Compare to global churn rate
- Identify high-risk customer segments

#### Mutual Information
- Measure dependency between features and churn
- Rank features by their predictive power
- Top features: contract type, online security, tech support

#### Correlation Analysis
- Analyze numerical features' relationship with churn
- Findings: 
  - Negative correlation with tenure (longer customers stay less likely to churn)
  - Positive correlation with monthly charges (higher charges = higher churn)

### 5. One-Hot Encoding
- Convert categorical features to numerical format
- Use `DictVectorizer` from Scikit-Learn
- Create binary columns for each category

### 6. Logistic Regression

#### Model Training
- Implement using Scikit-Learn's `LogisticRegression`
- Use sigmoid function to output probabilities
- Train on encoded features

#### Model Interpretation
- Analyze feature coefficients (weights)
- Positive weights → increase churn probability
- Negative weights → decrease churn probability
- Train smaller model with key features for easier interpretation

### 7. Model Evaluation
- Use 0.5 probability threshold for binary predictions
- Calculate accuracy on validation set
- Final evaluation on test set

### 8. Model Deployment
- Train final model on full training data
- Make predictions on individual customers
- Use probability scores for risk assessment

## 📈 Results

The model achieves strong performance in predicting customer churn:

- **Accuracy**: ~80% on test set
- **Key predictive features**: Contract type, tenure, monthly charges, online security, tech support
- **Model type**: Logistic Regression with L-BFGS solver

### Key Insights

1. **Contract type is the strongest predictor**:
   - Month-to-month contracts have highest churn risk
   - Two-year contracts have lowest churn risk

2. **Tenure matters**:
   - New customers (≤2 months) have very high churn rates
   - Long-term customers (>12 months) are much less likely to churn

3. **Service features impact churn**:
   - Customers without online security/tech support are more likely to churn
   - Internet service type affects churn probability

## 🛠️ Technologies Used

- **Python 3.x**
- **NumPy**: For numerical operations
- **Pandas**: For data manipulation and analysis
- **Matplotlib**: For data visualization
- **Scikit-Learn**: For model training and evaluation
  - `train_test_split`: Data splitting
  - `DictVectorizer`: One-hot encoding
  - `LogisticRegression`: Classification model
  - `mutual_info_score`: Feature importance
- **Jupyter Notebook**: For interactive development

## 📝 Key Concepts Explained

### Logistic Regression
Uses the sigmoid function to convert linear outputs to probabilities:
```
probability = 1 / (1 + e^(-z))
where z = w0 + w1*x1 + w2*x2 + ... + wn*xn
```

### One-Hot Encoding
Converts categorical features to binary columns:
```
gender: [male, female] → gender_male: [0,1], gender_female: [1,0]
```

### Risk Ratio
Measures relative churn risk for a group:
```
Risk Ratio = (Group Churn Rate) / (Global Churn Rate)
```
- Risk Ratio > 1: Higher risk than average
- Risk Ratio < 1: Lower risk than average

### Mutual Information
Measures how much information one variable provides about another:
- Higher values = stronger relationship
- 0 = no relationship

## 🎓 How to Use This Project

1. **Clone or download** the repository
2. **Install dependencies**:
   ```bash
   pip install pandas numpy matplotlib scikit-learn jupyter
   ```
3. **Open the notebook**:
   ```bash
   jupyter notebook notebook.ipynb
   ```
4. **Run cells sequentially** to:
   - Download the dataset automatically
   - See complete analysis and explanations
   - Train and evaluate the model

## 💡 Learning Tips

- Study the feature importance section carefully to understand data analysis
- Compare different features' risk ratios to identify patterns
- Experiment with different probability thresholds (not just 0.5)
- Try training models with different feature subsets
- Observe how model coefficients relate to feature importance

## 🔍 Key Takeaways

1. **Feature importance analysis** is crucial before modeling
2. **Contract type** is the most important predictor of churn
3. **Tenure** has strong negative correlation with churn
4. **One-hot encoding** is essential for categorical features
5. **Logistic regression** provides interpretable probability predictions
6. **Model coefficients** can be interpreted for business insights
7. **Risk ratios** help identify high-risk customer segments

## 🎯 Business Applications

This model can be used to:
- **Identify at-risk customers** for proactive retention efforts
- **Prioritize retention campaigns** based on churn probability
- **Understand churn drivers** to improve service offerings
- **Segment customers** by risk level
- **Calculate customer lifetime value** adjustments

## 📚 Further Exploration

### Suggested Improvements
- Try different classification algorithms (Random Forest, XGBoost)
- Implement cross-validation for robust evaluation
- Use precision, recall, and F1-score for imbalanced data
- Create ROC curves and analyze AUC
- Implement feature selection techniques

### Related Projects
- **Lead Scoring**: [Kaggle Dataset](https://www.kaggle.com/ashydv/leads-dataset)
- **Credit Card Default**: [UCI Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

## 👤 Author

This project is part of a machine learning learning journey focused on understanding classification techniques and their business applications.

## 📄 License

This project is open source and available for educational purposes.

---

**Happy Learning! 📞📊🎯**
