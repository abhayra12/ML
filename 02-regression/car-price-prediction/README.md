# Car Price Prediction Project

A comprehensive machine learning project to predict car prices using linear regression with regularization. This project demonstrates the complete workflow of a regression problem, from data exploration to model evaluation.

## 📋 Project Overview

This project implements a **car price prediction model** using linear regression techniques. The goal is to predict the MSRP (Manufacturer's Suggested Retail Price) of cars based on various features such as engine specifications, fuel efficiency, vehicle characteristics, and more.

## 🎯 Learning Objectives

This project is designed to help you understand:

- **Data Preparation**: Cleaning and standardizing data for machine learning
- **Exploratory Data Analysis (EDA)**: Understanding data distributions and patterns
- **Feature Engineering**: Creating new features to improve model performance
- **Linear Regression**: Implementing regression from scratch using the Normal Equation
- **Regularization**: Using Ridge regression to prevent overfitting
- **Model Evaluation**: Using RMSE and validation frameworks to assess performance

## 📊 Dataset

The dataset contains information about various car models with the following features:

- **Numerical Features**: engine_hp, engine_cylinders, highway_mpg, city_mpg, popularity, year
- **Categorical Features**: make, model, number_of_doors, engine_fuel_type, transmission_type, driven_wheels, market_category, vehicle_size, vehicle_style
- **Target Variable**: msrp (car price)

## 🔧 Project Structure

```
car-price-prediction/
│
├── 02-carprice.ipynb    # Main notebook with detailed explanations
├── data.csv             # Dataset
└── README.md            # Project documentation
```

## 🚀 Key Steps

### 1. Data Preparation
- Load and inspect the dataset
- Standardize column names and categorical values
- Handle missing values

### 2. Exploratory Data Analysis
- Visualize price distribution
- Apply log transformation to handle skewed data
- Analyze feature relationships

### 3. Validation Framework
- Split data into train (60%), validation (20%), and test (20%) sets
- Ensure reproducibility with random seed

### 4. Baseline Model
- Start with basic numerical features
- Implement linear regression using the Normal Equation
- Evaluate using RMSE (Root Mean Squared Error)

### 5. Feature Engineering
- Create age feature from year
- Apply one-hot encoding for categorical variables
- Iteratively add features to improve performance

### 6. Regularization
- Implement Ridge regression to prevent overfitting
- Tune regularization parameter (alpha)
- Balance model complexity and generalization

### 7. Final Evaluation
- Test the model on unseen test data
- Compare performance across train, validation, and test sets

## 📈 Results

The model achieves competitive performance in predicting car prices:

- **Best regularization parameter**: r = 0.01
- **Features used**: 40+ features including numerical and one-hot encoded categorical variables
- **Evaluation metric**: RMSE on log-transformed prices

## 🛠️ Technologies Used

- **Python 3.x**
- **NumPy**: For numerical operations and linear algebra
- **Pandas**: For data manipulation and analysis
- **Matplotlib & Seaborn**: For data visualization
- **Jupyter Notebook**: For interactive development

## 📝 Key Concepts Explained

### Linear Regression
The model uses the Normal Equation to find optimal weights:
```
w = (X^T * X)^-1 * X^T * y
```

### Ridge Regularization
Adds a penalty term to prevent overfitting:
```
w = (X^T * X + r*I)^-1 * X^T * y
```
where `r` is the regularization parameter.

### RMSE (Root Mean Squared Error)
Measures the average prediction error:
```
RMSE = sqrt(mean((y_pred - y_actual)^2))
```

## 🎓 How to Use This Project

1. **Clone or download** the repository
2. **Install dependencies**:
   ```bash
   pip install pandas numpy matplotlib seaborn jupyter
   ```
3. **Open the notebook**:
   ```bash
   jupyter notebook 02-carprice.ipynb
   ```
4. **Run cells sequentially** to see the complete workflow
5. **Experiment** with different features and parameters

## 💡 Learning Tips

- Read the comments in each code cell carefully
- Pay attention to the markdown explanations between cells
- Try modifying features to see how they affect performance
- Experiment with different regularization parameters
- Compare results with and without feature engineering

## 🔍 Key Takeaways

1. **Log transformation** helps normalize skewed price distributions
2. **Feature engineering** significantly improves model performance
3. **Regularization** is essential when using many features
4. **Validation framework** prevents overfitting and helps in hyperparameter tuning
5. **RMSE** provides an interpretable measure of prediction accuracy

## 📚 Further Reading

- [Understanding the Bias-Variance Tradeoff](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff)
- [Ridge Regression](https://en.wikipedia.org/wiki/Ridge_regression)
- [Feature Engineering Techniques](https://en.wikipedia.org/wiki/Feature_engineering)

## 👤 Author

This project is part of a machine learning learning journey focused on understanding regression techniques from first principles.

## 📄 License

This project is open source and available for educational purposes.

---

**Happy Learning! 🚗💰📊**