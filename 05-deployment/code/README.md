# ML Model Deployment: Churn Prediction Web Service

This folder contains everything needed to train a machine learning model and deploy it as a production-ready web service.

## 📋 Project Overview

**Objective**: Deploy a churn prediction model that can:
- Accept customer data via HTTP API
- Generate real-time churn probability predictions
- Return predictions in JSON format
- Scale to handle production workloads

**Use Case**: A telecom company wants to identify at-risk customers and send targeted retention campaigns.

**Model**: Logistic Regression trained on customer behavior data to predict probability of churn (leaving the company).

---

## 📁 Folder Structure

```
code/
├── 05-train-churn-model.ipynb    # Jupyter notebook with training pipeline
├── train.py                       # Standalone training script
├── predict.py                     # Flask web service with prediction API
├── predict-test.py                # Test script to validate predictions
├── ping.py                        # Health check endpoint
├── Dockerfile                     # Docker configuration for containerization
├── Pipfile                        # Python dependency management
├── plan.md                        # Project planning document
└── README.md                      # This file
```

---

## 🚀 Quick Start

### Option 1: Using Jupyter Notebook (Development)

```bash
# Install dependencies
pip install jupyter pandas numpy scikit-learn

# Start Jupyter and open 05-train-churn-model.ipynb
jupyter notebook 05-train-churn-model.ipynb
```

**What the notebook does:**
1. Loads churn dataset
2. Prepares and preprocesses data
3. Trains logistic regression model
4. Performs 5-fold cross-validation
5. Evaluates on test set
6. Saves trained model to disk (pickle format)
7. Demonstrates loading saved model
8. Shows how to make predictions

**Output**: `model_C=1.0.bin` - Serialized model and vectorizer

---

### Option 2: Using Python Scripts (Production)

#### Step 1: Train the Model

```bash
# Install dependencies
pip install -r requirements.txt  # or use: pipenv install

# Run training script
python train.py
```

This creates `model_C=1.0.bin` containing:
- **DictVectorizer**: Converts categorical/numerical features to numeric vectors
- **LogisticRegression**: The trained classification model

#### Step 2: Start the Web Service

```bash
# Start Flask web service
python predict.py
```

Service runs on `http://localhost:9696`

#### Step 3: Test the Service

```bash
# In another terminal, run tests
python predict-test.py
```

---

## 📊 Data Pipeline

### Input Data Format

The model expects customer data as JSON:

```json
{
  "gender": "female",
  "seniorcitizen": 0,
  "partner": "yes",
  "dependents": "no",
  "phoneservice": "no",
  "multiplelines": "no_phone_service",
  "internetservice": "dsl",
  "onlinesecurity": "no",
  "onlinebackup": "yes",
  "deviceprotection": "no",
  "techsupport": "no",
  "streamingtv": "no",
  "streamingmovies": "no",
  "contract": "month-to-month",
  "paperlessbilling": "yes",
  "paymentmethod": "electronic_check",
  "tenure": 1,
  "monthlycharges": 29.85,
  "totalcharges": 29.85
}
```

### Features

**Categorical Features (16):**
- Demographics: `gender`, `seniorcitizen`, `partner`, `dependents`
- Services: `phoneservice`, `multiplelines`, `internetservice`
- Add-ons: `onlinesecurity`, `onlinebackup`, `deviceprotection`, `techsupport`, `streamingtv`, `streamingmovies`
- Account: `contract`, `paperlessbilling`, `paymentmethod`

**Numerical Features (3):**
- `tenure` - Months as a customer
- `monthlycharges` - Monthly bill amount
- `totalcharges` - Total charges since joining

### Target Variable

**`churn`** - Binary classification
- `0` = Customer stays (no churn)
- `1` = Customer leaves (churn)

---

## 🔧 API Reference

### Prediction Endpoint

**Endpoint**: `POST /predict`

**Request**:
```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "female",
    "seniorcitizen": 0,
    ...other features...
  }'
```

**Response** (Success - 200):
```json
{
  "churn": true,
  "churn_probability": 0.847
}
```

**Response** (Error - 400):
```json
{
  "error": "Missing required fields"
}
```

### Health Check Endpoint

**Endpoint**: `GET /ping`

**Response**:
```json
{
  "status": "ok"
}
```

---

## 🧠 Model Details

### Algorithm: Logistic Regression

**Why Logistic Regression?**
- ✅ Interpretable - Know which features matter
- ✅ Fast - Makes predictions in milliseconds
- ✅ Probabilistic - Returns probability (0-1), not just yes/no
- ✅ Proven - Works well for binary classification

**Key Parameters:**
- `C=1.0` - Regularization strength (inverse)
  - Smaller C = stronger regularization (simpler model)
  - Larger C = weaker regularization (more complex)
  - Selected via 5-fold cross-validation

### Performance Metrics

**Cross-Validation (5-Fold):**
- Mean ROC-AUC: ~0.84
- Standard Deviation: ~0.02
- Interpretation: Model ranks random positive 84% higher than random negative

**Test Set:**
- Final ROC-AUC: Evaluated on held-out test data

**Metrics Explained:**
- **Accuracy**: % of correct predictions (misleading with imbalanced data)
- **Precision**: Of predicted churns, how many actually churned?
- **Recall**: Of actual churns, how many did we catch?
- **ROC-AUC**: Area under Receiver Operating Characteristic curve
  - 0.5 = Random guessing
  - 1.0 = Perfect predictions
  - 0.84 = Good model

---

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t churn-predictor .
```

### Run Container

```bash
docker run -p 9696:9696 churn-predictor
```

Service accessible at: `http://localhost:9696/predict`

### Push to Cloud

```bash
# Docker Hub
docker tag churn-predictor username/churn-predictor:v1
docker push username/churn-predictor:v1

# Cloud services support: AWS ECR, Google Container Registry, Azure
```

---

## 📚 Training Pipeline Explained

### Step 1: Data Loading & Preprocessing

```python
# Load data
df = pd.read_csv('data-week-3.csv')

# Standardize column names
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Handle missing values
df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

# Convert target to binary
df.churn = (df.churn == 'yes').astype(int)
```

### Step 2: Train/Test Split

```python
# 80% for training, 20% for final testing
df_train, df_test = train_test_split(df, test_size=0.2)
```

### Step 3: Feature Engineering

```python
# DictVectorizer handles:
# - Categorical encoding (one-hot encoding)
# - Standardization
# - Missing value imputation
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(df_train.to_dict(orient='records'))
```

### Step 4: Model Training

```python
# Logistic regression with L2 regularization
model = LogisticRegression(C=1.0, max_iter=1000)
model.fit(X_train, y_train)
```

### Step 5: Cross-Validation

```python
# 5-fold CV to estimate true performance
# Prevents overfitting estimate from single train/test split
kfold = KFold(n_splits=5, shuffle=True)

for train_idx, val_idx in kfold.split(df_train):
    model = train(df_train.iloc[train_idx], C=1.0)
    predictions = predict(df_train.iloc[val_idx])
    score = roc_auc_score(y_val, predictions)
```

### Step 6: Model Persistence

```python
# Save to disk for later use (deployment)
with open('model_C=1.0.bin', 'wb') as f:
    pickle.dump((dv, model), f)
```

### Step 7: Making Predictions

```python
# Load saved model
with open('model_C=1.0.bin', 'rb') as f:
    dv, model = pickle.load(f)

# Transform new customer data
X = dv.transform([customer_dict])

# Get prediction probability
prob = model.predict_proba(X)[0, 1]  # Probability of churn
```

---

## 🔄 Workflow Example

### Development Workflow

```
1. Explore Data
   ↓
2. Train Model (Notebook)
   ↓
3. Evaluate Performance
   ↓
4. Save Model
   ↓
5. Version Control (Git)
   ↓
6. Review & Approval
   ↓
7. Deploy to Production
```

### Prediction Request Workflow

```
Customer CRM System
  ↓
HTTP POST /predict
  ↓
Load Model from Disk
  ↓
Transform Features (DictVectorizer)
  ↓
Generate Prediction (LogisticRegression)
  ↓
Return Probability (0-1)
  ↓
CRM System Receives Result
  ↓
If churn > 0.7: Send Retention Email
```

---

## 🛠️ Troubleshooting

### Issue: Model File Not Found

```bash
# Solution: Run training first
python train.py
```

### Issue: Port 9696 Already in Use

```bash
# Solution: Kill existing process or use different port
# Linux/Mac:
lsof -i :9696
kill -9 <PID>

# Or modify port in predict.py:
app.run(debug=True, host='0.0.0.0', port=9697)
```

### Issue: Missing Dependencies

```bash
# Solution: Install all requirements
pip install pandas numpy scikit-learn flask requests
# Or use:
pipenv install
```

### Issue: Different Python Versions

```bash
# Use virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

---

## 📈 Performance Monitoring

### Key Metrics to Track in Production

1. **Model Performance**
   - ROC-AUC on new data
   - Precision/Recall
   - Prediction distribution

2. **Service Health**
   - Request latency (should be <100ms)
   - Error rate
   - API availability

3. **Data Quality**
   - Missing values
   - Outliers
   - Distribution shift

### Model Retraining Triggers

Retrain model when:
- ROC-AUC drops >5% from baseline
- Data distribution significantly changes
- New features become available
- Business requirements change

---

## 🚦 Next Steps

1. **Test locally** - Run `predict-test.py` to verify predictions
2. **Deploy to cloud** - Use Docker for containerization
3. **Set up monitoring** - Track prediction quality and latency
4. **Implement logging** - Log all predictions for audit trail
5. **Schedule retraining** - Automate monthly model updates
6. **Create dashboards** - Visualize predictions and performance
7. **Document API** - Create Swagger/OpenAPI documentation

---

## 📦 Dependencies

### Core Libraries

```
pandas>=1.3.0          # Data manipulation
numpy>=1.21.0          # Numerical computing
scikit-learn>=1.0.0    # Machine learning
```

### Deployment

```
flask>=2.0.0           # Web framework
requests>=2.26.0       # HTTP client testing
```

### Optional

```
jupyter>=1.0.0         # Interactive notebooks
gunicorn>=20.1.0       # Production WSGI server
```

### Install All

```bash
pip install pandas numpy scikit-learn flask requests jupyter gunicorn
```

---

## 📝 Files Explained

### `05-train-churn-model.ipynb`
Interactive Jupyter notebook demonstrating:
- Complete training pipeline
- Model serialization
- Loading and prediction
- API testing with requests

**Best for:** Learning and experimentation

### `train.py`
Standalone Python script to:
- Load data
- Train model
- Perform cross-validation
- Save model to disk

**Best for:** Automation and scheduled retraining

### `predict.py`
Flask web application with:
- `/predict` endpoint (POST) - Makes predictions
- `/ping` endpoint (GET) - Health check

**Best for:** Production deployment

### `predict-test.py`
Test script to:
- Load saved model
- Test predictions locally
- Verify API responses

**Best for:** Quality assurance

### `Dockerfile`
Container configuration:
- Python 3.9 base image
- Install dependencies
- Copy application
- Expose port 9696

**Best for:** Cloud deployment

---

## 🔐 Security Considerations

1. **API Validation** - Validate all input data
2. **Error Handling** - Don't expose internal errors
3. **Rate Limiting** - Prevent abuse
4. **Authentication** - Add API keys if needed
5. **HTTPS** - Use SSL/TLS in production
6. **Model Integrity** - Verify model file checksums
7. **Logging** - Audit all predictions

---

## 📊 Model Output Interpretation

### Churn Probability

```
Prediction < 0.3  → Low risk (likely to stay)
              ✅ Send regular engagement emails

Prediction 0.3-0.7 → Medium risk (uncertain)
              ⚠️  Send targeted offers

Prediction > 0.7  → High risk (likely to churn)
              🚨 Immediate retention action needed
```

### Business Actions

Based on prediction probability:
- **0.0-0.3**: Regular customer - routine communication
- **0.3-0.5**: At-risk customer - special offers
- **0.5-0.7**: Very at-risk - executive outreach
- **0.7-1.0**: Critical risk - immediate intervention

---

## 🎓 Learning Resources

**Topics Covered:**
- Machine Learning workflow
- Model training and evaluation
- Model serialization (pickle)
- Web service deployment
- RESTful API design
- Docker containerization
- Production ML systems

**Concepts:**
- Logistic Regression
- Cross-validation
- Feature engineering
- Binary classification
- Probability calibration

---

## 💼 Real-World Applications

This deployment architecture is used by:
- **Telecom Companies** - Predict customer churn
- **Banks** - Credit risk assessment
- **E-commerce** - Customer lifetime value
- **Subscription Services** - Retention prediction
- **Insurance** - Claim probability
- **Healthcare** - Patient risk stratification

---

## 📞 Support & Questions

For questions about:
- **Model Training** - See `05-train-churn-model.ipynb`
- **API Usage** - Check `predict-test.py` for examples
- **Deployment** - Review `Dockerfile` and `predict.py`
- **Data Format** - Check sample customer dict in notebook

---

## 📜 License

MIT License - Feel free to use and modify

---

## 🎯 Summary

This project demonstrates a complete ML workflow:

```
Raw Data → Exploration → Preprocessing → Training → Evaluation
   ↓                                          ↓
   └─────────────────────────────────────────┘
                     ↓
           Model Serialization
                     ↓
           Web Service Deployment
                     ↓
           Real-time Predictions
                     ↓
           Business Decisions
```

**Key Takeaway**: From data to production-ready prediction service in one cohesive project!
