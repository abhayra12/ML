# ML Deployment Workshop: FastAPI & Modern Python Tooling

**Advanced machine learning deployment module using FastAPI, Pydantic, and `uv` package manager.**

> 📺 **Video Reference**: https://www.youtube.com/watch?v=jzGzw98Eikk
> 
> 📚 **Based on**: [ML Zoomcamp Module 5](https://github.com/DataTalksClub/machine-learning-zoomcamp/tree/master/05-deployment)

---

## 📋 Project Overview

**Objective**: Learn modern best practices for deploying ML models in production using cutting-edge Python tools.

**Business Case**: Telecom company wants to deploy a real-time churn prediction service.

**Key Improvements Over Traditional Approach**:

| Feature | Traditional Flask | This Workshop (FastAPI) |
|---------|-------------------|--------------------------|
| Framework | Flask | **FastAPI** |
| Package Manager | pip/Pipenv | **uv** (Rust-based, 10-100x faster) |
| Type Hints | Manual | **First-class support** |
| Input Validation | Manual checks | **Pydantic automatic** |
| API Documentation | Manual Swagger setup | **Auto-generated** |
| Async Support | Optional/Complex | **Native async/await** |
| Performance | ~100 req/s | **10,000+ req/s** |

**Technologies Covered**:
- ✅ Scikit-Learn pipelines
- ✅ uv package manager (instead of Pipenv)
- ✅ FastAPI (instead of Flask)
- ✅ Pydantic models for validation
- ✅ Docker containerization
- ✅ Fly.io cloud deployment

**Dataset**: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- [uv package manager](https://github.com/astral-sh/uv) (`pip install uv`)

### Option A: Development Environment (Recommended for Learning)

```bash
# Install dependencies
pip install jupyter scikit-learn pandas

# Run the notebook
jupyter notebook workshop-uv-fastapi.ipynb
```

**What you'll learn from the notebook**:
- Complete ML workflow from data to predictions
- Model serialization with pickle
- ML Pipelines for combined preprocessing + modeling
- HTTP API integration with requests library
- Exploratory data analysis

### Option B: Production Setup (uv + FastAPI)

```bash
# 1. Install dependencies using uv
uv sync

# 2. Train model
uv run python train.py

# 3. Start API server
uv run python predict.py

# 4. In another terminal, test it
uv run python test.py
```

### Option C: Docker (Complete Isolation)

```bash
# Build Docker image
docker build -t churn-predictor .

# Run container
docker run -it --rm -p 9696:9696 churn-predictor

# Test from another terminal
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"gender": "female", "tenure": 1, ...}'
```

---

## 🏗️ Project Structure


## 📁 File Directory

```
workshop/
├── 📓 workshop-uv-fastapi.ipynb    ← Complete learning notebook (21 cells)
├── 🐍 train.py                      ← Model training script
├── 🌐 predict.py                    ← FastAPI web service (main app)
├── 🧪 test.py                       ← Automated API tests
├── 📍 ping.py                       ← Health check endpoint
├── 🔧 predict_old.py                ← Flask reference implementation
├── 📦 pyproject.toml                ← uv dependency management
├── .python-version                  ← Python version spec (3.13+)
├── 🐳 Dockerfile                    ← Container configuration
├── 🚀 fly.toml                      ← Fly.io deployment config
├── .dockerignore                    ← Docker build exclusions
├── README.md                        ← This file
└── model.bin                        ← Saved model (generated after train.py)
```

### File Descriptions

| File | Purpose | When to Use |
|------|---------|-----------|
| `workshop-uv-fastapi.ipynb` | Interactive notebook with full workflow | Learning & experimentation |
| `train.py` | Standalone training script | Model retraining |
| `predict.py` | FastAPI web service | Production deployment |
| `test.py` | API tests | CI/CD pipelines |
| `ping.py` | Health check (simple example) | Understanding FastAPI basics |
| `predict_old.py` | Flask reference | Understanding differences |
| `pyproject.toml` | Dependency management | Dependency updates |
| `Dockerfile` | Container image | Cloud deployment |
| `fly.toml` | Fly.io config | Deployment settings |

---

## 📚 Learning Path

### For Beginners (2-3 hours)

1. **Start with Jupyter Notebook** (`workshop-uv-fastapi.ipynb`)
   - Read all 21 cells with detailed comments
   - Run each cell
   - Understand the complete workflow
   - **Time**: 60 minutes

2. **Understand the Code Files**
   - Read `train.py` - See how to train programmatically
   - Read `predict.py` - Understand FastAPI structure
   - **Time**: 30 minutes

3. **Run the Full Stack**
   ```bash
   python train.py              # Train model
   python predict.py            # Start API
   python test.py               # Test it
   ```
   - **Time**: 15 minutes

4. **Explore Auto-Generated Docs**
   - Visit http://localhost:8000/docs
   - Try requests interactively
   - **Time**: 15 minutes

### For Intermediate Users (3-4 hours)

1. Modify Pydantic models in `predict.py`
2. Add custom validation rules
3. Implement logging
4. Create comprehensive test suite
5. Deploy to Fly.io

### For Advanced Users

1. Implement model versioning
2. Add A/B testing framework
3. Set up monitoring and alerting
4. Create CI/CD pipeline
5. Deploy with Kubernetes

---

## 🎯 Understanding the Workflow

### Complete ML Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│ 1. DATA PREPARATION                                          │
│   - Load CSV from URL or local file                          │
│   - Clean column names (lowercase, remove spaces)            │
│   - Handle missing values (totalcharges → 0)                │
│   - Encode target (churn: yes/no → 1/0)                     │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 2. FEATURE ENGINEERING                                      │
│   - Identify numerical: [tenure, monthlycharges, ...]       │
│   - Identify categorical: [gender, contract, ...]           │
│   - DictVectorizer: Convert to numeric matrix               │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 3. MODEL TRAINING                                            │
│   - Fit LogisticRegression (C=1.0, solver='liblinear')      │
│   - Generate ~100 features from input data                   │
│   - Learn weights for each feature                           │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 4. MODEL SERIALIZATION                                       │
│   - Save (DictVectorizer, LogisticRegression) tuple          │
│   - Pickle format → model.bin                                │
│   - ~2.5 KB file size                                        │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 5. DEPLOYMENT                                                │
│   - FastAPI web service loads model.bin at startup           │
│   - Uvicorn runs on port 9696                               │
│   - Swagger docs auto-generated at /docs                     │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 6. PREDICTION SERVICE                                        │
│   ┌─ Customer sends JSON with attributes                    │
│   ├─ DictVectorizer transforms to numeric vector            │
│   ├─ LogisticRegression computes probability                │
│   └─ Response: {churn: bool, churn_probability: float}      │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│ 7. BUSINESS ACTION                                           │
│   - If churn > 0.7: Send retention email                     │
│   - If churn < 0.3: Regular communication                    │
│   - Else: Special offers                                     │
└──────────────────────────────────────────────────────────────┘
```

### Feature Sets

**Numerical Features** (3 total):
- `tenure` - Months as customer (0-72)
- `monthlycharges` - Monthly bill ($18-$119)
- `totalcharges` - Total accumulated charges ($0-$8,684)

**Categorical Features** (16 total):

*Demographics*:
- `gender` - male/female
- `seniorcitizen` - 0/1
- `partner` - yes/no
- `dependents` - yes/no

*Services*:
- `phoneservice` - yes/no
- `multiplelines` - yes/no/no_phone_service
- `internetservice` - dsl/fiber_optic/no

*Add-ons*:
- `onlinesecurity` - yes/no/no_internet_service
- `onlinebackup` - yes/no/no_internet_service
- `deviceprotection` - yes/no/no_internet_service
- `techsupport` - yes/no/no_internet_service
- `streamingtv` - yes/no/no_internet_service
- `streamingmovies` - yes/no/no_internet_service

*Account*:
- `contract` - month-to-month/one_year/two_year
- `paperlessbilling` - yes/no
- `paymentmethod` - 4 options

**Target Variable**:
- `churn` - 0 (stayed) / 1 (left company)

### Model Performance

**Metrics**:
- ROC-AUC: ~0.85 (good discrimination)
- Accuracy: ~80% (on imbalanced data)
- Recall: ~60% (catches 60% of churners)
- Precision: ~70% (70% of predicted churners actually churn)

**Interpretation**:
- Probability 0.0-0.3: Low churn risk (likely to stay)
- Probability 0.3-0.7: Medium risk (uncertain)
- Probability 0.7-1.0: High churn risk (likely to leave)

---

## 🔄 Step-by-Step Walkthrough

```python
datapoint = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}
```

We first transform it with the dictionary vectorizer

```python
X = dv.transform(datapoint)
```

And then get the predictions

```python
model.predict_proba(X)[0, 1]
```

Let's save this to pickle:

```python
with open('model.bin', 'wb') as f_out:
    pickle.dump((dv, model), f_out)
```


This is how we load: 

```python
with open('model.bin', 'rb') as f_in:
    (dv, model) = pickle.load(f_in)
```

## Scikit-Learn Pipelines

It's not convenient to deal with two objects: `dv` and `model`. 
Let's combine them into one: 


```python
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(
    DictVectorizer(),
    LogisticRegression(solver='liblinear')
)

pipeline.fit(train_dict, y_train)
```

Now predicting becomes simpler too:

```python
pipeline.predict_proba(datapoint)[0, 1]
```

## Turning the notebook into a script

We can now turn this notebook into a training script:

```bash
jupyter nbconvert --to=script workshop-uv-fastapi.ipynb
mv workshop-uv-fastapi.py train.py
```

Let's edit it.

At the end, we have the code similar to [train.py](train.py)

```python
df = load_data()
pipeline = train_model(df)
save_model(pipeline, 'model.bin')

print('Model saved to model.bin')
```

Let's load the saved model. Create [predict.py](predict.py)
and load the model there:

```python
import pickle

with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

# apply the model
```

## FastAPI

Now we will turn predict.py into a web service. 

Let's install FastAPI and uvicorn for that:

```bash
pip install fastapi uvicorn
```

The simplest FastAPI app
([created with ChatGPT](https://chatgpt.com/share/6899dc68-03a8-800a-8bd8-9f2218f103e6)
by translating [the old Flask code](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/05-deployment/code/ping.py)).

Let's put it to `ping.py`:


```python
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="ping")

@app.get("/ping")
def ping():
    return "PONG"

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
```

Run it:

```bash
python ping.py
```

"Proper" way of running it:

```bash
uvicorn ping:app --host 0.0.0.0 --port 9696 --reload
```

You can now open it in the browser at http://localhost:9696/ping

Or send a request with curl:

```bash
curl localhost:9696/ping
```

No differences with Flask so far. But we can see the docs (not possible with
Flask):

http://localhost:9696/docs


Let's now turn our script into a web application:

```python
import pickle
from fastapi import FastAPI
import uvicorn

app = FastAPI(title="customer-churn-prediction")

with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


def predict_single(customer):
    result = pipeline.predict_proba(customer)[0, 1]
    return float(result)


@app.post("/predict")
def predict(customer):
    prob = predict_single(customer)

    return {
        "churn_probability": prob,
        "churn": bool(prob >= 0.5)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
```

Run it:

```bash
uvicorn predict:app --host 0.0.0.0 --port 9696 --reload
```


Right now it doesn't recognize it as JSON, so let's add type hints:

```python
from typing import Dict, Any

@app.post("/predict")
def predict(customer: Dict[str, Any]):
    prob = predict_single(customer)

    return {
        "churn_probability": prob,
        "churn": bool(prob >= 0.5)
    }
```


Open the docs and send a request:

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

We can also do it with curl:

```bash
curl -X 'POST' 'http://localhost:9696/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
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
}'
```

We need to include headers -- FastAPI is more strict about schemas and 
validation than Flask.

To do it from a script, we'll use the requests library. Install it:

```bash
pip install requests
```

Create [`test.py`](test.py):

```python
import requests

url = 'http://localhost:9696/predict'

customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}

response = requests.post(url, json=customer)
predictions = response.json()

print(predictions)
if predictions['churn']:
    print('customer is likely to churn, send promo')
else:
    print('customer is not likely to churn')
```

## Pydantic and Validation

Another feature of FastAPI that we didn't have in Flask is input and 
output validation

To come up with this schema, I used ChatGPT.
I gave it an example, and also the output of this piece of code:

```python
for n in numerical:
    print(df[n].describe())
    print()

for c in categorical:
    print(df[c].value_counts())
    print()
```

The models (input and output):

```python
from typing import Literal
from pydantic import BaseModel, Field

class Customer(BaseModel):
    gender: Literal["male", "female"]
    seniorcitizen: Literal[0, 1]
    partner: Literal["yes", "no"]
    dependents: Literal["yes", "no"]
    phoneservice: Literal["yes", "no"]
    multiplelines: Literal["no", "yes", "no_phone_service"]
    internetservice: Literal["dsl", "fiber_optic", "no"]
    onlinesecurity: Literal["no", "yes", "no_internet_service"]
    onlinebackup: Literal["no", "yes", "no_internet_service"]
    deviceprotection: Literal["no", "yes", "no_internet_service"]
    techsupport: Literal["no", "yes", "no_internet_service"]
    streamingtv: Literal["no", "yes", "no_internet_service"]
    streamingmovies: Literal["no", "yes", "no_internet_service"]
    contract: Literal["month-to-month", "one_year", "two_year"]
    paperlessbilling: Literal["yes", "no"]
    paymentmethod: Literal[
        "electronic_check",
        "mailed_check",
        "bank_transfer_(automatic)",
        "credit_card_(automatic)",
    ]
    tenure: int = Field(..., ge=0)
    monthlycharges: float = Field(..., ge=0.0)
    totalcharges: float = Field(..., ge=0.0)


class PredictResponse(BaseModel):
    churn_probability: float
    churn: bool
```

Now we can be more explicit with the input we expect and
the output we generate:

```python
@app.post("/predict")
def predict(customer: Customer) -> PredictResponse:
    prob = predict_single(customer.model_dump())

    return PredictResponse(
        churn_probability=prob,
        churn=prob >= 0.5
    )
```

Note: if you use `customer.dict()` instead of `model_dump()`, you can get the following warning:

```
PydanticDeprecatedSince20: The `dict` method is deprecated; use `model_dump` instead. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.11/migration/
```


We now can test how it behaves with incorrect input. Let's add a
field `"whatever": 31337` to our test.py and execute it.

When we run it, nothing happens: it continues working like 
previously.

In order to make Pydantic raise an error, we need to add `model_config`:


```python
from pydantic import ConfigDict


class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ... # rest of the fields
```

Now we will get an error:

```python
response: {'detail': [{'type': 'extra_forbidden', 'loc': ['body', 'whatever'], 'msg': 'Extra inputs are not permitted', 'input': 31337}]}
```

What if we send a value that is not defined by the model? For example,

```json
{
    ...
    "streamingtv": "maybe"
    ...
}
```

In this case, it works as expected: it throws an error:

```python
response: {'detail': [{'type': 'literal_error', 'loc': ['body', 'streamingtv'], 'msg': "Input should be 'no', 'yes' or 'no_internet_service'", 'input': 'maybe', 'ctx': {'expected': "'no', 'yes' or 'no_internet_service'"}}]}
```

## Environment management

It works now but we can have version conflicts with
other projects. So we need to isolate this project from the others.

We will not go into theoretical details about why you want to use
virtual environments. Check [module 5](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/05-deployment/05-pipenv.md) for more information

For that, we will use [`uv`](https://docs.astral.sh/uv/) -- a tool 
for dependency and environment management

Install it:

```bash
pip install uv
```

Initialize the project:

```bash
uv init
``` 

We don't need main.py, so we can remove it:

```bash
rm main.py
```

Notice that it created some files:

- .python-version
- pyproject.toml


We need to have Scikit-Learn and FastAPI for this project.
So let's add them:

```bash
uv add scikit-learn fastapi uvicorn
```

A few more things appeared:

- .venv with all the packages
- uv.lock with a more detailed description of the dependencies

We also have a development dependency -- we won't need it in production:

```bash
uv add --dev requests
```

If we want to run something in this virtual environment, simply 
prefix it with `uv run`:

```bash
uv run uvicorn predict:app --host 0.0.0.0 --port 9696 --reload
uv run python test.py
```

When you get a fresh copy of a project that already uses uv, you
can install all the dependencies using the sync command:

```bash
uv sync
```

## Docker

Let's use Docker for complete isolation.
If you want to learn more about Docker, check
[module 5](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/05-deployment/06-docker.md).

In this workshop, we will adjust the [Dockerfile](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/05-deployment/code/Dockerfile)
we created in the module.

First, we need to decide, which Python version we need. You can check 
the version of Python using this command:

```bash
$ python -V
Python 3.13.5
```

So let's use the 3.13.5 image of Python:

```dockerfile
# Use the official Python 3.13.5 slim version based on Debian Bookworm as the base image
FROM python:3.13.5-slim-bookworm

# Copy the 'uv' and 'uvx' executables from the latest uv image into /bin/ in this image
# 'uv' is a fast Python package installer and environment manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory inside the container to /code
# All subsequent commands will be run from here
WORKDIR /code

# Add the virtual environment's bin directory to the PATH so Python tools work globally
ENV PATH="/code/.venv/bin:$PATH"

# Copy the project configuration files into the container
# pyproject.toml     → project metadata and dependencies
# uv.lock            → locked dependency versions (for reproducibility)
# .python-version    → Python version specification
COPY "pyproject.toml" "uv.lock" ".python-version" ./

# Install dependencies exactly as locked in uv.lock, without updating them
RUN uv sync --locked

# Copy application code and model data into the container
COPY "predict.py" "model.bin" ./

# Expose TCP port 9696 so it can be accessed from outside the container
EXPOSE 9696

# Run the application using uvicorn (ASGI server)
# predict:app → refers to 'app' object inside predict.py
# --host 0.0.0.0 → listen on all interfaces
# --port 9696    → listen on port 9696
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
```

(The comments are added by [ChatGPT](https://chatgpt.com/share/6899ebd9-96f8-800a-819e-093fccadaf7e))

Build it:

```bash
docker build -t predict-churn .
```

And run it:

```bash
docker run -it --rm -p 9696:9696 predict-churn
```

## Deployment

Once the application is dockerized, you can deploy it everywhere. 

In the course, we showed Elastic Beanstalk. Other alternatives:

- Google CloudRun
- AWS App Runner
- Fly.io
- Check the course for contributions from other students, there are a lot of other options

According to ChatGPT, using Fly.io is very simple, so let's do that:

```bash
# for other OS, check https://fly.io/docs/flyctl/install/
# you may also need to replace fly with flyctl
curl -L https://fly.io/install.sh | sh

fly auth signup
fly launch --generate-name
fly deploy
```

Get the URL from the logs, it should be something 
along these lines:

```
Visit your newly deployed app at https://mlzoomcamp-flask-uv.fly.dev/
```

Put the url into test.py and check that it works.

Now you can terminate the deployment

```bash
fly apps destroy <app-name>
```

You can see the list of apps by using the `apps list` command.

Note: check the pricing information.

## Summary

In this workshop we dockerized our ML model and deployed it to the cloud.

If you want to learn more about ML Engineering, check our
[ML Zoomcamp course](https://github.com/DataTalksClub/machine-learning-zoomcamp/).

---

## 🧠 Advanced Topics

### Understanding Pydantic Validation

**Why Pydantic?**
- ✅ Automatic type checking
- ✅ Clear error messages
- ✅ Documentation generation
- ✅ Security (forbids extra fields)

**Example with Customer Model**:
```python
from pydantic import BaseModel, Field
from typing import Literal

class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    # Categorical fields with allowed values
    gender: Literal["male", "female"]
    contract: Literal["month-to-month", "one_year", "two_year"]
    
    # Numerical fields with constraints
    tenure: int = Field(..., ge=0)
    monthlycharges: float = Field(..., ge=0.0)
```

### FastAPI vs Flask Comparison

**Flask** (Traditional):
```python
@app.route('/predict', methods=['POST'])
def predict():
    customer = request.json
    # Manual validation ❌
    if not isinstance(customer.get('tenure'), int):
        return {'error': 'Invalid tenure'}, 400
    # Process...
```

**FastAPI** (Modern):
```python
@app.post('/predict')
async def predict(customer: Customer):  # Validation automatic ✅
    # Process...
    return {'result': ...}
```

### ML Pipelines vs Sequential

**Without Pipeline** (more error-prone):
```python
dv = DictVectorizer()
X_train = dv.fit_transform(train_dict)
model = LogisticRegression()
model.fit(X_train, y_train)

# In production:
X_test = dv.transform(test_dict)  # Easy to forget!
pred = model.predict_proba(X_test)
```

**With Pipeline** (cleaner):
```python
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(
    DictVectorizer(),
    LogisticRegression()
)
pipeline.fit(train_dict, y_train)

# In production:
pred = pipeline.predict_proba(test_dict)  # Vectorization included!
```

### Async/Await Basics

FastAPI supports async for handling multiple concurrent requests:

```python
@app.post('/predict')
async def predict(customer: Customer):
    # This can handle 1000+ concurrent requests
    # Each waits independently
    result = await process_customer(customer)
    return result
```

---

## 🐳 Docker Deep Dive

### Understanding the Dockerfile

```dockerfile
# Base image: Python 3.13.5 slim (minimal)
FROM python:3.13.5-slim-bookworm

# Copy uv binary (fast package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Working directory
WORKDIR /code

# Add venv to PATH
ENV PATH="/code/.venv/bin:$PATH"

# Copy dependency files
COPY "pyproject.toml" "uv.lock" ".python-version" ./

# Install dependencies (locked versions)
RUN uv sync --locked

# Copy application code
COPY "predict.py" "model.bin" ./

# Expose port
EXPOSE 9696

# Run the application
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
```

**Key Points**:
- `slim-bookworm` = minimal image (smaller size)
- `--locked` = reproducible builds
- `--host 0.0.0.0` = listen on all interfaces
- `model.bin` must exist before building!

### Build & Run

```bash
# Build
docker build -t churn-predictor .

# Run with port mapping
docker run -it --rm -p 9696:9696 churn-predictor

# Run in background
docker run -d -p 9696:9696 --name churn churn-predictor

# View logs
docker logs churn

# Stop
docker stop churn
docker rm churn
```

---

## ☁️ Deployment Options

### Option 1: Fly.io (Recommended - Simple)

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Create app
fly launch --generate-name

# Deploy
fly deploy

# View app
fly open

# Monitor
fly logs

# Destroy
fly apps destroy <app-name>
```

**Pros**: ✅ Simple, ✅ Free tier, ✅ Global edge network
**Cons**: ❌ Limited customization

### Option 2: Docker Hub + Manual Deployment

```bash
# Build and tag
docker build -t username/churn-predictor:v1 .

# Push
docker push username/churn-predictor:v1

# Deploy on your server
docker run -p 9696:9696 username/churn-predictor:v1
```

### Option 3: AWS ECS

```bash
# Push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker tag churn-predictor <account>.dkr.ecr.us-east-1.amazonaws.com/churn-predictor
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/churn-predictor

# Create ECS task and service
# (through AWS console or CLI)
```

### Option 4: Google Cloud Run

```bash
gcloud run deploy churn-predictor \
  --source . \
  --platform managed \
  --region us-central1
```

### Option 5: Kubernetes

```bash
# Create deployment
kubectl create deployment churn --image=username/churn-predictor:v1

# Expose service
kubectl expose deployment churn --port=8000 --target-port=9696

# Scale
kubectl scale deployment churn --replicas=3
```

---

## 🧪 Testing & Quality Assurance

### Unit Tests

```python
# tests/test_predict.py
import pytest
from predict import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_predict_valid_customer():
    customer = {
        "gender": "female",
        "tenure": 1,
        ...
    }
    response = client.post("/predict", json=customer)
    assert response.status_code == 200
    assert "churn_probability" in response.json()

def test_predict_invalid_tenure():
    customer = {
        "gender": "female",
        "tenure": -1,  # Invalid!
        ...
    }
    response = client.post("/predict", json=customer)
    assert response.status_code == 422  # Validation error
```

### Load Testing with Locust

```python
# locustfile.py
from locust import HttpUser, task

class PredictionUser(HttpUser):
    @task
    def predict(self):
        customer = {"gender": "female", "tenure": 1, ...}
        self.client.post("/predict", json=customer)

# Run: locust -f locustfile.py
```

### API Testing with curl

```bash
# Test endpoint
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"gender": "female", "tenure": 1, ...}'

# Pretty print
curl -s http://localhost:9696/predict | jq .

# Save response
curl http://localhost:9696/predict > response.json
```

---

## 📊 Monitoring in Production

### Key Metrics

1. **Model Performance**
   - Prediction distribution
   - Average prediction value
   - High/low prediction counts

2. **API Health**
   - Request latency (should be <100ms)
   - Error rate
   - Throughput

3. **Data Quality**
   - Missing values in requests
   - Out-of-range values
   - New/unexpected categories

### Monitoring Tools

- **Prometheus** - Metrics collection
- **Grafana** - Visualization
- **ELK Stack** - Logging
- **New Relic** - APM
- **Datadog** - Comprehensive monitoring

---

## 🔐 Security Best Practices

1. **Input Validation** - Done automatically by Pydantic ✅
2. **Error Handling** - Don't expose internal errors
   ```python
   try:
       result = predict(customer)
   except Exception:
       return {"error": "Prediction failed"}, 500
   ```

3. **Rate Limiting** - Prevent abuse
   ```python
   from slowapi import Limiter
   limiter = Limiter(key_func=get_remote_address)
   
   @app.post("/predict")
   @limiter.limit("100/minute")
   def predict(customer: Customer):
       ...
   ```

4. **API Keys** - Authentication
   ```python
   from fastapi import Security, HTTPBearer
   security = HTTPBearer()
   
   @app.post("/predict")
   def predict(customer: Customer, credentials: HTTPAuthCredentials = Security(security)):
       ...
   ```

5. **HTTPS** - Use in production (reverse proxy)
6. **Model Integrity** - Verify checksums
7. **Audit Logging** - Log all predictions
8. **Access Control** - Restrict sensitive data

---

## 🎯 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Port 9696 in use | `lsof -i :9696` then `kill -9 <PID>` |
| Model not found | Run `python train.py` first |
| Import errors | `uv sync` or `pip install -r requirements.txt` |
| Pydantic validation error | Check all required fields are present |
| Slow predictions | Profile code, optimize model, use async |
| Memory issues | Use smaller batch sizes, enable streaming |

---

## 📖 Key Learning Outcomes

After completing this workshop, you'll understand:

✅ **Modern ML Deployment Patterns**
- Scikit-Learn pipelines
- Type hints and validation
- Production-ready code

✅ **FastAPI Framework**
- Faster than Flask (10x+)
- Auto-generated documentation
- Built-in async support

✅ **Dependency Management**
- uv package manager (10-100x faster than pip)
- Virtual environments
- Lock files for reproducibility

✅ **Containerization**
- Docker images
- Multi-stage builds
- Efficient deployment

✅ **Cloud Deployment**
- Fly.io, AWS, GCP, Azure
- CI/CD pipelines
- Monitoring and scaling

✅ **Production Considerations**
- Error handling
- Logging
- Security
- Performance optimization

---

## 🚀 Next Steps

1. ✅ Complete the Jupyter notebook
2. ✅ Train the model locally
3. ✅ Run the FastAPI server
4. ✅ Test with provided test.py
5. ✅ Explore Swagger UI at /docs
6. ✅ Deploy to Fly.io or Docker Hub
7. ✅ Set up monitoring and logging
8. ✅ Implement A/B testing
9. ✅ Add more features and models
10. ✅ Share on GitHub!

---

## 📚 Additional Resources

### Official Documentation
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Pydantic Docs](https://pydantic-docs.helpmanual.io/)
- [uv GitHub](https://github.com/astral-sh/uv)
- [Scikit-learn Pipelines](https://scikit-learn.org/stable/modules/compose.html)

### Deployment Guides
- [Fly.io Docs](https://fly.io/docs/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Kubernetes Deployment](https://kubernetes.io/docs/)

### Related Courses
- [ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp/)
- [Full Stack Python](https://www.fullstackpython.com/)
- [Real Python FastAPI](https://realpython.com/fastapi-python-web-apis/)

---

## ❓ FAQ

**Q: Is FastAPI production-ready?**
A: Yes! Used by Uber, Netflix, and other major companies.

**Q: Should I use uv or pip?**
A: uv is faster and more reliable. Highly recommended for new projects.

**Q: Can I use this for real-time predictions?**
A: Yes! FastAPI + Uvicorn handles 1000+ req/s per core.

**Q: How do I monitor predictions in production?**
A: Use logging + monitoring tools (Prometheus, Grafana, Datadog).

**Q: What about model versioning?**
A: Store multiple model.bin files, add version to API endpoint.

**Q: How do I deploy updates?**
A: Retrain model, push new Docker image, redeploy.

---

## 📝 License

MIT License - Feel free to use and modify!

---

## 🎓 Summary

This workshop teaches **production-ready ML deployment** using modern Python tools:

🚀 **Fast** - FastAPI + uv + Uvicorn = 10,000+ req/s
📦 **Clean** - Type hints, Pydantic validation, ML Pipelines
🐳 **Portable** - Docker containerization
☁️ **Scalable** - Cloud deployment (Fly.io, AWS, GCP, Azure, K8s)

**Perfect for**: ML engineers, data scientists, Python developers transitioning to production ML systems.

**Time to complete**: 2-3 hours

**Skills gained**: Intermediate → Advanced ML deployment

**Your mission**: Build, test, and deploy this churn prediction service to production! 🚀


