"""
Reference Implementation - Direct Model Usage

This is the OLD approach (before FastAPI) showing how to:
1. Load a trained model from disk
2. Make predictions directly in Python

Advantages:
- Simple and direct
- No web server overhead
- Good for batch predictions

Disadvantages:
- NOT suitable for production
- Can't serve HTTP requests
- No automatic validation
- No API documentation
- Hard to scale

Modern approach: Use FastAPI (predict.py) instead!

Usage:
    python predict_old.py
    # Shows: Probability of churn: 0.XXX
"""

import pickle  # For loading serialized model

# Load the trained model (DictVectorizer + LogisticRegression pipeline)
with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)

# Sample customer data for testing
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

# Make prediction
# pipeline.predict_proba() returns [[prob_no_churn, prob_churn]]
# [0, 1] gets first prediction, churn probability
result = pipeline.predict_proba(datapoint)[0, 1]

# Display result
print(f'Probability of churn: {result:.3f}')

# ============================================================================
# WHY THIS IS "OLD" APPROACH
# ============================================================================
# 
# To serve this in production, you'd need:
# 1. Manual HTTP server setup (Flask, Django, etc.)
# 2. Manual request parsing (JSON → dict)
# 3. Manual input validation (type checks)
# 4. Manual response formatting (dict → JSON)
# 5. Manual error handling
# 6. Manual documentation (Swagger, etc.)
# 
# Modern approach (FastAPI):
# All of above is automatic! See predict.py
# ============================================================================