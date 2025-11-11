"""
Test Script for Churn Prediction API

This script tests the FastAPI prediction service by:
1. Connecting to the running API
2. Sending sample customer data
3. Receiving and interpreting predictions
4. Taking business action based on result

Usage:
    # Make sure API is running:
    python predict.py
    
    # In another terminal:
    python test.py
    
    # For cloud deployment, change url to deployed service:
    url = 'https://your-app.fly.dev/predict'
"""

import requests  # For HTTP requests to API


# ============================================================================
# CONFIGURATION
# ============================================================================

# Local testing
url = 'http://localhost:9696/predict'

# Uncomment for cloud deployment (e.g., Fly.io)
# url = 'https://mlzoomcamp-flask-uv.fly.dev/predict'


# ============================================================================
# TEST CUSTOMER DATA
# ============================================================================

# Sample customer with high churn risk characteristics:
# - New customer (tenure = 1 month)
# - Month-to-month contract (lower commitment)
# - No add-ons (low engagement)
customer = {
    'gender': 'female',
    'seniorcitizen': 0,
    'partner': 'yes',
    'dependents': 'no',
    'phoneservice': 'no',
    'multiplelines': 'no_phone_service',
    'internetservice': 'dsl',
    'onlinesecurity': 'no',      # No add-ons = higher churn risk
    'onlinebackup': 'yes',
    'deviceprotection': 'no',
    'techsupport': 'no',
    'streamingtv': 'no',
    'streamingmovies': 'no',
    'contract': 'month-to-month',  # Least commitment = higher risk
    'paperlessbilling': 'yes',
    'paymentmethod': 'electronic_check',
    'tenure': 1,                  # Just started = high risk
    'monthlycharges': 29.85,
    'totalcharges': 29.85
}


# ============================================================================
# MAIN TESTING
# ============================================================================

print("=" * 60)
print("CHURN PREDICTION API TEST")
print("=" * 60)
print(f"Connecting to: {url}\n")

try:
    # Send POST request with customer data as JSON
    print("Sending customer data...")
    response = requests.post(url, json=customer)
    
    # Check if request was successful
    if response.status_code != 200:
        print(f"✗ Error: API returned status code {response.status_code}")
        print(f"Response: {response.text}")
        exit(1)
    
    # Parse JSON response
    predictions = response.json()
    
    # Extract prediction results
    churn_probability = predictions['churn_probability']
    churn_prediction = predictions['churn']
    
    print("✓ API responded successfully\n")
    
    # ========================================================================
    # DISPLAY RESULTS
    # ========================================================================
    
    print("PREDICTION RESULTS:")
    print(f"  Churn Probability: {churn_probability:.1%}")
    print(f"  Binary Prediction: {'WILL CHURN' if churn_prediction else 'WILL NOT CHURN'}\n")
    
    # ========================================================================
    # BUSINESS LOGIC
    # ========================================================================
    
    # Take action based on prediction
    print("RECOMMENDED ACTION:")
    if churn_probability >= 0.7:
        print("  🚨 HIGH RISK - Send immediate retention offer")
        print('  customer is likely to churn, send promo')
    elif churn_probability >= 0.5:
        print("  ⚠️  MEDIUM RISK - Contact with special offer")
    elif churn_probability >= 0.3:
        print("  ℹ️  LOW RISK - Regular engagement campaign")
    else:
        print("  ✓ VERY LOW RISK - Standard communication")
        print('  customer is not likely to churn')
    
    print("\n" + "=" * 60)
    
except requests.exceptions.ConnectionError:
    print("✗ Error: Could not connect to API")
    print("Make sure the API is running: python predict.py")
    exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    exit(1)