"""
FastAPI Web Service for Churn Prediction

This module provides a production-ready REST API for making churn predictions.

Features:
- Automatic input validation (Pydantic)
- Type hints for all requests/responses
- Auto-generated API documentation at /docs
- Error handling with meaningful messages
- CORS support for cross-origin requests

Usage:
    python predict.py
    # Visit http://localhost:9696/docs for interactive API documentation
    
    OR
    
    uvicorn predict:app --host 0.0.0.0 --port 9696 --reload
"""

import pickle  # For loading serialized model
from typing import Literal  # For restricting enum values
from pydantic import BaseModel, Field, ConfigDict  # For request/response validation

from fastapi import FastAPI  # Web framework
import uvicorn  # ASGI server


# ============================================================================
# PYDANTIC MODELS - Request/Response Validation
# ============================================================================

class Customer(BaseModel):
    """
    Customer data for churn prediction request.
    
    Pydantic automatically:
    - Validates each field type
    - Checks enum values (Literal)
    - Validates numeric ranges (Field constraints)
    - Converts types (e.g., "1" string → 1 int)
    - Generates JSON schema for API docs
    
    Example:
        {
            "gender": "female",
            "tenure": 12,
            "monthlycharges": 65.5,
            ...
        }
    """
    # Demographic information
    gender: Literal["male", "female"]  # Must be one of these values
    seniorcitizen: Literal[0, 1]        # Binary: 0 (not senior), 1 (senior)
    partner: Literal["yes", "no"]       # Has a partner?
    dependents: Literal["yes", "no"]    # Has dependents?
    
    # Service subscriptions
    phoneservice: Literal["yes", "no"]
    multiplelines: Literal["no", "yes", "no_phone_service"]
    internetservice: Literal["dsl", "fiber_optic", "no"]
    
    # Internet add-ons
    onlinesecurity: Literal["no", "yes", "no_internet_service"]
    onlinebackup: Literal["no", "yes", "no_internet_service"]
    deviceprotection: Literal["no", "yes", "no_internet_service"]
    techsupport: Literal["no", "yes", "no_internet_service"]
    streamingtv: Literal["no", "yes", "no_internet_service"]
    streamingmovies: Literal["no", "yes", "no_internet_service"]
    
    # Contract and billing
    contract: Literal["month-to-month", "one_year", "two_year"]  # Higher tenure with longer contracts
    paperlessbilling: Literal["yes", "no"]
    paymentmethod: Literal[
        "electronic_check",
        "mailed_check",
        "bank_transfer_(automatic)",
        "credit_card_(automatic)",
    ]
    
    # Numerical customer metrics
    # Field(..., ge=0) means: required, must be >= 0
    tenure: int = Field(..., ge=0, description="Months as customer (0-72)")
    monthlycharges: float = Field(..., ge=0.0, description="Monthly bill ($18-$119)")
    totalcharges: float = Field(..., ge=0.0, description="Total charges ($0-$8,684)")


class PredictResponse(BaseModel):
    """
    Response from churn prediction API.
    
    Fields:
        churn_probability (float): Probability customer will churn (0.0-1.0)
        churn (bool): Simple binary prediction (prob >= 0.5)
    
    Example:
        {
            "churn_probability": 0.847,
            "churn": true
        }
    """
    churn_probability: float = Field(..., ge=0.0, le=1.0, description="Churn probability (0-1)")
    churn: bool = Field(..., description="Simple binary prediction")


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

# Initialize FastAPI application with title (shows in /docs)
app = FastAPI(
    title="customer-churn-prediction",
    description="Predict customer churn probability",
    version="1.0.0"
)

# Load trained model at startup
# This happens once when the server starts, not on every request
with open('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)
    print("✓ Model loaded successfully")


def predict_single(customer: dict) -> float:
    """
    Make prediction for a single customer.
    
    Args:
        customer (dict): Customer features dict from Pydantic model
        
    Returns:
        float: Churn probability (0.0-1.0)
        
    Implementation:
        1. pipeline.predict_proba() returns [[prob_no_churn, prob_churn]]
        2. [0, 1] gets first prediction, churn column (second)
        3. float() converts NumPy float to Python float
    """
    # Get probability from pipeline
    # Shape: (1, 2) → [[prob_no_churn, prob_churn]]
    result = pipeline.predict_proba(customer)[0, 1]
    
    # Convert NumPy float64 to Python float (for JSON serialization)
    return float(result)


@app.get("/ping")
def ping():
    """
    Health check endpoint.
    
    Returns:
        dict: Simple status indicator
        
    Usage:
        GET http://localhost:9696/ping
        Response: {"status": "ok"}
    """
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: Customer) -> PredictResponse:
    """
    Make churn prediction for a customer.
    
    Pydantic automatically:
    - Validates all fields based on Customer model
    - Returns 422 error if validation fails
    - Converts JSON input to Python types
    
    Args:
        customer (Customer): Customer data (auto-validated by Pydantic)
        
    Returns:
        PredictResponse: Prediction probability and binary decision
        
    Example:
        POST http://localhost:9696/predict
        
        Request body:
        {
            "gender": "female",
            "tenure": 1,
            "monthlycharges": 29.85,
            ...
        }
        
        Response:
        {
            "churn_probability": 0.847,
            "churn": true
        }
    """
    # Convert Pydantic model to dict for pipeline
    prob = predict_single(customer.model_dump())
    
    # Return structured response with both probability and binary decision
    # Binary decision: churn if probability >= 0.5
    return PredictResponse(
        churn_probability=prob,
        churn=prob >= 0.5
    )


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run the FastAPI application using Uvicorn ASGI server
    # host="0.0.0.0" → listen on all network interfaces
    # port=9696 → listen on port 9696
    # reload=True → auto-reload on code changes (dev mode)
    uvicorn.run(app, host="0.0.0.0", port=9696, reload=False)




