"""
Simple Health Check Service using FastAPI

This is a minimal example showing how to create a FastAPI application.
It demonstrates:
- Creating a FastAPI app
- Defining a simple GET endpoint
- Running with Uvicorn

This can be used for:
- Monitoring/health checks
- Load balancer heartbeat
- Service availability testing
- Understanding FastAPI basics

Usage:
    python ping.py
    # Visit http://localhost:9696/ping in browser
    # Or: curl http://localhost:9696/ping
    # Or: curl http://localhost:9696/docs for auto-generated docs
"""

from fastapi import FastAPI
import uvicorn


# Create FastAPI application
app = FastAPI(title="ping")


@app.get("/ping")
def ping():
    """
    Health check endpoint.
    
    Returns:
        str: "PONG" - Simple acknowledgment that service is running
        
    Usage:
        GET http://localhost:9696/ping
        Response: "PONG"
        
    This endpoint can be:
    - Used by load balancers to check if service is alive
    - Used by monitoring systems to verify uptime
    - Used in container orchestration (Kubernetes, Docker Compose)
    """
    return "PONG"


if __name__ == "__main__":
    # Run the FastAPI application using Uvicorn ASGI server
    # host="0.0.0.0" → accept requests from any network interface
    # port=9696 → listen on port 9696
    uvicorn.run(app, host="0.0.0.0", port=9696)
