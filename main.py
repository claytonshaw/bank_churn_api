import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pickle
import uvicorn
import pandas as pd
import xgboost as xgb

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastAPI app instance with additional metadata
app = FastAPI(
    title="ML Model API",
    description="API for serving XGBoost model predictions",
    version="1.0"
)

# Enable CORS (adjust allowed origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model with field descriptions and example values
class InputData(BaseModel):
    age: int = Field(..., example=30, description="Age of the customer")
    balance: float = Field(..., example=50000.0, description="Account balance")
    products_number: int = Field(..., example=2, description="Number of products held")
    estimated_salary: float = Field(..., example=60000.0, description="Estimated salary")
    credit_score: int = Field(..., example=700, description="Credit score of the customer")
    country: str = Field(..., example="USA", description="Country of residence")
    active_member: int = Field(..., example=1, description="Whether the customer is active (1) or not (0)")
    gender: str = Field(..., example="Male", description="Gender of the customer")
    tenure: int = Field(..., example=5, description="Number of years with the bank")
    credit_card: int = Field(..., example=1, description="Whether the customer has a credit card (1) or not (0)")

# Global variable for the model
model = None

# Startup event to load the model
@app.on_event("startup")
def load_model():
    global model
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise RuntimeError("Model could not be loaded.")

# Custom exception handler for HTTPExceptions (if needed)
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"message": exc.detail})

# Prediction endpoint with enhanced error handling
@app.post("/predict")
def predict(data: InputData):
    try:
        # Create a dictionary of features
        data_dict = {
            "age": [data.age],
            "balance": [data.balance],
            "products_number": [data.products_number],
            "estimated_salary": [data.estimated_salary],
            "credit_score": [data.credit_score],
            "country": [data.country],
            "active_member": [data.active_member],
            "gender": [data.gender],
            "tenure": [data.tenure],
            "credit_card": [data.credit_card]
        }
        
        # Convert dictionary to DataFrame
        df = pd.DataFrame(data_dict)
        
        # Convert categorical columns to Pandas category dtype
        df["country"] = df["country"].astype("category")
        df["gender"] = df["gender"].astype("category")
        
        # Reorder columns to match training order
        expected_order = [
            "credit_score", 
            "country", 
            "gender", 
            "age", 
            "tenure", 
            "balance", 
            "products_number", 
            "credit_card", 
            "active_member", 
            "estimated_salary"
        ]
        df = df[expected_order]
        
        # Check if model is loaded
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Make prediction
        prediction = model.predict(df)
        # Convert NumPy type to native Python int
        result = int(prediction[0])
        logger.info(f"Prediction successful: {result}")
        return {"prediction": result}
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error processing prediction")

# Shutdown event (if you need to clean up resources)
@app.on_event("shutdown")
def shutdown_event():
    logger.info("Shutting down API...")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
