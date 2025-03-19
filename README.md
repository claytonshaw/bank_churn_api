# ML Model API for Bank Customer Churn Prediction

This repository contains code and resources for serving an XGBoost model prediction through a FastAPI endpoint. The project includes a FastAPI application, a pre-trained model, and supporting files to predict bank customer churn based on various customer attributes.

## Overview

The FastAPI application (`main.py`) exposes a `/predict` endpoint that accepts customer data in JSON format and returns a churn prediction. The model, trained using XGBoost, is loaded from a serialized file (`model.pkl`) during application startup.

## Repository Contents

- **main.py:** The FastAPI application code that sets up the API, loads the model, and defines the prediction endpoint.
- **model.pkl:** A serialized XGBoost model used for making predictions.
- **bank_customer_churn.ipynb:** A Jupyter Notebook that contains the data exploration and model training process.
- **Bank Customer Churn Prediction.csv:** The dataset used for training the model (sourced from Kaggle).  
  **Note:** Due to licensing restrictions, this file may be omitted from the public repository. See the [Dataset](#dataset) section for more details.

## Dataset

The dataset was obtained from Kaggle.

## Getting Started

### Prerequisites

- Python 3.7+
- Required Python packages:
  - FastAPI
  - Uvicorn
  - pandas
  - xgboost
  - pydantic

You can install the dependencies using pip:

```bash
pip install fastapi uvicorn pandas xgboost pydantic
