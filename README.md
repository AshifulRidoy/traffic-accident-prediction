# CollisionCast

This project is an end-to-end pipeline designed to predict traffic accident risks 24 hours in advance. By combining historical accident records with weather forecasts and temporal data, the system identifies high-risk zones to help city authorities allocate emergency resources more effectively.

## Core Capabilities
Risk Forecasting: Generates 24-hour lead-time predictions for accident probability.

Automated Pipeline: Handles everything from data ingestion and feature engineering to model training and daily batch predictions.

Production Ready: Includes a REST API for integration, Docker support for deployment, and MLflow for experiment tracking.

Geospatial Analysis: Uses PostgreSQL and PostGIS to manage spatial data and identify specific high-risk coordinates.

##  Key Features

- **Feature Engineering**: Temporal, weather, spatial, and lag features with data leakage prevention
- **ML Models**: Logistic Regression, Random Forest, and XGBoost
- **MLflow Integration**: Complete experiment tracking and model versioning
- **Batch Prediction Pipeline**: Daily automated predictions
- **REST API**: Easy access to predictions and high-risk zones
- 

##  Architecture

```
Data Sources → Ingestion → Feature Engineering → Model Training → Predictions → API
                    ↓              ↓                    ↓              ↓
                PostgreSQL    S3 Storage           MLflow      Monitoring
```

##  Project Structure

```
traffic-accident-prediction/
├── data/
│   ├── raw/              # Raw datasets 
│   ├── processed/        # Cleaned and validated data
│   ├── features/         # Engineered features
│   └── predictions/      # Model predictions
├── src/
│   ├── ingestion/        # Data ingestion scripts
│   ├── features/         # Feature engineering pipeline
│   ├── models/           # Model training and evaluation
│   ├── api/              # REST API service
│   └── utils/            # Shared utilities
├── notebooks/            
├── models/               # Trained model artifacts
├── config/               # Configuration files
├── scripts/              # Automation scripts
├── tests/                
└── docker/               
```

## Getting Started

### Prerequisites

- Python 3.9+
- PostgreSQL 14+ with PostGIS extension
- Docker and Docker Compose (optional)
- Kaggle US Accidents dataset

### Installation

1. **Clone and setup environment:**
```bash
cd traffic-accident-prediction
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Setup PostgreSQL database:**
```bash
# Start PostgreSQL with PostGIS
docker-compose up -d postgres

# Initialize database schema
python scripts/init_database.py
```

3. **Place your dataset:**
```bash
# Copy the Kaggle US Accidents dataset to data/raw/
cp /path/to/US_Accidents.csv data/raw/
```

4. **Configure the system:**
```bash
# Copy example config and update with your settings
cp config/config.example.yaml config/config.yaml
# Edit config.yaml with your database credentials and API keys
```

### Running the Pipeline

** Run Complete Pipeline**
```bash
# This runs: ingestion → feature engineering → training → prediction
python scripts/run_pipeline.py --all
```
**or

** Run Individual Steps**
```bash
# 1. Data ingestion
python src/ingestion/ingest_accidents.py
python src/ingestion/ingest_weather.py

# 2. Feature engineering
python src/features/build_features.py

# 3. Model training
python src/models/train.py

# 4. Generate predictions
python src/models/predict.py

# 5. Start API server
python src/api/app.py
```

### Using Docker

```bash
# Build and run all services
docker-compose up --build

# Access the API at http://localhost:8000
# Access MLflow UI at http://localhost:5000
```


## API Endpoints

```bash
# Get all predictions for today
GET /api/predictions/today

# Get predictions for specific location
GET /api/predictions/location/{location_id}

# Get top N highest risk zones
GET /api/high-risk-zones?top_n=10

# Health check
GET /health
```



##  Development Notes
- Data Integrity: The system uses strict time-based splits for training and validation to prevent data leakage.

- Monitoring: Use the MLflow UI (mlflow ui) to track model versions and performance metrics like AUC-ROC and Recall.

- Testing: Run pytest to execute the test suite.

##  Configuration

Edit `config/config.yaml` to customize:

- Database connection settings
- Weather API credentials
- Model hyperparameters
- Feature engineering parameters
- Prediction thresholds
- API settings

## 📝 Data Requirements

### Accident Data (Required)
- Kaggle US Accidents Dataset
- Format: CSV with coordinates, timestamp, severity, weather conditions
- Size: ~3GB (2016-2023 data)


### Calendar Data (Generated)
- System automatically generates holiday and calendar features



##  Model Retraining

The system supports automated retraining:

```bash
# Manual retraining with latest data
python scripts/retrain_model.py

# Scheduled retraining (add to cron)
0 2 * * 0 /path/to/venv/bin/python /path/to/scripts/retrain_model.py
```

## Technical Stack
Language: Python 3.9+

- Data/ML: Pandas, Scikit-learn, XGBoost, LightGBM

- Storage: PostgreSQL with PostGIS

-Deployment: FastAPI, Docker, Docker Compose

-Tracking: MLflow

## ⚠️ Important Notes

### Data Leakage Prevention
All rolling and lag features use ONLY past data. Time-based splits ensure training data precedes validation/test periods.

