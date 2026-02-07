"""
FastAPI Application for Traffic Accident Risk Prediction
Provides REST API endpoints for accessing predictions
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, date
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.database import get_db_manager
from src.utils.config import get_config
from src.utils.logging_utils import setup_logger

# Initialize
config = get_config()
db = get_db_manager()
logger = setup_logger(__name__, 'logs/api.log')

# Create FastAPI app
app = FastAPI(
    title="Traffic Accident Risk Prediction API",
    description="API for accessing traffic accident risk predictions",
    version="1.0.0"
)


# Response models
class PredictionResponse(BaseModel):
    """Single prediction response"""
    prediction_date: str
    prediction_hour: int
    location_id: str
    latitude: float
    longitude: float
    risk_score: float
    risk_category: str
    model_version: Optional[str] = None


class HighRiskZone(BaseModel):
    """High risk zone response"""
    location_id: str
    latitude: float
    longitude: float
    prediction_date: str
    prediction_hour: int
    risk_score: float
    risk_category: str


class HighRiskZonesResponse(BaseModel):
    """Response for high-risk zones endpoint"""
    prediction_date: str
    total_zones: int
    zones: List[HighRiskZone]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    database_connected: bool


class StatsResponse(BaseModel):
    """Statistics response"""
    total_predictions: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    latest_prediction_date: Optional[str]
    model_version: Optional[str]


# Endpoints

@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "Traffic Accident Risk Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predictions_today": "/api/predictions/today",
            "predictions_by_location": "/api/predictions/location/{location_id}",
            "high_risk_zones": "/api/high-risk-zones",
            "stats": "/api/stats",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        db.execute_query("SELECT 1;", fetch=True)
        db_connected = True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        db_connected = False
    
    return {
        "status": "healthy" if db_connected else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "database_connected": db_connected
    }


@app.get("/api/predictions/today", response_model=List[PredictionResponse], tags=["Predictions"])
async def get_predictions_today(
    limit: int = Query(1000, ge=1, le=10000, description="Maximum number of results")
):
    """
    Get all predictions for today
    
    Args:
        limit: Maximum number of results to return
        
    Returns:
        List of predictions
    """
    try:
        today = datetime.now().date()
        
        query = f"""
            SELECT 
                prediction_date::text,
                prediction_hour,
                location_id,
                latitude,
                longitude,
                risk_score,
                risk_category,
                model_version
            FROM predictions
            WHERE prediction_date = '{today}'
            ORDER BY risk_score DESC
            LIMIT {limit};
        """
        
        results = db.execute_query(query, fetch=True)
        
        if not results:
            return []
        
        predictions = [
            {
                "prediction_date": r['prediction_date'],
                "prediction_hour": r['prediction_hour'],
                "location_id": r['location_id'],
                "latitude": r['latitude'],
                "longitude": r['longitude'],
                "risk_score": r['risk_score'],
                "risk_category": r['risk_category'],
                "model_version": r['model_version']
            }
            for r in results
        ]
        
        return predictions
        
    except Exception as e:
        logger.error(f"Error fetching today's predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions/date/{prediction_date}", response_model=List[PredictionResponse], tags=["Predictions"])
async def get_predictions_by_date(
    prediction_date: str,
    limit: int = Query(1000, ge=1, le=10000)
):
    """
    Get predictions for a specific date
    
    Args:
        prediction_date: Date in YYYY-MM-DD format
        limit: Maximum number of results
        
    Returns:
        List of predictions
    """
    try:
        # Validate date format
        datetime.strptime(prediction_date, '%Y-%m-%d')
        
        query = f"""
            SELECT 
                prediction_date::text,
                prediction_hour,
                location_id,
                latitude,
                longitude,
                risk_score,
                risk_category,
                model_version
            FROM predictions
            WHERE prediction_date = '{prediction_date}'
            ORDER BY risk_score DESC
            LIMIT {limit};
        """
        
        results = db.execute_query(query, fetch=True)
        
        if not results:
            return []
        
        predictions = [
            {
                "prediction_date": r['prediction_date'],
                "prediction_hour": r['prediction_hour'],
                "location_id": r['location_id'],
                "latitude": r['latitude'],
                "longitude": r['longitude'],
                "risk_score": r['risk_score'],
                "risk_category": r['risk_category'],
                "model_version": r['model_version']
            }
            for r in results
        ]
        
        return predictions
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    except Exception as e:
        logger.error(f"Error fetching predictions for {prediction_date}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/predictions/location/{location_id}", response_model=List[PredictionResponse], tags=["Predictions"])
async def get_predictions_by_location(
    location_id: str,
    prediction_date: Optional[str] = None
):
    """
    Get predictions for a specific location
    
    Args:
        location_id: Location identifier
        prediction_date: Optional date filter (YYYY-MM-DD)
        
    Returns:
        List of predictions for the location
    """
    try:
        date_filter = f"AND prediction_date = '{prediction_date}'" if prediction_date else ""
        
        query = f"""
            SELECT 
                prediction_date::text,
                prediction_hour,
                location_id,
                latitude,
                longitude,
                risk_score,
                risk_category,
                model_version
            FROM predictions
            WHERE location_id = '{location_id}'
            {date_filter}
            ORDER BY prediction_date DESC, prediction_hour;
        """
        
        results = db.execute_query(query, fetch=True)
        
        if not results:
            raise HTTPException(status_code=404, detail=f"No predictions found for location {location_id}")
        
        predictions = [
            {
                "prediction_date": r['prediction_date'],
                "prediction_hour": r['prediction_hour'],
                "location_id": r['location_id'],
                "latitude": r['latitude'],
                "longitude": r['longitude'],
                "risk_score": r['risk_score'],
                "risk_category": r['risk_category'],
                "model_version": r['model_version']
            }
            for r in results
        ]
        
        return predictions
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching predictions for location {location_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/high-risk-zones", response_model=HighRiskZonesResponse, tags=["Risk Zones"])
async def get_high_risk_zones(
    prediction_date: Optional[str] = Query(None, description="Date (YYYY-MM-DD), defaults to today"),
    top_n: int = Query(10, ge=1, le=100, description="Number of top risk zones"),
    min_risk_score: float = Query(0.75, ge=0.0, le=1.0, description="Minimum risk score")
):
    """
    Get top N highest risk zones
    
    Args:
        prediction_date: Date to query (defaults to today)
        top_n: Number of top zones to return
        min_risk_score: Minimum risk score threshold
        
    Returns:
        High-risk zones
    """
    try:
        if prediction_date is None:
            prediction_date = datetime.now().date().isoformat()
        else:
            # Validate date format
            datetime.strptime(prediction_date, '%Y-%m-%d')
        
        query = f"""
            SELECT 
                location_id,
                latitude,
                longitude,
                prediction_date::text,
                prediction_hour,
                risk_score,
                risk_category
            FROM predictions
            WHERE prediction_date = '{prediction_date}'
            AND risk_score >= {min_risk_score}
            ORDER BY risk_score DESC
            LIMIT {top_n};
        """
        
        results = db.execute_query(query, fetch=True)
        
        if not results:
            return {
                "prediction_date": prediction_date,
                "total_zones": 0,
                "zones": []
            }
        
        zones = [
            {
                "location_id": r['location_id'],
                "latitude": r['latitude'],
                "longitude": r['longitude'],
                "prediction_date": r['prediction_date'],
                "prediction_hour": r['prediction_hour'],
                "risk_score": r['risk_score'],
                "risk_category": r['risk_category']
            }
            for r in results
        ]
        
        return {
            "prediction_date": prediction_date,
            "total_zones": len(zones),
            "zones": zones
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    except Exception as e:
        logger.error(f"Error fetching high-risk zones: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats", response_model=StatsResponse, tags=["Statistics"])
async def get_statistics():
    """
    Get prediction statistics
    
    Returns:
        Statistics about predictions
    """
    try:
        # Get total predictions
        total_query = "SELECT COUNT(*) as count FROM predictions;"
        total_result = db.execute_query(total_query, fetch=True)
        total_predictions = total_result[0]['count'] if total_result else 0
        
        # Get counts by risk category
        stats_query = """
            SELECT 
                risk_category,
                COUNT(*) as count
            FROM predictions
            GROUP BY risk_category;
        """
        stats_results = db.execute_query(stats_query, fetch=True)
        
        risk_counts = {r['risk_category']: r['count'] for r in stats_results} if stats_results else {}
        
        # Get latest prediction date and model version
        latest_query = """
            SELECT 
                MAX(prediction_date)::text as latest_date,
                model_version
            FROM predictions
            GROUP BY model_version
            ORDER BY MAX(prediction_date) DESC
            LIMIT 1;
        """
        latest_result = db.execute_query(latest_query, fetch=True)
        
        latest_date = latest_result[0]['latest_date'] if latest_result else None
        model_version = latest_result[0]['model_version'] if latest_result else None
        
        return {
            "total_predictions": total_predictions,
            "high_risk_count": risk_counts.get('High', 0),
            "medium_risk_count": risk_counts.get('Medium', 0),
            "low_risk_count": risk_counts.get('Low', 0),
            "latest_prediction_date": latest_date,
            "model_version": model_version
        }
        
    except Exception as e:
        logger.error(f"Error fetching statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Resource not found"}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


# Startup and shutdown events

@app.on_event("startup")
async def startup_event():
    """Startup tasks"""
    logger.info("Starting Traffic Accident Risk Prediction API")
    logger.info(f"API Configuration: {config.api}")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown tasks"""
    logger.info("Shutting down Traffic Accident Risk Prediction API")


if __name__ == "__main__":
    import uvicorn
    
    api_config = config.api
    
    uvicorn.run(
        "app:app",
        host=api_config.get('host', '0.0.0.0'),
        port=api_config.get('port', 8000),
        reload=api_config.get('reload', False),
        workers=api_config.get('workers', 1)
    )
