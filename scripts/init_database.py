"""
Database Schema Initialization Script
Creates all necessary tables for the Traffic Accident Risk Prediction System
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import yaml
import os
from pathlib import Path


def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_database(config):
    """Create database if it doesn't exist"""
    db_config = config['database']
    
    # Connect to PostgreSQL server
    conn = psycopg2.connect(
        host=db_config['host'],
        port=db_config['port'],
        user=db_config['user'],
        password=db_config['password'],
        database='postgres'
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    # Check if database exists
    cursor.execute(
        f"SELECT 1 FROM pg_database WHERE datname = '{db_config['name']}'"
    )
    exists = cursor.fetchone()
    
    if not exists:
        cursor.execute(f"CREATE DATABASE {db_config['name']}")
        print(f"Database '{db_config['name']}' created successfully")
    else:
        print(f"Database '{db_config['name']}' already exists")
    
    cursor.close()
    conn.close()


def create_schema(config):
    """Create all database tables and PostGIS extension"""
    db_config = config['database']
    
    conn = psycopg2.connect(
        host=db_config['host'],
        port=db_config['port'],
        user=db_config['user'],
        password=db_config['password'],
        database=db_config['name']
    )
    cursor = conn.cursor()
    
    # Enable PostGIS extension
    cursor.execute("CREATE EXTENSION IF NOT EXISTS postgis;")
    print("PostGIS extension enabled")
    
    # Create accidents table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS accidents (
            accident_id VARCHAR(255) PRIMARY KEY,
            timestamp TIMESTAMP NOT NULL,
            location GEOMETRY(Point, 4326) NOT NULL,
            latitude DOUBLE PRECISION NOT NULL,
            longitude DOUBLE PRECISION NOT NULL,
            severity INTEGER,
            distance_mi DOUBLE PRECISION,
            temperature_f DOUBLE PRECISION,
            wind_chill_f DOUBLE PRECISION,
            humidity DOUBLE PRECISION,
            pressure_in DOUBLE PRECISION,
            visibility_mi DOUBLE PRECISION,
            wind_speed_mph DOUBLE PRECISION,
            precipitation_in DOUBLE PRECISION,
            weather_condition VARCHAR(100),
            wind_direction VARCHAR(10),
            amenity BOOLEAN,
            bump BOOLEAN,
            crossing BOOLEAN,
            give_way BOOLEAN,
            junction BOOLEAN,
            no_exit BOOLEAN,
            railway BOOLEAN,
            roundabout BOOLEAN,
            station BOOLEAN,
            stop BOOLEAN,
            traffic_calming BOOLEAN,
            traffic_signal BOOLEAN,
            turning_loop BOOLEAN,
            sunrise_sunset VARCHAR(10),
            civil_twilight VARCHAR(10),
            nautical_twilight VARCHAR(10),
            astronomical_twilight VARCHAR(10),
            street VARCHAR(255),
            city VARCHAR(100),
            county VARCHAR(100),
            state VARCHAR(50),
            zipcode VARCHAR(20),
            country VARCHAR(50),
            timezone VARCHAR(50),
            airport_code VARCHAR(10),
            weather_timestamp TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    print("Table 'accidents' created")
    
    # Create spatial index on accidents
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_accidents_location 
        ON accidents USING GIST(location);
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_accidents_timestamp 
        ON accidents(timestamp);
    """)
    print("Indexes created on accidents table")
    
    # Create weather_daily table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weather_daily (
            id SERIAL PRIMARY KEY,
            date DATE NOT NULL,
            location GEOMETRY(Point, 4326),
            latitude DOUBLE PRECISION,
            longitude DOUBLE PRECISION,
            temperature_f DOUBLE PRECISION,
            feels_like_f DOUBLE PRECISION,
            temp_min_f DOUBLE PRECISION,
            temp_max_f DOUBLE PRECISION,
            pressure_hpa DOUBLE PRECISION,
            humidity INTEGER,
            visibility_m INTEGER,
            wind_speed_mph DOUBLE PRECISION,
            wind_deg INTEGER,
            precipitation_mm DOUBLE PRECISION,
            weather_main VARCHAR(50),
            weather_description VARCHAR(255),
            clouds INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date, latitude, longitude)
        );
    """)
    print("Table 'weather_daily' created")
    
    # Create calendar table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS calendar (
            date DATE PRIMARY KEY,
            year INTEGER NOT NULL,
            month INTEGER NOT NULL,
            day INTEGER NOT NULL,
            day_of_week INTEGER NOT NULL,
            day_name VARCHAR(20) NOT NULL,
            is_weekend BOOLEAN NOT NULL,
            is_holiday BOOLEAN NOT NULL,
            holiday_name VARCHAR(255),
            quarter INTEGER NOT NULL,
            week_of_year INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    print("Table 'calendar' created")
    
    # Create features_daily table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS features_daily (
            id SERIAL PRIMARY KEY,
            feature_date DATE NOT NULL,
            location_id VARCHAR(100) NOT NULL,
            latitude DOUBLE PRECISION,
            longitude DOUBLE PRECISION,
            
            -- Temporal features
            hour INTEGER,
            day_of_week INTEGER,
            month INTEGER,
            is_weekend BOOLEAN,
            is_holiday BOOLEAN,
            is_rush_hour BOOLEAN,
            
            -- Weather features
            temperature_f DOUBLE PRECISION,
            visibility_mi DOUBLE PRECISION,
            precipitation_in DOUBLE PRECISION,
            wind_speed_mph DOUBLE PRECISION,
            humidity DOUBLE PRECISION,
            pressure_in DOUBLE PRECISION,
            weather_condition VARCHAR(100),
            is_severe_weather BOOLEAN,
            
            -- Spatial features
            accident_freq_historical INTEGER,
            rolling_7day_accidents INTEGER,
            rolling_30day_accidents INTEGER,
            cluster_id INTEGER,
            road_type VARCHAR(50),
            
            -- Lag features
            accidents_last_24h INTEGER,
            accidents_last_7days INTEGER,
            moving_avg_3day DOUBLE PRECISION,
            moving_avg_7day DOUBLE PRECISION,
            moving_avg_30day DOUBLE PRECISION,
            
            -- Target variable
            accident_occurred_next_24h INTEGER,
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(feature_date, location_id, hour)
        );
    """)
    print("Table 'features_daily' created")
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_features_date 
        ON features_daily(feature_date);
    """)
    
    # Create predictions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            prediction_date DATE NOT NULL,
            prediction_hour INTEGER NOT NULL,
            location_id VARCHAR(100) NOT NULL,
            latitude DOUBLE PRECISION NOT NULL,
            longitude DOUBLE PRECISION NOT NULL,
            risk_score DOUBLE PRECISION NOT NULL,
            risk_category VARCHAR(20) NOT NULL,
            model_version VARCHAR(50),
            features JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(prediction_date, prediction_hour, location_id)
        );
    """)
    print("Table 'predictions' created")
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_predictions_date 
        ON predictions(prediction_date);
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_predictions_risk_score 
        ON predictions(risk_score DESC);
    """)
    
    # Create model_metadata table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_metadata (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(100) NOT NULL,
            model_version VARCHAR(50) NOT NULL,
            model_type VARCHAR(50) NOT NULL,
            training_date TIMESTAMP NOT NULL,
            
            -- Performance metrics
            auc_score DOUBLE PRECISION,
            precision_score DOUBLE PRECISION,
            recall_score DOUBLE PRECISION,
            f1_score DOUBLE PRECISION,
            accuracy_score DOUBLE PRECISION,
            
            -- Training metadata
            train_samples INTEGER,
            val_samples INTEGER,
            test_samples INTEGER,
            features_used TEXT[],
            hyperparameters JSONB,
            
            -- Model artifacts
            model_path VARCHAR(500),
            mlflow_run_id VARCHAR(100),
            
            -- Deployment
            is_production BOOLEAN DEFAULT FALSE,
            deployed_at TIMESTAMP,
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(model_name, model_version)
        );
    """)
    print("Table 'model_metadata' created")
    
    # Create monitoring_metrics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS monitoring_metrics (
            id SERIAL PRIMARY KEY,
            metric_date DATE NOT NULL,
            model_version VARCHAR(50) NOT NULL,
            
            -- Performance metrics
            auc_score DOUBLE PRECISION,
            precision_score DOUBLE PRECISION,
            recall_score DOUBLE PRECISION,
            f1_score DOUBLE PRECISION,
            
            -- Prediction statistics
            total_predictions INTEGER,
            high_risk_predictions INTEGER,
            medium_risk_predictions INTEGER,
            low_risk_predictions INTEGER,
            
            -- Actual outcomes (if available)
            true_positives INTEGER,
            false_positives INTEGER,
            true_negatives INTEGER,
            false_negatives INTEGER,
            
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(metric_date, model_version)
        );
    """)
    print("Table 'monitoring_metrics' created")
    
    # Create location_clusters table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS location_clusters (
            cluster_id INTEGER PRIMARY KEY,
            centroid_latitude DOUBLE PRECISION NOT NULL,
            centroid_longitude DOUBLE PRECISION NOT NULL,
            centroid_location GEOMETRY(Point, 4326),
            accident_count INTEGER DEFAULT 0,
            avg_severity DOUBLE PRECISION,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    print("Table 'location_clusters' created")
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_location_clusters_location 
        ON location_clusters USING GIST(centroid_location);
    """)
    
    conn.commit()
    cursor.close()
    conn.close()
    
    print("\n✓ Database schema created successfully!")


def main():
    """Main function to initialize database"""
    print("Initializing Traffic Accident Prediction Database...\n")
    
    config = load_config()
    
    # Create database
    create_database(config)
    
    # Create schema
    create_schema(config)
    
    print("\nDatabase initialization complete!")
    print(f"Database: {config['database']['name']}")
    print(f"Host: {config['database']['host']}")
    print(f"Port: {config['database']['port']}")


if __name__ == "__main__":
    main()
