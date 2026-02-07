"""
Batch Prediction Pipeline
Generates daily predictions for accident risk
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta
import joblib
from typing import Optional
import json

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.database import get_db_manager
from src.utils.config import get_config
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__, 'logs/prediction.log')


class BatchPredictor:
    """Generate batch predictions for accident risk"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize batch predictor
        
        Args:
            config_path: Path to config.yaml
        """
        self.config = get_config(config_path)
        self.db = get_db_manager(config_path)
        
        self.models_dir = Path(self.config.paths['models'])
        self.predictions_dir = Path(self.config.paths['predictions'])
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        
        self.risk_thresholds = self.config.get('evaluation.risk_thresholds', {
            'low': 0.0,
            'medium': 0.5,
            'high': 0.75
        })
        
        self.model = None
        self.model_version = None
        self.feature_names = None
    
    def load_model(self, model_name: str = 'xgboost'):
        """
        Load trained model
        
        Args:
            model_name: Name of model to load
        """
        logger.info(f"Loading {model_name} model...")
        
        model_path = self.models_dir / f"{model_name}_model.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = joblib.load(model_path)
        self.model_version = f"{model_name}_{datetime.now().strftime('%Y%m%d')}"
        
        # Load feature names from model metadata
        query = f"""
            SELECT features_used, model_version
            FROM model_metadata
            WHERE model_name = '{model_name}'
            ORDER BY training_date DESC
            LIMIT 1;
        """
        
        result = self.db.execute_query(query, fetch=True)
        
        if result and result[0]['features_used']:
            self.feature_names = result[0]['features_used']
            self.model_version = result[0]['model_version']
        
        logger.info(f"✓ Model loaded: {self.model_version}")
        logger.info(f"  Features: {len(self.feature_names) if self.feature_names else 'unknown'}")
    
    def load_features_for_prediction(
        self,
        prediction_date: str,
        hours: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Load features for specific prediction date
        
        Args:
            prediction_date: Date to predict for (YYYY-MM-DD)
            hours: List of hours to predict (default: all 24 hours)
            
        Returns:
            Feature DataFrame
        """
        if hours is None:
            hours = list(range(24))
        
        hours_str = ','.join(map(str, hours))
        
        logger.info(f"Loading features for {prediction_date}, hours: {hours}")
        
        query = f"""
            SELECT *
            FROM features_daily
            WHERE feature_date = '{prediction_date}'
            AND hour IN ({hours_str})
            ORDER BY location_id, hour;
        """
        
        df = self.db.read_sql(query)
        
        if len(df) == 0:
            logger.warning(f"No features found for {prediction_date}")
            return df
        
        logger.info(f"Loaded {len(df):,} feature records")
        return df
    
    def prepare_features_for_prediction(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for prediction
        
        Args:
            df: Feature DataFrame
            
        Returns:
            Feature matrix
        """
        # Columns to exclude
        exclude_cols = [
            'id', 'feature_date', 'location_id', 'latitude', 'longitude',
            'accident_occurred_next_24h', 'created_at', 'weather_condition'
        ]
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle categorical features
        categorical_cols = ['road_type']
        for col in categorical_cols:
            if col in feature_cols:
                # One-hot encode
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                feature_cols.remove(col)
                feature_cols.extend(dummies.columns.tolist())
        
        # If we have feature names from training, ensure consistency
        if self.feature_names:
            # Add missing features
            for feat in self.feature_names:
                if feat not in feature_cols:
                    df[feat] = 0
            
            # Use only training features in the same order
            X = df[self.feature_names].fillna(0).values
        else:
            # Use all available features
            X = df[feature_cols].fillna(0).values
        
        return X
    
    def categorize_risk(self, risk_score: float) -> str:
        """
        Categorize risk score into Low/Medium/High
        
        Args:
            risk_score: Risk probability score (0-1)
            
        Returns:
            Risk category
        """
        if risk_score >= self.risk_thresholds['high']:
            return 'High'
        elif risk_score >= self.risk_thresholds['medium']:
            return 'Medium'
        else:
            return 'Low'
    
    def generate_predictions(
        self,
        prediction_date: str,
        hours: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Generate predictions for a specific date
        
        Args:
            prediction_date: Date to predict for
            hours: Hours to predict (default: all 24)
            
        Returns:
            DataFrame with predictions
        """
        logger.info(f"Generating predictions for {prediction_date}...")
        
        # Load features
        features_df = self.load_features_for_prediction(prediction_date, hours)
        
        if len(features_df) == 0:
            logger.warning("No features available for prediction")
            return pd.DataFrame()
        
        # Prepare feature matrix
        X = self.prepare_features_for_prediction(features_df)
        
        # Generate predictions
        risk_scores = self.model.predict_proba(X)[:, 1]
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'prediction_date': prediction_date,
            'prediction_hour': features_df['hour'],
            'location_id': features_df['location_id'],
            'latitude': features_df['latitude'],
            'longitude': features_df['longitude'],
            'risk_score': risk_scores,
            'risk_category': [self.categorize_risk(score) for score in risk_scores],
            'model_version': self.model_version
        })
        
        # Add features as JSON (optional)
        if self.config.get('prediction.include_features', False):
            predictions_df['features'] = features_df[self.feature_names if self.feature_names else []].to_dict('records')
        else:
            predictions_df['features'] = None
        
        logger.info(f"Generated {len(predictions_df):,} predictions")
        logger.info(f"  High risk: {(predictions_df['risk_category'] == 'High').sum():,}")
        logger.info(f"  Medium risk: {(predictions_df['risk_category'] == 'Medium').sum():,}")
        logger.info(f"  Low risk: {(predictions_df['risk_category'] == 'Low').sum():,}")
        
        return predictions_df
    
    def save_predictions_to_database(self, predictions_df: pd.DataFrame):
        """
        Save predictions to database
        
        Args:
            predictions_df: Predictions DataFrame
        """
        logger.info("Saving predictions to database...")
        
        # Convert features to JSON string if present
        if 'features' in predictions_df.columns and predictions_df['features'].notna().any():
            predictions_df['features'] = predictions_df['features'].apply(
                lambda x: json.dumps(x) if x is not None else None
            )
        
        # Delete existing predictions for this date
        pred_date = predictions_df['prediction_date'].iloc[0]
        delete_query = f"DELETE FROM predictions WHERE prediction_date = '{pred_date}';"
        self.db.execute_query(delete_query)
        
        # Insert new predictions
        self.db.to_sql(
            predictions_df,
            'predictions',
            if_exists='append',
            index=False,
            chunksize=1000
        )
        
        logger.info(f"✓ Saved {len(predictions_df):,} predictions to database")
    
    def save_predictions_to_file(
        self,
        predictions_df: pd.DataFrame,
        output_format: str = 'csv'
    ):
        """
        Save predictions to file
        
        Args:
            predictions_df: Predictions DataFrame
            output_format: Output format ('csv' or 'parquet')
        """
        pred_date = predictions_df['prediction_date'].iloc[0]
        
        if output_format == 'parquet':
            output_file = self.predictions_dir / f"predictions_{pred_date}.parquet"
            predictions_df.to_parquet(output_file, index=False)
        else:
            output_file = self.predictions_dir / f"predictions_{pred_date}.csv"
            
            # Drop features column for CSV (too large)
            df_save = predictions_df.drop('features', axis=1, errors='ignore')
            df_save.to_csv(output_file, index=False)
        
        logger.info(f"✓ Saved predictions to {output_file}")
    
    def get_high_risk_zones(
        self,
        predictions_df: pd.DataFrame,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Get top N highest risk zones
        
        Args:
            predictions_df: Predictions DataFrame
            top_n: Number of top zones to return
            
        Returns:
            Top risk zones
        """
        high_risk = predictions_df[predictions_df['risk_category'] == 'High'].copy()
        high_risk = high_risk.sort_values('risk_score', ascending=False).head(top_n)
        
        return high_risk
    
    def run(
        self,
        prediction_date: Optional[str] = None,
        model_name: str = 'xgboost',
        save_to_db: bool = True,
        save_to_file: bool = True
    ):
        """
        Run batch prediction pipeline
        
        Args:
            prediction_date: Date to predict for (default: tomorrow)
            model_name: Model to use for predictions
            save_to_db: Whether to save to database
            save_to_file: Whether to save to file
        """
        try:
            # Set default prediction date to tomorrow
            if prediction_date is None:
                prediction_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            
            logger.info(f"Starting batch prediction for {prediction_date}")
            
            # Load model
            self.load_model(model_name)
            
            # Generate predictions
            predictions_df = self.generate_predictions(prediction_date)
            
            if len(predictions_df) == 0:
                logger.warning("No predictions generated")
                return
            
            # Save predictions
            if save_to_db:
                self.save_predictions_to_database(predictions_df)
            
            if save_to_file:
                output_format = self.config.get('prediction.output_format', 'csv')
                self.save_predictions_to_file(predictions_df, output_format)
            
            # Show high-risk zones
            high_risk_zones = self.get_high_risk_zones(predictions_df, top_n=10)
            
            logger.info("\n" + "="*80)
            logger.info("TOP 10 HIGH-RISK ZONES")
            logger.info("="*80)
            
            for idx, row in high_risk_zones.iterrows():
                logger.info(f"\n{idx+1}. Location: ({row['latitude']:.4f}, {row['longitude']:.4f})")
                logger.info(f"   Time: {row['prediction_date']} {row['prediction_hour']:02d}:00")
                logger.info(f"   Risk Score: {row['risk_score']:.4f}")
                logger.info(f"   Category: {row['risk_category']}")
            
            logger.info("\n✓ Batch prediction complete!")
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}", exc_info=True)
            raise


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate batch predictions')
    parser.add_argument('--date', type=str, help='Prediction date (YYYY-MM-DD)')
    parser.add_argument('--model', type=str, default='xgboost', 
                       help='Model name (default: xgboost)')
    
    args = parser.parse_args()
    
    predictor = BatchPredictor()
    predictor.run(prediction_date=args.date, model_name=args.model)


if __name__ == "__main__":
    main()
