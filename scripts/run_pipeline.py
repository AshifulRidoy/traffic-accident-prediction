#!/usr/bin/env python
"""
Main Pipeline Orchestration Script
Runs the complete end-to-end pipeline
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logging_utils import setup_logger
from src.ingestion.ingest_accidents import AccidentDataIngestion
from src.ingestion.generate_calendar import CalendarDataGenerator
from src.features.build_features import FeatureEngineer
from src.models.train import ModelTrainer
from src.models.predict import BatchPredictor

logger = setup_logger(__name__, 'logs/pipeline.log')


class Pipeline:
    """End-to-end pipeline orchestration"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize pipeline
        
        Args:
            config_path: Path to config.yaml
        """
        self.config_path = config_path
    
    def run_ingestion(self):
        """Run data ingestion"""
        logger.info("\n" + "="*80)
        logger.info("STEP 1: DATA INGESTION")
        logger.info("="*80 + "\n")
        
        # Ingest accidents
        logger.info("Ingesting accident data...")
        accident_ingestion = AccidentDataIngestion(self.config_path)
        accident_ingestion.run()
        
        # Generate calendar data
        logger.info("\nGenerating calendar data...")
        calendar_gen = CalendarDataGenerator(self.config_path)
        calendar_gen.run()
        
        logger.info("\n✓ Data ingestion complete!\n")
    
    def run_feature_engineering(self):
        """Run feature engineering"""
        logger.info("\n" + "="*80)
        logger.info("STEP 2: FEATURE ENGINEERING")
        logger.info("="*80 + "\n")
        
        engineer = FeatureEngineer(self.config_path)
        engineer.run()
        
        logger.info("\n✓ Feature engineering complete!\n")
    
    def run_training(self):
        """Run model training"""
        logger.info("\n" + "="*80)
        logger.info("STEP 3: MODEL TRAINING")
        logger.info("="*80 + "\n")
        
        trainer = ModelTrainer(self.config_path)
        trainer.run()
        
        logger.info("\n✓ Model training complete!\n")
    
    def run_prediction(self, prediction_date: str = None):
        """Run batch prediction"""
        logger.info("\n" + "="*80)
        logger.info("STEP 4: BATCH PREDICTION")
        logger.info("="*80 + "\n")
        
        predictor = BatchPredictor(self.config_path)
        predictor.run(prediction_date=prediction_date)
        
        logger.info("\n✓ Batch prediction complete!\n")
    
    def run_all(self, skip_ingestion: bool = False, skip_training: bool = False):
        """
        Run complete pipeline
        
        Args:
            skip_ingestion: Skip data ingestion if data already loaded
            skip_training: Skip training if models already trained
        """
        start_time = datetime.now()
        
        logger.info("\n" + "="*80)
        logger.info("TRAFFIC ACCIDENT RISK PREDICTION - FULL PIPELINE")
        logger.info("="*80)
        logger.info(f"Start time: {start_time}")
        logger.info("="*80 + "\n")
        
        try:
            # Step 1: Data Ingestion
            if not skip_ingestion:
                self.run_ingestion()
            else:
                logger.info("Skipping data ingestion (--skip-ingestion flag)\n")
            
            # Step 2: Feature Engineering
            self.run_feature_engineering()
            
            # Step 3: Model Training
            if not skip_training:
                self.run_training()
            else:
                logger.info("Skipping model training (--skip-training flag)\n")
            
            # Step 4: Batch Prediction
            self.run_prediction()
            
            # Summary
            end_time = datetime.now()
            duration = end_time - start_time
            
            logger.info("\n" + "="*80)
            logger.info("PIPELINE COMPLETE!")
            logger.info("="*80)
            logger.info(f"Start time: {start_time}")
            logger.info(f"End time: {end_time}")
            logger.info(f"Duration: {duration}")
            logger.info("="*80 + "\n")
            
            logger.info("Next steps:")
            logger.info("1. View MLflow experiments: http://localhost:5000")
            logger.info("2. Start API server: python src/api/app.py")
            logger.info("3. Access API docs: http://localhost:8000/docs")
            
        except Exception as e:
            logger.error(f"\n❌ Pipeline failed: {e}", exc_info=True)
            raise


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Run Traffic Accident Risk Prediction Pipeline'
    )
    
    parser.add_argument(
        '--step',
        choices=['ingestion', 'features', 'training', 'prediction', 'all'],
        default='all',
        help='Pipeline step to run (default: all)'
    )
    
    parser.add_argument(
        '--skip-ingestion',
        action='store_true',
        help='Skip data ingestion step (use if data already loaded)'
    )
    
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip model training step (use if models already trained)'
    )
    
    parser.add_argument(
        '--prediction-date',
        type=str,
        help='Date for prediction (YYYY-MM-DD, default: tomorrow)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config.yaml file'
    )
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = Pipeline(config_path=args.config)
    
    # Run selected step
    if args.step == 'ingestion':
        pipeline.run_ingestion()
    elif args.step == 'features':
        pipeline.run_feature_engineering()
    elif args.step == 'training':
        pipeline.run_training()
    elif args.step == 'prediction':
        pipeline.run_prediction(prediction_date=args.prediction_date)
    elif args.step == 'all':
        pipeline.run_all(
            skip_ingestion=args.skip_ingestion,
            skip_training=args.skip_training
        )


if __name__ == "__main__":
    main()
