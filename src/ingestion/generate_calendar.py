"""
Calendar data generation script
Generates calendar features including holidays, weekends, etc.
"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys
import holidays

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.database import get_db_manager
from src.utils.config import get_config
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__, 'logs/ingestion.log')


class CalendarDataGenerator:
    """Generates calendar features for the prediction system"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize calendar generator
        
        Args:
            config_path: Path to config.yaml
        """
        self.config = get_config(config_path)
        self.db = get_db_manager(config_path)
        
        # US holidays
        self.us_holidays = holidays.US()
    
    def generate_calendar_data(
        self,
        start_date: str = '2016-01-01',
        end_date: str = '2024-12-31'
    ) -> pd.DataFrame:
        """
        Generate calendar data for date range
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with calendar features
        """
        logger.info(f"Generating calendar data from {start_date} to {end_date}")
        
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create DataFrame
        df = pd.DataFrame({'date': date_range})
        
        # Extract temporal features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
        df['day_name'] = df['date'].dt.day_name()
        df['is_weekend'] = df['day_of_week'].isin([5, 6])  # Saturday, Sunday
        df['quarter'] = df['date'].dt.quarter
        df['week_of_year'] = df['date'].dt.isocalendar().week
        
        # Check if date is a holiday
        df['is_holiday'] = df['date'].apply(lambda x: x in self.us_holidays)
        df['holiday_name'] = df['date'].apply(
            lambda x: self.us_holidays.get(x) if x in self.us_holidays else None
        )
        
        logger.info(f"Generated {len(df):,} calendar records")
        logger.info(f"Holidays found: {df['is_holiday'].sum()}")
        
        return df
    
    def insert_to_database(self, df: pd.DataFrame):
        """
        Insert calendar data into database
        
        Args:
            df: Calendar DataFrame
        """
        logger.info("Inserting calendar data into database...")
        
        # Check if table has data
        existing_count = self.db.get_table_row_count('calendar')
        
        if existing_count > 0:
            logger.info(f"Truncating existing {existing_count:,} calendar records...")
            self.db.truncate_table('calendar')
        
        # Insert data
        self.db.to_sql(df, 'calendar', if_exists='append', index=False)
        
        final_count = self.db.get_table_row_count('calendar')
        logger.info(f"✓ Inserted {final_count:,} calendar records")
    
    def run(self, start_date: str = None, end_date: str = None):
        """
        Run calendar data generation
        
        Args:
            start_date: Start date (defaults to 2016-01-01)
            end_date: End date (defaults to today + 1 year)
        """
        try:
            # Set default dates if not provided
            if start_date is None:
                start_date = '2016-01-01'
            
            if end_date is None:
                # Generate calendar data for 1 year into future for predictions
                end_date = (datetime.now() + timedelta(days=365)).strftime('%Y-%m-%d')
            
            # Generate calendar data
            df = self.generate_calendar_data(start_date, end_date)
            
            # Insert to database
            self.insert_to_database(df)
            
            logger.info("✓ Calendar data generation complete!")
            
        except Exception as e:
            logger.error(f"Error during calendar generation: {e}", exc_info=True)
            raise


def main():
    """Main function"""
    generator = CalendarDataGenerator()
    generator.run()


if __name__ == "__main__":
    main()
