"""
Accident data ingestion script
Loads US Accidents dataset from Kaggle into PostgreSQL database
"""

import pandas as pd
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.database import get_db_manager
from src.utils.config import get_config
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__, "logs/ingestion.log")


class AccidentDataIngestion:
    """Handles ingestion of accident data into database"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ingestion

        Args:
            config_path: Path to config.yaml
        """
        self.config = get_config(config_path)
        self.db = get_db_manager(config_path)

        # Get paths from config
        self.raw_data_path = Path(self.config.paths["raw_data"])
        self.accident_file = self.config.get(
            "ingestion.accident_file", "US_Accidents.csv"
        )
        self.batch_size = self.config.get("ingestion.batch_size", 10000)

    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw accident data from CSV

        Returns:
            DataFrame with accident data
        """
        file_path = self.raw_data_path / self.accident_file

        if not file_path.exists():
            raise FileNotFoundError(
                f"Accident data file not found: {file_path}\n"
                f"Please download the Kaggle US Accidents dataset and place it in {self.raw_data_path}/"
            )

        logger.info(f"Loading accident data from {file_path}")

        # Load data in chunks to handle large file
        chunks = []
        chunk_iter = pd.read_csv(file_path, chunksize=self.batch_size)

        for chunk in tqdm(chunk_iter, desc="Loading CSV"):
            chunks.append(chunk)

        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Loaded {len(df):,} accident records")

        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate accident data

        Args:
            df: Raw accident DataFrame

        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning accident data...")

        initial_count = len(df)

        # Create a copy to avoid modifying original
        df = df.copy()

        # Rename columns to match database schema
        column_mapping = {
            "ID": "accident_id",
            "Severity": "severity",
            "Start_Time": "timestamp",
            "Start_Lat": "latitude",
            "Start_Lng": "longitude",
            "Distance(mi)": "distance_mi",
            "Temperature(F)": "temperature_f",
            "Wind_Chill(F)": "wind_chill_f",
            "Humidity(%)": "humidity",
            "Pressure(in)": "pressure_in",
            "Visibility(mi)": "visibility_mi",
            "Wind_Speed(mph)": "wind_speed_mph",
            "Precipitation(in)": "precipitation_in",
            "Weather_Condition": "weather_condition",
            "Wind_Direction": "wind_direction",
            "Amenity": "amenity",
            "Bump": "bump",
            "Crossing": "crossing",
            "Give_Way": "give_way",
            "Junction": "junction",
            "No_Exit": "no_exit",
            "Railway": "railway",
            "Roundabout": "roundabout",
            "Station": "station",
            "Stop": "stop",
            "Traffic_Calming": "traffic_calming",
            "Traffic_Signal": "traffic_signal",
            "Turning_Loop": "turning_loop",
            "Sunrise_Sunset": "sunrise_sunset",
            "Civil_Twilight": "civil_twilight",
            "Nautical_Twilight": "nautical_twilight",
            "Astronomical_Twilight": "astronomical_twilight",
            "Street": "street",
            "City": "city",
            "County": "county",
            "State": "state",
            "Zipcode": "zipcode",
            "Country": "country",
            "Timezone": "timezone",
            "Airport_Code": "airport_code",
            "Weather_Timestamp": "weather_timestamp",
        }

        # Rename columns that exist
        existing_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=existing_cols)

        # Essential columns that must be present
        essential_cols = ["accident_id", "timestamp", "latitude", "longitude"]
        missing_cols = [col for col in essential_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing essential columns: {missing_cols}")

        # Remove rows with missing essential data
        df = df.dropna(subset=essential_cols)
        logger.info(
            f"Removed {initial_count - len(df):,} rows with missing essential data"
        )

        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])

        # Convert weather_timestamp if present
        if "weather_timestamp" in df.columns:
            df["weather_timestamp"] = pd.to_datetime(
                df["weather_timestamp"], errors="coerce"
            )
            df["weather_timestamp"] = df["weather_timestamp"].where(
                df["weather_timestamp"].notna(), None
            )

        # Validate latitude and longitude ranges
        df = df[
            (df["latitude"].between(-90, 90)) & (df["longitude"].between(-180, 180))
        ]

        # Convert boolean columns
        bool_columns = [
            "amenity",
            "bump",
            "crossing",
            "give_way",
            "junction",
            "no_exit",
            "railway",
            "roundabout",
            "station",
            "stop",
            "traffic_calming",
            "traffic_signal",
            "turning_loop",
        ]

        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].fillna(False).astype(bool)

        # Clean numeric columns
        numeric_cols = [
            "severity",
            "distance_mi",
            "temperature_f",
            "wind_chill_f",
            "humidity",
            "pressure_in",
            "visibility_mi",
            "wind_speed_mph",
            "precipitation_in",
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Ensure severity is integer
        if "severity" in df.columns:
            df["severity"] = df["severity"].fillna(0).astype(int)

        # Clean string columns
        string_cols = [
            "weather_condition",
            "wind_direction",
            "sunrise_sunset",
            "civil_twilight",
            "nautical_twilight",
            "astronomical_twilight",
            "street",
            "city",
            "county",
            "state",
            "zipcode",
            "country",
            "timezone",
            "airport_code",
        ]

        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).replace("nan", None)

        # Sort by timestamp
        df = df.sort_values("timestamp")

        logger.info(f"Cleaned data: {len(df):,} valid records")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        return df

    def prepare_for_database(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DataFrame for database insertion

        Args:
            df: Cleaned DataFrame

        Returns:
            DataFrame ready for database
        """
        # Select only columns that exist in database schema
        db_columns = [
            "accident_id",
            "timestamp",
            "latitude",
            "longitude",
            "severity",
            "distance_mi",
            "temperature_f",
            "wind_chill_f",
            "humidity",
            "pressure_in",
            "visibility_mi",
            "wind_speed_mph",
            "precipitation_in",
            "weather_condition",
            "wind_direction",
            "amenity",
            "bump",
            "crossing",
            "give_way",
            "junction",
            "no_exit",
            "railway",
            "roundabout",
            "station",
            "stop",
            "traffic_calming",
            "traffic_signal",
            "turning_loop",
            "sunrise_sunset",
            "civil_twilight",
            "nautical_twilight",
            "astronomical_twilight",
            "street",
            "city",
            "county",
            "state",
            "zipcode",
            "country",
            "timezone",
            "airport_code",
            "weather_timestamp",
        ]

        # Keep only columns that exist in the dataframe
        available_cols = [col for col in db_columns if col in df.columns]
        df_db = df[available_cols].copy()
        for col in df_db.columns:
            if pd.api.types.is_datetime64_any_dtype(df_db[col]):
                df_db[col] = df_db[col].where(df_db[col].notna(), None)
        if "weather_timestamp" in df_db.columns:
            df_db["weather_timestamp"] = df_db["weather_timestamp"].replace(
                ["NaT", "nat", "NAN"], None
            )
            df_db["weather_timestamp"] = df_db["weather_timestamp"].where(
                pd.notna(df_db["weather_timestamp"]), None
            )

        return df_db

    def insert_to_database(self, df: pd.DataFrame):
        """
        Insert accident data into database

        Args:
            df: DataFrame to insert
        """
        logger.info(f"Inserting {len(df):,} records into database...")

        # Check if table exists and has data
        existing_count = self.db.get_table_row_count("accidents")

        if existing_count > 0:
            logger.warning(f"Table 'accidents' already has {existing_count:,} records")
            response = input(
                "Do you want to (A)ppend, (R)eplace, or (C)ancel? [A/R/C]: "
            )

            if response.upper() == "R":
                logger.info("Truncating existing data...")
                self.db.truncate_table("accidents")
            elif response.upper() == "C":
                logger.info("Insertion cancelled")
                return

        # Insert in batches

        for i in tqdm(range(0, len(df), self.batch_size), desc="Inserting batches"):
            batch = df.iloc[i : i + self.batch_size]

            # Prepare insert query
            columns = ", ".join(batch.columns)
            placeholders = ", ".join(["%s"] * len(batch.columns))

            # Handle PostGIS geometry column separately
            insert_query = f"""
                INSERT INTO accidents ({columns}, location)
                VALUES ({placeholders}, ST_SetSRID(ST_MakePoint(%s, %s), 4326))
                ON CONFLICT (accident_id) DO NOTHING;
            """

            # Prepare data tuples
            data = []
            for _, row in batch.iterrows():
                row_values = []
                for val in row.values:
                    if pd.isna(val):
                        row_values.append(None)
                    else:
                        row_values.append(val)
                clean_values = tuple(
                    None if pd.isna(val) else val for val in row.values
                )

                values = clean_values + (row["longitude"], row["latitude"])
                data.append(values)

            # Execute batch insert
            self.db.execute_many(insert_query, data, batch_size=1000)

        # Create spatial index
        logger.info("Creating spatial index...")
        self.db.create_spatial_index("accidents", "location")

        # Optimize table
        logger.warning(
            "Skipping VACUUM ANALYZE due to memory constraints. "
            "Run it manually later if needed."
        )

        # logger.info("Optimizing table...")
        # self.db.vacuum_analyze("accidents")

        final_count = self.db.get_table_row_count("accidents")
        logger.info(
            f"✓ Successfully inserted data. Total records in database: {final_count:,}"
        )

    def run(self):
        """Run the complete ingestion pipeline"""
        try:
            # Load data
            df_raw = self.load_raw_data()

            # Clean data
            df_clean = self.clean_data(df_raw)

            # Prepare for database
            df_db = self.prepare_for_database(df_clean)

            # Insert to database
            self.insert_to_database(df_db)

            logger.info("✓ Accident data ingestion complete!")

        except Exception as e:
            logger.error(f"Error during ingestion: {e}", exc_info=True)
            raise


def main():
    """Main function"""
    ingestion = AccidentDataIngestion()
    ingestion.run()


if __name__ == "__main__":
    main()
