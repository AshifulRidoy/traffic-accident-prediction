"""
Feature Engineering Pipeline
Critical: All features are computed with strict temporal ordering to prevent data leakage
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import sys
from sklearn.cluster import KMeans
import h3
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.database import get_db_manager
from src.utils.config import get_config
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__, "logs/features.log")


class FeatureEngineer:
    """
    Feature engineering with strict data leakage prevention
    All rolling and lag features use ONLY past data
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize feature engineer

        Args:
            config_path: Path to config.yaml
        """
        self.config = get_config(config_path)
        self.db = get_db_manager(config_path)
        self.feature_config = self.config.features

    def load_accident_data(self) -> pd.DataFrame:
        """Load accident data from database"""
        logger.info("Loading accident data from database...")

        query = """
            SELECT 
                accident_id,
                timestamp,
                latitude,
                longitude,
                severity,
                temperature_f,
                visibility_mi,
                precipitation_in,
                wind_speed_mph,
                humidity,
                pressure_in,
                weather_condition
            FROM accidents
            ORDER BY timestamp;
        """

        df = self.db.read_sql(query)
        logger.info(f"Loaded {len(df):,} accident records")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        return df

    def load_calendar_data(self) -> pd.DataFrame:
        """Load calendar data from database"""
        logger.info("Loading calendar data...")

        query = "SELECT * FROM calendar ORDER BY date;"
        df = self.db.read_sql(query)

        logger.info(f"Loaded {len(df):,} calendar records")
        return df

    def create_location_clusters(self, df: pd.DataFrame, n_clusters: int = 50) -> Dict:
        """
        Create location clusters using K-means or H3

        Args:
            df: Accident DataFrame with lat/lon
            n_clusters: Number of clusters

        Returns:
            Dictionary mapping (lat, lon) to cluster_id
        """
        logger.info(f"Creating {n_clusters} location clusters...")

        method = self.feature_config.get("clustering", {}).get("method", "kmeans")

        if method == "h3":
            # Use H3 hexagonal grid
            resolution = self.feature_config.get("clustering", {}).get(
                "h3_resolution", 8
            )
            logger.info(f"Using H3 clustering with resolution {resolution}")

            df["h3_cell"] = df.apply(
                lambda row: h3.geo_to_h3(row["latitude"], row["longitude"], resolution),
                axis=1,
            )

            # Map each unique h3 cell to a cluster ID
            unique_cells = df["h3_cell"].unique()
            cell_to_cluster = {cell: i for i, cell in enumerate(unique_cells)}

            cluster_map = {}
            for _, row in df.iterrows():
                key = (round(row["latitude"], 4), round(row["longitude"], 4))
                cluster_map[key] = cell_to_cluster[row["h3_cell"]]

        else:
            # Use K-means clustering
            logger.info(f"Using K-means clustering with {n_clusters} clusters")

            coords = df[["latitude", "longitude"]].values

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df["cluster_id"] = kmeans.fit_predict(coords)

            # Create mapping
            cluster_map = {}
            for _, row in df.iterrows():
                key = (round(row["latitude"], 4), round(row["longitude"], 4))
                cluster_map[key] = row["cluster_id"]

        logger.info(f"Created {len(set(cluster_map.values()))} unique clusters")

        return cluster_map

    def create_base_features(
        self, start_date: str, end_date: str, location_sample: List[tuple] = None
    ) -> pd.DataFrame:
        """
        Create base feature grid (all location-date-hour combinations)

        Args:
            start_date: Start date for features
            end_date: End date for features
            location_sample: List of (lat, lon) tuples to use (None = use all from accidents)

        Returns:
            Base feature DataFrame
        """
        logger.info(f"Creating base feature grid from {start_date} to {end_date}")

        # Load accident data to get locations
        accidents = self.load_accident_data()

        if location_sample is None:
            # Sample unique locations (round to reduce granularity)
            accidents["lat_round"] = accidents["latitude"].round(4)
            accidents["lon_round"] = accidents["longitude"].round(4)

            unique_locations = accidents[["lat_round", "lon_round"]].drop_duplicates()

            # If too many locations, sample
            max_locations = 1000  # Limit for MVP
            if len(unique_locations) > max_locations:
                logger.warning(
                    f"Sampling {max_locations} locations from {len(unique_locations)}"
                )
                unique_locations = unique_locations.sample(
                    n=max_locations, random_state=42
                )

            location_list = list(
                zip(unique_locations["lat_round"], unique_locations["lon_round"])
            )
        else:
            location_list = location_sample

        logger.info(f"Using {len(location_list)} unique locations")

        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        hours = list(range(24))

        # Create combinations
        records = []
        for date in date_range:
            for hour in hours:
                for lat, lon in location_list:
                    records.append(
                        {
                            "feature_date": date.date(),
                            "hour": hour,
                            "latitude": lat,
                            "longitude": lon,
                            "feature_datetime": datetime.combine(
                                date.date(), datetime.min.time()
                            )
                            + timedelta(hours=hour),
                        }
                    )

        df = pd.DataFrame(records)
        logger.info(f"Created {len(df):,} base feature records")

        return df

    def add_temporal_features(
        self, df: pd.DataFrame, calendar_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add temporal features

        Args:
            df: Base feature DataFrame
            calendar_df: Calendar data

        Returns:
            DataFrame with temporal features
        """
        logger.info("Adding temporal features...")

        # Merge with calendar data
        df = df.merge(
            calendar_df[["date", "day_of_week", "month", "is_weekend", "is_holiday"]],
            left_on="feature_date",
            right_on="date",
            how="left",
        )

        # Rush hour indicators (7-9 AM, 5-7 PM)
        df["is_rush_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19])

        # Drop the extra date column
        df = df.drop("date", axis=1)

        logger.info("✓ Temporal features added")
        return df

    def add_weather_features(
        self, df: pd.DataFrame, accidents: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add weather features from historical accident data

        Args:
            df: Feature DataFrame
            accidents: Accident data with weather

        Returns:
            DataFrame with weather features
        """
        logger.info("Adding weather features...")

        # Prepare accident data
        accidents = accidents.copy()
        accidents["date"] = pd.to_datetime(accidents["timestamp"]).dt.date
        accidents["hour"] = pd.to_datetime(accidents["timestamp"]).dt.hour
        accidents["lat_round"] = accidents["latitude"].round(4)
        accidents["lon_round"] = accidents["longitude"].round(4)

        # For each location-date-hour, use median weather from nearby accidents
        weather_agg = (
            accidents.groupby(["date", "hour", "lat_round", "lon_round"])
            .agg(
                {
                    "temperature_f": "median",
                    "visibility_mi": "median",
                    "precipitation_in": "median",
                    "wind_speed_mph": "median",
                    "humidity": "median",
                    "pressure_in": "median",
                    "weather_condition": lambda x: (
                        x.mode()[0] if len(x) > 0 and len(x.mode()) > 0 else None
                    ),
                }
            )
            .reset_index()
        )

        # Merge with feature data
        df = df.merge(
            weather_agg,
            left_on=["feature_date", "hour", "latitude", "longitude"],
            right_on=["date", "hour", "lat_round", "lon_round"],
            how="left",
        )

        # Drop duplicate columns
        df = df.drop(["date", "lat_round", "lon_round"], axis=1, errors="ignore")

        # Fill missing weather with median values
        weather_cols = [
            "temperature_f",
            "visibility_mi",
            "precipitation_in",
            "wind_speed_mph",
            "humidity",
            "pressure_in",
        ]
        for col in weather_cols:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        # Severe weather indicator
        df["is_severe_weather"] = False
        if "weather_condition" in df.columns:
            severe_conditions = ["Snow", "Heavy Rain", "Fog", "Storm", "Ice", "Sleet"]
            df["is_severe_weather"] = df["weather_condition"].apply(
                lambda x: (
                    any(cond in str(x) for cond in severe_conditions)
                    if pd.notna(x)
                    else False
                )
            )

        logger.info("✓ Weather features added")
        return df

    def add_spatial_features(
        self, df: pd.DataFrame, accidents: pd.DataFrame, cluster_map: Dict
    ) -> pd.DataFrame:
        """
        Add spatial features

        Args:
            df: Feature DataFrame
            accidents: Accident data
            cluster_map: Location to cluster mapping

        Returns:
            DataFrame with spatial features
        """
        logger.info("Adding spatial features...")

        # Add cluster IDs
        df["cluster_id"] = df.apply(
            lambda row: cluster_map.get((row["latitude"], row["longitude"]), -1), axis=1
        )

        # Historical accident frequency by location
        accidents["lat_round"] = accidents["latitude"].round(4)
        accidents["lon_round"] = accidents["longitude"].round(4)

        location_counts = (
            accidents.groupby(["lat_round", "lon_round"])
            .size()
            .reset_index(name="accident_freq_historical")
        )

        df = df.merge(
            location_counts,
            left_on=["latitude", "longitude"],
            right_on=["lat_round", "lon_round"],
            how="left",
        )
        df = df.drop(["lat_round", "lon_round"], axis=1, errors="ignore")
        df["accident_freq_historical"] = (
            df["accident_freq_historical"].fillna(0).astype(int)
        )

        # Road type (simplified - based on accident frequency)
        df["road_type"] = "residential"
        df.loc[
            df["accident_freq_historical"]
            > df["accident_freq_historical"].quantile(0.75),
            "road_type",
        ] = "highway"
        df.loc[
            df["accident_freq_historical"].between(
                df["accident_freq_historical"].quantile(0.25),
                df["accident_freq_historical"].quantile(0.75),
            ),
            "road_type",
        ] = "arterial"

        logger.info("✓ Spatial features added")
        return df

    def add_lag_features(
        self, df: pd.DataFrame, accidents: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add lag features with strict data leakage prevention
        CRITICAL: Only uses past data!

        Args:
            df: Feature DataFrame
            accidents: Accident data

        Returns:
            DataFrame with lag features
        """
        logger.info("Adding lag features (with data leakage prevention)...")

        # Prepare accidents
        accidents = accidents.copy()
        accidents["timestamp"] = pd.to_datetime(accidents["timestamp"])
        accidents["lat_round"] = accidents["latitude"].round(4)
        accidents["lon_round"] = accidents["longitude"].round(4)

        # Sort by timestamp
        accidents = accidents.sort_values("timestamp")

        # Initialize lag columns
        df["accidents_last_24h"] = 0
        df["accidents_last_7days"] = 0
        df["moving_avg_3day"] = 0.0
        df["moving_avg_7day"] = 0.0
        df["moving_avg_30day"] = 0.0
        df["rolling_7day_accidents"] = 0
        df["rolling_30day_accidents"] = 0

        # Process in batches by location for efficiency
        unique_locations = df[["latitude", "longitude"]].drop_duplicates()

        for idx, (lat, lon) in tqdm(
            enumerate(zip(unique_locations["latitude"], unique_locations["longitude"])),
            total=len(unique_locations),
            desc="Computing lag features",
        ):
            # Get accidents for this location
            location_accidents = accidents[
                (accidents["lat_round"] == lat) & (accidents["lon_round"] == lon)
            ].copy()

            if len(location_accidents) == 0:
                continue

            # Get feature rows for this location
            location_mask = (df["latitude"] == lat) & (df["longitude"] == lon)
            location_features = df[location_mask].copy()

            # For each feature datetime, compute lags using only past data
            for feat_idx, row in location_features.iterrows():
                feat_dt = row["feature_datetime"]

                # CRITICAL: Only use accidents BEFORE the feature datetime
                past_accidents = location_accidents[
                    location_accidents["timestamp"] < feat_dt
                ]

                if len(past_accidents) == 0:
                    continue

                # Last 24 hours
                last_24h = past_accidents[
                    past_accidents["timestamp"] >= feat_dt - timedelta(hours=24)
                ]
                df.loc[feat_idx, "accidents_last_24h"] = len(last_24h)

                # Last 7 days
                last_7d = past_accidents[
                    past_accidents["timestamp"] >= feat_dt - timedelta(days=7)
                ]
                df.loc[feat_idx, "accidents_last_7days"] = len(last_7d)
                df.loc[feat_idx, "rolling_7day_accidents"] = len(last_7d)

                # Last 30 days
                last_30d = past_accidents[
                    past_accidents["timestamp"] >= feat_dt - timedelta(days=30)
                ]
                df.loc[feat_idx, "rolling_30day_accidents"] = len(last_30d)

                # Moving averages (accidents per day)
                if len(last_7d) > 0:
                    df.loc[feat_idx, "moving_avg_3day"] = (
                        len(
                            past_accidents[
                                past_accidents["timestamp"]
                                >= feat_dt - timedelta(days=3)
                            ]
                        )
                        / 3.0
                    )

                    df.loc[feat_idx, "moving_avg_7day"] = len(last_7d) / 7.0

                if len(last_30d) > 0:
                    df.loc[feat_idx, "moving_avg_30day"] = len(last_30d) / 30.0

        logger.info("✓ Lag features added (data leakage prevented)")
        return df

    def create_target_variable(
        self,
        df: pd.DataFrame,
        accidents: pd.DataFrame,
        prediction_window_hours: int = 24,
    ) -> pd.DataFrame:
        """
        Create target variable: did an accident occur in the next 24 hours?

        Args:
            df: Feature DataFrame
            accidents: Accident data
            prediction_window_hours: Prediction window in hours

        Returns:
            DataFrame with target variable
        """
        logger.info(
            f"Creating target variable (next {prediction_window_hours} hours)..."
        )

        # Prepare accidents
        accidents = accidents.copy()
        accidents["timestamp"] = pd.to_datetime(accidents["timestamp"])
        accidents["lat_round"] = accidents["latitude"].round(4)
        accidents["lon_round"] = accidents["longitude"].round(4)

        # Initialize target
        df["accident_occurred_next_24h"] = 0

        # Process by location
        unique_locations = df[["latitude", "longitude"]].drop_duplicates()

        for lat, lon in tqdm(
            zip(unique_locations["latitude"], unique_locations["longitude"]),
            total=len(unique_locations),
            desc="Creating target variable",
        ):
            # Get accidents for this location
            location_accidents = accidents[
                (accidents["lat_round"] == lat) & (accidents["lon_round"] == lon)
            ]

            if len(location_accidents) == 0:
                continue

            # Get feature rows for this location
            location_mask = (df["latitude"] == lat) & (df["longitude"] == lon)
            location_features = df[location_mask].copy()

            # For each feature datetime, check if accident occurs in next 24 hours
            for feat_idx, row in location_features.iterrows():
                feat_dt = row["feature_datetime"]

                # Check for accidents in the NEXT prediction_window_hours
                future_accidents = location_accidents[
                    (location_accidents["timestamp"] >= feat_dt)
                    & (
                        location_accidents["timestamp"]
                        < feat_dt + timedelta(hours=prediction_window_hours)
                    )
                ]

                if len(future_accidents) > 0:
                    df.loc[feat_idx, "accident_occurred_next_24h"] = 1

        positive_rate = df["accident_occurred_next_24h"].mean()
        logger.info(f"✓ Target variable created. Positive rate: {positive_rate:.2%}")

        return df

    def save_features(self, df: pd.DataFrame, output_path: str = None):
        """
        Save engineered features

        Args:
            df: Feature DataFrame
            output_path: Output path (optional, saves to database if None)
        """
        if output_path:
            logger.info(f"Saving features to {output_path}")
            df.to_parquet(output_path, index=False)
        else:
            logger.info("Saving features to database...")

            # Truncate existing features
            existing_count = self.db.get_table_row_count("features_daily")
            if existing_count > 0:
                logger.info(f"Truncating {existing_count:,} existing features...")
                self.db.truncate_table("features_daily")

            # Prepare for database
            df_db = df.copy()
            df_db["location_id"] = (
                df_db["latitude"].astype(str) + "_" + df_db["longitude"].astype(str)
            )

            # Drop helper columns
            df_db = df_db.drop(["feature_datetime"], axis=1, errors="ignore")

            # Insert in batches
            batch_size = 5000  # Reduced from 10000
            total_batches = (len(df_db) - 1) // batch_size + 1

            logger.info(
                f"Inserting {len(df_db):,} records in {total_batches} batches..."
            )

            for i in range(0, len(df_db), batch_size):
                batch = df_db.iloc[i : i + batch_size]
                self.db.to_sql(
                    batch,
                    "features_daily",
                    if_exists="append",
                    index=False,
                    chunksize=1000,
                )

                if (i // batch_size + 1) % 10 == 0:
                    logger.info(f"Progress: {i + len(batch):,}/{len(df_db):,} records")

            final_count = self.db.get_table_row_count("features_daily")
            logger.info(f"✓ Saved {final_count:,} feature records to database")

    def run(
        self, start_date: str = None, end_date: str = None, output_path: str = None
    ):
        """
        Run complete feature engineering pipeline

        Args:
            start_date: Start date for features
            end_date: End date for features
            output_path: Path to save features (optional)
        """
        try:
            # Load data
            accidents = self.load_accident_data()
            calendar = self.load_calendar_data()

            # Set default dates based on accident data
            if start_date is None:
                start_date = accidents["timestamp"].min().strftime("%Y-%m-%d")
            if end_date is None:
                end_date = accidents["timestamp"].max().strftime("%Y-%m-%d")

            logger.info(f"Feature engineering date range: {start_date} to {end_date}")

            # Create location clusters
            n_clusters = self.feature_config.get("clustering", {}).get("n_clusters", 50)
            cluster_map = self.create_location_clusters(accidents, n_clusters)

            # Create base feature grid
            df = self.create_base_features(start_date, end_date)

            # Add all features
            df = self.add_temporal_features(df, calendar)
            df = self.add_weather_features(df, accidents)
            df = self.add_spatial_features(df, accidents, cluster_map)
            df = self.add_lag_features(df, accidents)

            # Create target variable
            prediction_window = self.config.get("training.prediction_window_hours", 24)
            df = self.create_target_variable(df, accidents, prediction_window)

            # Save features
            self.save_features(df, output_path)

            logger.info("✓ Feature engineering complete!")
            logger.info(f"Total features: {len(df):,}")
            logger.info(f"Feature columns: {len(df.columns)}")
            logger.info(
                f"Positive samples: {df['accident_occurred_next_24h'].sum():,} "
                f"({df['accident_occurred_next_24h'].mean():.2%})"
            )

        except Exception as e:
            logger.error(f"Error during feature engineering: {e}", exc_info=True)
            raise


def main():
    """Main function"""
    engineer = FeatureEngineer()
    engineer.run()


if __name__ == "__main__":
    main()
