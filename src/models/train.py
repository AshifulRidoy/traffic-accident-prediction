"""
Model Training Pipeline
Trains multiple models with MLflow experiment tracking
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any, Tuple
from datetime import datetime
import joblib
import json

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
import xgboost as xgb
import lightgbm as lgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.database import get_db_manager
from src.utils.config import get_config
from src.utils.logging_utils import setup_logger

logger = setup_logger(__name__, "logs/training.log")


class ModelTrainer:
    """Train and evaluate accident prediction models"""

    def __init__(self, config_path: str = None):
        """
        Initialize model trainer

        Args:
            config_path: Path to config.yaml
        """
        self.config = get_config(config_path)
        self.db = get_db_manager(config_path)

        self.training_config = self.config.training
        self.mlflow_config = self.config.mlflow

        # Setup MLflow
        mlflow.set_tracking_uri(self.mlflow_config["tracking_uri"])
        mlflow.set_experiment(self.mlflow_config["experiment_name"])

        self.models_dir = Path(self.config.paths["models"])
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def load_features(self) -> pd.DataFrame:
        """Load engineered features from database"""
        logger.info("Loading features from database...")

        # Get positive and negative samples separately for balanced dataset
        query_positive = """
            SELECT * FROM features_daily 
            WHERE accident_occurred_next_24h = 1
            LIMIT 50000;
        """

        query_negative = """
            SELECT * FROM features_daily 
            WHERE accident_occurred_next_24h = 0
            ORDER BY RANDOM()
            LIMIT 200000;
        """

        df_pos = self.db.read_sql(query_positive)
        df_neg = self.db.read_sql(query_negative)

        df = pd.concat([df_pos, df_neg], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

        logger.info(f"Loaded {len(df):,} feature records")
        logger.info(
            f"Positive samples: {len(df_pos):,}, Negative samples: {len(df_neg):,}"
        )
        return df

    def prepare_train_test_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data using stratified sampling instead of time-based"""
        from sklearn.model_selection import train_test_split

        logger.info("Splitting data (stratified)...")

        # First split: train vs (val+test)
        train_df, temp_df = train_test_split(
            df,
            test_size=0.3,
            random_state=42,
            stratify=df["accident_occurred_next_24h"],
        )

        # Second split: val vs test
        val_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=42,
            stratify=temp_df["accident_occurred_next_24h"],
        )

        logger.info(f"Train set: {len(train_df):,} samples")
        logger.info(f"Val set: {len(val_df):,} samples")
        logger.info(f"Test set: {len(test_df):,} samples")

        return train_df, val_df, test_df

    def prepare_features_and_target(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple:
        """
        Prepare feature matrices and target vectors

        Args:
            train_df, val_df, test_df: DataFrames for each split

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, feature_names)
        """
        logger.info("Preparing feature matrices...")

        # Target variable
        target_col = self.training_config["target"]

        # Columns to exclude from features
        exclude_cols = [
            "id",
            "feature_date",
            "location_id",
            "latitude",
            "longitude",
            target_col,
            "created_at",
            "weather_condition",  # Categorical, handle separately
        ]

        # Get feature columns
        feature_cols = [col for col in train_df.columns if col not in exclude_cols]

        # Handle categorical features
        categorical_cols = ["road_type"]
        for col in categorical_cols:
            if col in feature_cols:
                # One-hot encode
                train_dummies = pd.get_dummies(
                    train_df[col], prefix=col, drop_first=True
                )
                val_dummies = pd.get_dummies(val_df[col], prefix=col, drop_first=True)
                test_dummies = pd.get_dummies(test_df[col], prefix=col, drop_first=True)

                # Ensure all sets have same columns
                all_cols = (
                    set(train_dummies.columns)
                    | set(val_dummies.columns)
                    | set(test_dummies.columns)
                )
                for dummy_col in all_cols:
                    if dummy_col not in train_dummies.columns:
                        train_dummies[dummy_col] = 0
                    if dummy_col not in val_dummies.columns:
                        val_dummies[dummy_col] = 0
                    if dummy_col not in test_dummies.columns:
                        test_dummies[dummy_col] = 0

                # Add to dataframes
                train_df = pd.concat([train_df, train_dummies], axis=1)
                val_df = pd.concat([val_df, val_dummies], axis=1)
                test_df = pd.concat([test_df, test_dummies], axis=1)

                # Update feature columns
                feature_cols.remove(col)
                feature_cols.extend(train_dummies.columns.tolist())

        # Extract features and target
        X_train = train_df[feature_cols].fillna(0).values
        X_val = val_df[feature_cols].fillna(0).values
        X_test = test_df[feature_cols].fillna(0).values

        y_train = train_df[target_col].values
        y_val = val_df[target_col].values
        y_test = test_df[target_col].values

        logger.info(f"Feature matrix shape: {X_train.shape}")
        logger.info(f"Number of features: {len(feature_cols)}")
        logger.info(f"Train positive rate: {y_train.mean():.2%}")
        logger.info(f"Val positive rate: {y_val.mean():.2%}")
        logger.info(f"Test positive rate: {y_test.mean():.2%}")

        return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols

    def evaluate_model(
        self, model: Any, X: np.ndarray, y: np.ndarray, dataset_name: str = "test"
    ) -> Dict[str, float]:
        """
        Evaluate model performance

        Args:
            model: Trained model
            X: Feature matrix
            y: True labels
            dataset_name: Name of dataset (for logging)

        Returns:
            Dictionary of metrics
        """
        # Get predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]

        # Calculate metrics
        metrics = {
            f"{dataset_name}_auc": roc_auc_score(y, y_pred_proba),
            f"{dataset_name}_precision": precision_score(y, y_pred),
            f"{dataset_name}_recall": recall_score(y, y_pred),
            f"{dataset_name}_f1": f1_score(y, y_pred),
            f"{dataset_name}_accuracy": accuracy_score(y, y_pred),
        }

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        metrics[f"{dataset_name}_true_negatives"] = int(cm[0, 0])
        metrics[f"{dataset_name}_false_positives"] = int(cm[0, 1])
        metrics[f"{dataset_name}_false_negatives"] = int(cm[1, 0])
        metrics[f"{dataset_name}_true_positives"] = int(cm[1, 1])

        # Log metrics
        logger.info(f"\n{dataset_name.upper()} Metrics:")
        logger.info(f"  AUC-ROC: {metrics[f'{dataset_name}_auc']:.4f}")
        logger.info(f"  Precision: {metrics[f'{dataset_name}_precision']:.4f}")
        logger.info(f"  Recall: {metrics[f'{dataset_name}_recall']:.4f}")
        logger.info(f"  F1: {metrics[f'{dataset_name}_f1']:.4f}")
        logger.info(f"  Accuracy: {metrics[f'{dataset_name}_accuracy']:.4f}")

        return metrics

    def train_logistic_regression(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        feature_names: list,
    ):
        """Train Logistic Regression model"""
        logger.info("Training Logistic Regression...")

        with mlflow.start_run(run_name="logistic_regression"):
            # Get hyperparameters
            params = self.training_config["models"]["logistic_regression"]["params"]

            # Train model
            model = LogisticRegression(**params)
            model.fit(X_train, y_train)

            # Evaluate
            train_metrics = self.evaluate_model(model, X_train, y_train, "train")
            val_metrics = self.evaluate_model(model, X_val, y_val, "val")
            test_metrics = self.evaluate_model(model, X_test, y_test, "test")

            # Log parameters and metrics
            mlflow.log_params(params)
            mlflow.log_metrics({**train_metrics, **val_metrics, **test_metrics})

            # Log model
            mlflow.sklearn.log_model(model, "model")

            # Save model locally
            model_path = self.models_dir / "logistic_regression_model.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved model to {model_path}")

            return model, {**test_metrics}

    def train_random_forest(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        feature_names: list,
    ):
        """Train Random Forest model"""
        logger.info("Training Random Forest...")

        with mlflow.start_run(run_name="random_forest"):
            # Get hyperparameters
            params = self.training_config["models"]["random_forest"]["params"]

            # Train model
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)

            # Evaluate
            train_metrics = self.evaluate_model(model, X_train, y_train, "train")
            val_metrics = self.evaluate_model(model, X_val, y_val, "val")
            test_metrics = self.evaluate_model(model, X_test, y_test, "test")

            # Log parameters and metrics
            mlflow.log_params(params)
            mlflow.log_metrics({**train_metrics, **val_metrics, **test_metrics})

            # Log feature importance
            feature_importance = pd.DataFrame(
                {"feature": feature_names, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)

            feature_importance.to_csv("feature_importance_rf.csv", index=False)
            mlflow.log_artifact("feature_importance_rf.csv")

            # Log model
            mlflow.sklearn.log_model(model, "model")

            # Save model locally
            model_path = self.models_dir / "random_forest_model.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved model to {model_path}")

            return model, {**test_metrics}

    def train_xgboost(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        feature_names: list,
    ):
        """Train XGBoost model"""
        logger.info("Training XGBoost...")

        with mlflow.start_run(run_name="xgboost"):
            # Get hyperparameters
            params = self.training_config["models"]["xgboost"]["params"]

            # Train model
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            # Evaluate
            train_metrics = self.evaluate_model(model, X_train, y_train, "train")
            val_metrics = self.evaluate_model(model, X_val, y_val, "val")
            test_metrics = self.evaluate_model(model, X_test, y_test, "test")

            # Log parameters and metrics
            mlflow.log_params(params)
            mlflow.log_metrics({**train_metrics, **val_metrics, **test_metrics})

            # Log feature importance
            feature_importance = pd.DataFrame(
                {"feature": feature_names, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)

            feature_importance.to_csv("feature_importance_xgb.csv", index=False)
            mlflow.log_artifact("feature_importance_xgb.csv")

            # Log model
            mlflow.xgboost.log_model(model, "model")

            # Save model locally
            model_path = self.models_dir / "xgboost_model.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved model to {model_path}")

            return model, {**test_metrics}

    def train_lightgbm(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        feature_names: list,
    ):
        """Train LightGBM model"""
        logger.info("Training LightGBM...")

        with mlflow.start_run(run_name="lightgbm"):
            # Get hyperparameters
            params = self.training_config["models"]["lightgbm"]["params"]

            # Train model
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.log_evaluation(period=0)],
            )

            # Evaluate
            train_metrics = self.evaluate_model(model, X_train, y_train, "train")
            val_metrics = self.evaluate_model(model, X_val, y_val, "val")
            test_metrics = self.evaluate_model(model, X_test, y_test, "test")

            # Log parameters and metrics
            mlflow.log_params(params)
            mlflow.log_metrics({**train_metrics, **val_metrics, **test_metrics})

            # Log feature importance
            feature_importance = pd.DataFrame(
                {"feature": feature_names, "importance": model.feature_importances_}
            ).sort_values("importance", ascending=False)

            feature_importance.to_csv("feature_importance_lgb.csv", index=False)
            mlflow.log_artifact("feature_importance_lgb.csv")

            # Log model
            mlflow.lightgbm.log_model(model, "model")

            # Save model locally
            model_path = self.models_dir / "lightgbm_model.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved model to {model_path}")

            return model, {**test_metrics}

    def save_model_metadata(self, model_name: str, metrics: Dict, feature_names: list):
        """Save model metadata to database"""
        logger.info(f"Saving model metadata for {model_name}...")

        # Get MLflow run ID
        run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None

        metadata = {
            "model_name": model_name,
            "model_version": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "model_type": model_name,
            "training_date": datetime.now(),
            "auc_score": metrics.get("test_auc", 0),
            "precision_score": metrics.get("test_precision", 0),
            "recall_score": metrics.get("test_recall", 0),
            "f1_score": metrics.get("test_f1", 0),
            "accuracy_score": metrics.get("test_accuracy", 0),
            "features_used": feature_names,
            "model_path": str(self.models_dir / f"{model_name}_model.pkl"),
            "mlflow_run_id": run_id,
            "is_production": False,
        }

        # Convert to DataFrame for database insertion
        df = pd.DataFrame([metadata])

        # Handle array column
        df["features_used"] = df["features_used"].apply(lambda x: x)

        self.db.to_sql(df, "model_metadata", if_exists="append", index=False)

        logger.info("✓ Model metadata saved")

    def run(self):
        """Run complete model training pipeline"""
        try:
            # Load features
            df = self.load_features()

            # Split data
            train_df, val_df, test_df = self.prepare_train_test_split(df)

            # Prepare features and target
            X_train, X_val, X_test, y_train, y_val, y_test, feature_names = (
                self.prepare_features_and_target(train_df, val_df, test_df)
            )

            # Train all enabled models
            results = {}

            if self.training_config["models"]["logistic_regression"]["enabled"]:
                model, metrics = self.train_logistic_regression(
                    X_train, X_val, X_test, y_train, y_val, y_test, feature_names
                )
                results["logistic_regression"] = metrics
                self.save_model_metadata("logistic_regression", metrics, feature_names)

            if self.training_config["models"]["random_forest"]["enabled"]:
                model, metrics = self.train_random_forest(
                    X_train, X_val, X_test, y_train, y_val, y_test, feature_names
                )
                results["random_forest"] = metrics
                self.save_model_metadata("random_forest", metrics, feature_names)

            if self.training_config["models"]["xgboost"]["enabled"]:
                model, metrics = self.train_xgboost(
                    X_train, X_val, X_test, y_train, y_val, y_test, feature_names
                )
                results["xgboost"] = metrics
                self.save_model_metadata("xgboost", metrics, feature_names)

            if self.training_config["models"]["lightgbm"]["enabled"]:
                model, metrics = self.train_lightgbm(
                    X_train, X_val, X_test, y_train, y_val, y_test, feature_names
                )
                results["lightgbm"] = metrics
                self.save_model_metadata("lightgbm", metrics, feature_names)

            # Print summary
            logger.info("\n" + "=" * 80)
            logger.info("TRAINING SUMMARY")
            logger.info("=" * 80)

            for model_name, metrics in results.items():
                logger.info(f"\n{model_name.upper()}:")
                logger.info(f"  AUC: {metrics['test_auc']:.4f}")
                logger.info(f"  Precision: {metrics['test_precision']:.4f}")
                logger.info(f"  Recall: {metrics['test_recall']:.4f}")
                logger.info(f"  F1: {metrics['test_f1']:.4f}")

            # Find best model by AUC
            best_model = max(results.items(), key=lambda x: x[1]["test_auc"])
            logger.info(
                f"\n🏆 Best model: {best_model[0].upper()} (AUC: {best_model[1]['test_auc']:.4f})"
            )

            logger.info("\n✓ Model training complete!")
            logger.info(f"View experiments at: {self.mlflow_config['tracking_uri']}")

        except Exception as e:
            logger.error(f"Error during training: {e}", exc_info=True)
            raise


def main():
    """Main function"""
    trainer = ModelTrainer()
    trainer.run()


if __name__ == "__main__":
    main()
