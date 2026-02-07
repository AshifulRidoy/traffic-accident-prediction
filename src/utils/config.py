"""
Configuration management utilities
"""

import yaml
from pathlib import Path
from typing import Any, Dict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration manager for the application"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to config.yaml file
        """
        if config_path is None:
            # Default to config/config.yaml in project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
        self._apply_env_overrides()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _apply_env_overrides(self):
        """Override configuration with environment variables"""
        # Database overrides
        if os.getenv('DB_HOST'):
            self._config['database']['host'] = os.getenv('DB_HOST')
        if os.getenv('DB_PORT'):
            self._config['database']['port'] = int(os.getenv('DB_PORT'))
        if os.getenv('DB_NAME'):
            self._config['database']['name'] = os.getenv('DB_NAME')
        if os.getenv('DB_USER'):
            self._config['database']['user'] = os.getenv('DB_USER')
        if os.getenv('DB_PASSWORD'):
            self._config['database']['password'] = os.getenv('DB_PASSWORD')
        
        # Weather API
        if os.getenv('WEATHER_API_KEY'):
            self._config['ingestion']['weather_api_key'] = os.getenv('WEATHER_API_KEY')
        
        # MLflow
        if os.getenv('MLFLOW_TRACKING_URI'):
            self._config['mlflow']['tracking_uri'] = os.getenv('MLFLOW_TRACKING_URI')
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Args:
            key: Configuration key (e.g., 'database.host')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration section"""
        return self._config[key]
    
    def to_dict(self) -> Dict[str, Any]:
        """Get entire configuration as dictionary"""
        return self._config.copy()
    
    @property
    def database(self) -> Dict[str, Any]:
        """Get database configuration"""
        return self._config['database']
    
    @property
    def paths(self) -> Dict[str, str]:
        """Get paths configuration"""
        return self._config['paths']
    
    @property
    def training(self) -> Dict[str, Any]:
        """Get training configuration"""
        return self._config['training']
    
    @property
    def features(self) -> Dict[str, Any]:
        """Get feature engineering configuration"""
        return self._config['features']
    
    @property
    def mlflow(self) -> Dict[str, Any]:
        """Get MLflow configuration"""
        return self._config['mlflow']
    
    @property
    def api(self) -> Dict[str, Any]:
        """Get API configuration"""
        return self._config['api']


# Singleton instance
_config = None

def get_config(config_path: str = None) -> Config:
    """
    Get or create Config singleton instance
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config
