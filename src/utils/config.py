"""
Configuration management for the invoice OCR project.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager that loads from YAML files."""
    
    def __init__(self, config_dir: str = "config"):
        """
        Initialize configuration.
        
        Args:
            config_dir: Directory containing config files
        """
        self.config_dir = Path(config_dir)
        self.data_config: Dict[str, Any] = {}
        self.model_config: Dict[str, Any] = {}
        
        self._load_configs()
    
    def _load_configs(self) -> None:
        """Load all configuration files."""
        # Load data config
        data_config_path = self.config_dir / "data_config.yaml"
        if data_config_path.exists():
            with open(data_config_path, 'r') as f:
                self.data_config = yaml.safe_load(f)
            logger.info(f"Loaded data config from {data_config_path}")
        else:
            logger.warning(f"Data config not found: {data_config_path}")
        
        # Load model config
        model_config_path = self.config_dir / "model_config.yaml"
        if model_config_path.exists():
            with open(model_config_path, 'r') as f:
                self.model_config = yaml.safe_load(f)
            logger.info(f"Loaded model config from {model_config_path}")
        else:
            logger.warning(f"Model config not found: {model_config_path}")
    
    def get_data_path(self, key: str, default: Any = None) -> Any:
        """
        Get data path from configuration.
        
        Args:
            key: Dot notation key (e.g., 'paths.processed.cleaned_data')
            default: Default value if key not found
            
        Returns:
            Configured value or default
        """
        return self._get_nested(self.data_config, key, default)
    
    def get_model_param(self, key: str, default: Any = None) -> Any:
        """
        Get model parameter from configuration.
        
        Args:
            key: Dot notation key (e.g., 'training.learning_rate')
            default: Default value if key not found
            
        Returns:
            Configured value or default
        """
        return self._get_nested(self.model_config, key, default)
    
    def _get_nested(self, config_dict: Dict[str, Any], key: str, default: Any) -> Any:
        """
        Get nested value from dictionary using dot notation.
        
        Args:
            config_dict: Configuration dictionary
            key: Dot notation key
            default: Default value
            
        Returns:
            Value or default
        """
        keys = key.split('.')
        current = config_dict
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def update_data_config(self, updates: Dict[str, Any]) -> None:
        """
        Update data configuration.
        
        Args:
            updates: Dictionary of updates in dot notation
        """
        self._update_nested(self.data_config, updates)
    
    def update_model_config(self, updates: Dict[str, Any]) -> None:
        """
        Update model configuration.
        
        Args:
            updates: Dictionary of updates in dot notation
        """
        self._update_nested(self.model_config, updates)
    
    def _update_nested(self, config_dict: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """
        Update nested dictionary with dot notation keys.
        
        Args:
            config_dict: Dictionary to update
            updates: Dictionary with dot notation keys
        """
        for key, value in updates.items():
            keys = key.split('.')
            current = config_dict
            
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            current[keys[-1]] = value
    
    def save_configs(self) -> None:
        """Save configurations back to files."""
        # Save data config
        data_config_path = self.config_dir / "data_config.yaml"
        with open(data_config_path, 'w') as f:
            yaml.dump(self.data_config, f, default_flow_style=False)
        logger.info(f"Saved data config to {data_config_path}")
        
        # Save model config
        model_config_path = self.config_dir / "model_config.yaml"
        with open(model_config_path, 'w') as f:
            yaml.dump(self.model_config, f, default_flow_style=False)
        logger.info(f"Saved model config to {model_config_path}")


# Global configuration instance
_config_instance = None

def get_config(config_dir: str = "config") -> Config:
    """
    Get or create global configuration instance.
    
    Args:
        config_dir: Directory containing config files
        
    Returns:
        Config instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_dir)
    return _config_instance