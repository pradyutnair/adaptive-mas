"""Configuration management for ARAG."""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional


class Config:
    """Configuration manager that loads from YAML/JSON files and environment variables."""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        self._config = config_dict or {}
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        import yaml
        
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f) or {}
        
        return cls(config_dict)
    
    @classmethod
    def from_json(cls, path: str) -> "Config":
        """Load configuration from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls(config_dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports nested keys with dot notation)."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()
    
    def __getitem__(self, key: str) -> Any:
        return self.get(key)
    
    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None
