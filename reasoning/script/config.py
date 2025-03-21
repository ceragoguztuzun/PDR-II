# config.py
import yaml
from dataclasses import dataclass
from typing import Dict, Any, Optional
import os
from pathlib import Path

@dataclass
class TrainingConfig:
    num_epoch: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 10
    validate_every: int = 1
    num_negatives: int = 10

@dataclass
class ModelConfig:
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1

@dataclass
class FineTuningConfig:
    num_epoch: int = 50
    learning_rate: float = 1e-5
    lambda1: float = 0.01  # PRS weight
    lambda2: float = 0.01  # Expression weight

@dataclass
class Config:
    training: TrainingConfig
    model: ModelConfig
    fine_tuning: FineTuningConfig
    data_dir: str
    checkpoint_dir: str
    patient_data_dir: str
    fm_dir: str
    seed: int = 42
    device: str = "cuda"

def load_config(config_path: str = 'config.yml') -> Config:
    """
    Loads configuration from a YAML file and returns a Config object.
    """
    # Places to look for the config file
    possible_locations = [
        config_path,  # Try direct path first
        os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path),  # Script directory
        os.path.join(os.getcwd(), config_path)  # Current working directory
    ]
    
    for location in possible_locations:
        if os.path.exists(location):
            try:
                with open(location, 'r') as f:
                    config_dict = yaml.safe_load(f)
                print(f"Loaded config from: {location}")

                training_cfg = TrainingConfig(**config_dict.get('training', {}))
                model_cfg = ModelConfig(**config_dict.get('model', {}))
                fine_tuning_cfg = FineTuningConfig(**config_dict.get('fine_tuning', {}))
                
                return Config(
                    training=training_cfg,
                    model=model_cfg,
                    fine_tuning=fine_tuning_cfg,
                    data_dir=config_dict['data_dir'],
                    checkpoint_dir=config_dict['checkpoint_dir'],
                    patient_data_dir=config_dict['patient_data_dir'],
                    fm_dir=config_dict['fm_dir'],
                    seed=config_dict.get('seed', 42),
                    device=config_dict.get('device', 'cuda')
                )

                #return Config(**config_dict)
            except Exception as e:
                print(f"Error loading config from {location}: {str(e)}")
                continue
    
    # If we get here, no config file was found
    raise FileNotFoundError(
        f"Could not find config file '{config_path}' in any of these locations:\n" +
        "\n".join(possible_locations)
    )
