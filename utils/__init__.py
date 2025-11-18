#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
"""
STARFlow utilities package.

This package contains various utilities for STARFlow training and inference,
organized by functionality for better maintainability.
"""

# Import everything from the original utils.py for backward compatibility
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Re-export everything from the original utils.py to maintain compatibility
import sys
import pathlib

# Add the parent directory to path to import the original utils
parent_dir = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import from new modular structure
from .common import (
    load_model_config, preprocess_text, encode_text, drop_label, add_noise,
    get_data, save_samples_unified, read_tsv, set_random_seed
)
from .model_setup import (
    setup_transformer, setup_vae, VAE, setup_encoder, 
    LookupTableTokenizer, TextEmbedder, LabelEmbdder
)
from .training import (
    CosineLRSchedule, Distributed, get_local_rank, parallelize_model, 
    save_model, save_optimizer, sync_ctx
)
from .inference import (
    FID, IS, CLIP, Metrics,
    self_denoise, apply_denoising, process_denoising, simple_denoising
)

# Define what gets exported when someone does "from utils import *"
__all__ = [
    # Configuration
    'load_model_config',
    
    # Text processing
    'preprocess_text',
    'encode_text', 
    'drop_label',
    
    # Noise
    'add_noise',
    
    # Denoising
    'self_denoise',
    'apply_denoising',
    'process_denoising',
    'simple_denoising',
    
    # Saving
    'save_samples_unified',
    
    # Training
    'CosineLRSchedule',
    'Distributed',
    'set_random_seed',
    
    # Metrics
    'FID',
    'IS', 
    'CLIP',
    'Metrics',
    
    # Models
    'setup_transformer',
    'setup_vae',
    'VAE',
    
    # Encoders
    'setup_encoder',
    'LookupTableTokenizer',
    'TextEmbedder',
    'LabelEmbdder',
    'read_tsv',
    
    # Distributed
    'parallelize_model',
    'save_model',
    'save_optimizer',
    'get_local_rank',
    'sync_ctx',
    
    # Data
    'get_data',
]