"""
Extended Model Registry for Origin vLLM.
This module provides enhanced model registry by inheriting from base vLLM classes.
"""

import sys
import os

# Add the vllm-0.5.4 directory to sys.path for importing
vllm_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'vllm-0.5.4')
if vllm_path not in sys.path:
    sys.path.insert(0, vllm_path)

# Import base classes from vllm
from vllm.model_executor.models import ModelRegistry as BaseModelRegistry
from typing import Any, Dict, List, Optional, Union, Type


class ModelRegistry(BaseModelRegistry):
    """Extended ModelRegistry with additional functionality.
    
    This class inherits from the base vLLM ModelRegistry and can be extended
    with custom functionality while maintaining full compatibility.
    """
    
    @classmethod
    def load_model_cls(cls, *args, **kwargs):
        """Load model class with potential enhancements."""
        # Call parent method
        result = super().load_model_cls(*args, **kwargs)
        # Add any custom processing here
        return result
    
    @classmethod
    def get_model_cls(cls, *args, **kwargs):
        """Get model class with potential enhancements."""
        # Call parent method
        result = super().get_model_cls(*args, **kwargs)
        # Add any custom processing here
        return result
    
    @classmethod
    def register_model(cls, *args, **kwargs):
        """Register model with potential enhancements."""
        # Call parent method
        result = super().register_model(*args, **kwargs)
        # Add any custom processing here
        return result
    
    @classmethod
    def _try_load_model_cls(cls, *args, **kwargs):
        """Try to load model class with potential enhancements."""
        # Call parent method
        result = super()._try_load_model_cls(*args, **kwargs)
        # Add any custom processing here
        return result
    
    @classmethod
    def _normalize_model_cls_name(cls, *args, **kwargs):
        """Normalize model class name with potential enhancements."""
        # Call parent method
        result = super()._normalize_model_cls_name(*args, **kwargs)
        # Add any custom processing here
        return result
    
    @classmethod
    def is_text_generation_model(cls, *args, **kwargs):
        """Check if text generation model with potential enhancements."""
        # Call parent method
        result = super().is_text_generation_model(*args, **kwargs)
        # Add any custom processing here
        return result
    
    @classmethod
    def is_multimodal_model(cls, *args, **kwargs):
        """Check if multimodal model with potential enhancements."""
        # Call parent method
        result = super().is_multimodal_model(*args, **kwargs)
        # Add any custom processing here
        return result
    
    @classmethod
    def is_embedding_model(cls, *args, **kwargs):
        """Check if embedding model with potential enhancements."""
        # Call parent method
        result = super().is_embedding_model(*args, **kwargs)
        # Add any custom processing here
        return result
