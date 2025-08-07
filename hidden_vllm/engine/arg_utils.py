"""
Extended argument utilities for Origin vLLM.
This module provides enhanced argument handling by inheriting from base vLLM classes.
"""

from ..config import safe_inherit, get_config

# Setup configuration
config = get_config()

# Safely create inherited classes
EngineArgs = safe_inherit(
    'vllm.engine.arg_utils', 
    'EngineArgs', 
    'EngineArgs',
    {
        'create_engine_config': lambda self, *args, **kwargs: self._enhanced_create_engine_config(*args, **kwargs),
        'add_cli_args': lambda self, parser, **kwargs: self._enhanced_add_cli_args(parser, **kwargs),
        '_enhanced_create_engine_config': lambda self, *args, **kwargs: super(EngineArgs, self).create_engine_config(*args, **kwargs),
        '_enhanced_add_cli_args': lambda self, parser, **kwargs: super(EngineArgs, self).add_cli_args(parser, **kwargs)
    }
)

AsyncEngineArgs = safe_inherit(
    'vllm.engine.arg_utils',
    'AsyncEngineArgs',
    'AsyncEngineArgs', 
    {
        'create_engine_config': lambda self, *args, **kwargs: self._enhanced_create_engine_config(*args, **kwargs),
        '_enhanced_create_engine_config': lambda self, *args, **kwargs: super(AsyncEngineArgs, self).create_engine_config(*args, **kwargs)
    }
)

# Import utility function
try:
    from vllm.engine.arg_utils import nullable_str
except ImportError:
    def nullable_str(val: str):
        """Fallback implementation of nullable_str."""
        if not val or val == "None":
            return None
        return val

# Import config classes
try:
    from vllm.config import (
        CacheConfig, DecodingConfig, DeviceConfig,
        EngineConfig, LoadConfig, LoRAConfig, ModelConfig,
        MultiModalConfig, ObservabilityConfig, ParallelConfig,
        PromptAdapterConfig, SchedulerConfig,
        SpeculativeConfig, TokenizerPoolConfig
    )
except ImportError:
    # Create placeholder classes if not available
    class CacheConfig: pass
    class DecodingConfig: pass
    class DeviceConfig: pass
    class EngineConfig: pass
    class LoadConfig: pass
    class LoRAConfig: pass
    class ModelConfig: pass
    class MultiModalConfig: pass
    class ObservabilityConfig: pass
    class ParallelConfig: pass
    class PromptAdapterConfig: pass
    class SchedulerConfig: pass
    class SpeculativeConfig: pass
    class TokenizerPoolConfig: pass

# Re-export for compatibility
__all__ = [
    'EngineArgs',
    'AsyncEngineArgs', 
    'nullable_str',
    'CacheConfig',
    'DecodingConfig',
    'DeviceConfig',
    'EngineConfig',
    'LoadConfig',
    'LoRAConfig',
    'ModelConfig',
    'MultiModalConfig',
    'ObservabilityConfig',
    'ParallelConfig',
    'PromptAdapterConfig',
    'SchedulerConfig',
    'SpeculativeConfig',
    'TokenizerPoolConfig'
]
