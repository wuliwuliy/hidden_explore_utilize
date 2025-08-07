"""
Configuration utilities for Origin vLLM inheritance system.
This module provides helper functions to configure and customize the inheritance behavior.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Type


class InheritanceConfig:
    """Configuration class for managing inheritance behavior."""
    
    def __init__(self):
        self.base_vllm_path: Optional[str] = None
        self.custom_extensions: Dict[str, Any] = {}
        self.enable_logging: bool = True
        self.fallback_mode: bool = True
        
    def set_base_vllm_path(self, path: str) -> None:
        """Set the path to the base vLLM installation."""
        self.base_vllm_path = path
        if path not in sys.path:
            sys.path.insert(0, path)
    
    def add_custom_extension(self, name: str, extension: Any) -> None:
        """Add a custom extension to the inheritance system."""
        self.custom_extensions[name] = extension
    
    def enable_fallback_mode(self, enable: bool = True) -> None:
        """Enable or disable fallback mode when base vLLM is not available."""
        self.fallback_mode = enable
    
    def setup_paths(self) -> bool:
        """Setup the necessary paths for inheritance to work."""
        # Get the directory containing this config file
        config_dir = Path(__file__).parent
        base_dir = config_dir.parent
        
        # Add vllm-0.5.4 path
        vllm_path = base_dir / "vllm-0.5.4"
        if vllm_path.exists():
            self.set_base_vllm_path(str(vllm_path))
            return True
        
        # Try alternative paths
        alternative_paths = [
            base_dir / "vllm",
            base_dir / "our_vllm",
            "/opt/vllm",
            "/usr/local/lib/python*/site-packages/vllm"
        ]
        
        for alt_path in alternative_paths:
            if alt_path.exists():
                self.set_base_vllm_path(str(alt_path))
                return True
        
        return False


# Global configuration instance
_config = InheritanceConfig()


def get_config() -> InheritanceConfig:
    """Get the global inheritance configuration."""
    return _config


def setup_inheritance() -> bool:
    """Setup the inheritance system."""
    return _config.setup_paths()


def create_inherited_class(base_class: Type, 
                          class_name: str,
                          custom_methods: Optional[Dict[str, Any]] = None) -> Type:
    """
    Dynamically create an inherited class with custom methods.
    
    Args:
        base_class: The base class to inherit from
        class_name: Name for the new class
        custom_methods: Dictionary of custom methods to add
        
    Returns:
        The new inherited class
    """
    if custom_methods is None:
        custom_methods = {}
    
    # Add standard inheritance behavior
    def __init__(self, *args, **kwargs):
        super(inherited_class, self).__init__(*args, **kwargs)
        if hasattr(self, '_custom_init'):
            self._custom_init()
    
    custom_methods['__init__'] = __init__
    
    # Create the new class
    inherited_class = type(class_name, (base_class,), custom_methods)
    
    return inherited_class


def get_base_class(module_path: str, class_name: str) -> Optional[Type]:
    """
    Safely import and return a base class.
    
    Args:
        module_path: Path to the module (e.g., 'vllm.engine.llm_engine')
        class_name: Name of the class to import
        
    Returns:
        The imported class or None if import fails
    """
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError:
        if _config.enable_logging:
            print(f"Failed to import {class_name} from {module_path}")
        return None


def safe_inherit(base_module_path: str, 
                base_class_name: str, 
                new_class_name: str,
                custom_methods: Optional[Dict[str, Any]] = None) -> Type:
    """
    Safely create an inherited class with fallback behavior.
    
    Args:
        base_module_path: Path to the base module
        base_class_name: Name of the base class
        new_class_name: Name for the new inherited class
        custom_methods: Custom methods to add to the class
        
    Returns:
        The inherited class or a placeholder class if base class is not available
    """
    base_class = get_base_class(base_module_path, base_class_name)
    
    if base_class is not None:
        return create_inherited_class(base_class, new_class_name, custom_methods)
    
    # Create placeholder class if base class is not available
    if _config.fallback_mode:
        def placeholder_init(self, *args, **kwargs):
            raise ImportError(f"Base class {base_class_name} from {base_module_path} is not available. "
                            f"Please install the required dependencies.")
        
        placeholder_methods = {'__init__': placeholder_init}
        if custom_methods:
            placeholder_methods.update(custom_methods)
        
        return type(new_class_name, (), placeholder_methods)
    
    else:
        raise ImportError(f"Cannot create {new_class_name}: base class {base_class_name} not available")


# Initialize the configuration
setup_inheritance()
