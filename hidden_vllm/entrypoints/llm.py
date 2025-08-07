"""
Extended LLM entrypoint for Origin vLLM.
This module provides enhanced LLM entrypoint by inheriting from base vLLM classes.
"""

from ..config import safe_inherit

# Safely create inherited LLM class
LLM = safe_inherit(
    'vllm.entrypoints.llm',
    'LLM',
    'LLM',
    {
        '_custom_init': lambda self: setattr(self, '_custom_initialized', True),
        'generate': lambda self, *args, **kwargs: self._enhanced_generate(*args, **kwargs),
        'chat': lambda self, *args, **kwargs: self._enhanced_chat(*args, **kwargs),
        'complete': lambda self, *args, **kwargs: self._enhanced_complete(*args, **kwargs),
        'encode': lambda self, *args, **kwargs: self._enhanced_encode(*args, **kwargs),
        'get_tokenizer': lambda self, *args, **kwargs: self._enhanced_get_tokenizer(*args, **kwargs),
        'start_profile': lambda self, *args, **kwargs: self._enhanced_start_profile(*args, **kwargs),
        'stop_profile': lambda self, *args, **kwargs: self._enhanced_stop_profile(*args, **kwargs),
        
        # Enhanced method implementations
        '_enhanced_generate': lambda self, *args, **kwargs: super(LLM, self).generate(*args, **kwargs),
        '_enhanced_chat': lambda self, *args, **kwargs: super(LLM, self).chat(*args, **kwargs),
        '_enhanced_complete': lambda self, *args, **kwargs: super(LLM, self).complete(*args, **kwargs),
        '_enhanced_encode': lambda self, *args, **kwargs: super(LLM, self).encode(*args, **kwargs),
        '_enhanced_get_tokenizer': lambda self, *args, **kwargs: super(LLM, self).get_tokenizer(*args, **kwargs),
        '_enhanced_start_profile': lambda self, *args, **kwargs: super(LLM, self).start_profile(*args, **kwargs),
        '_enhanced_stop_profile': lambda self, *args, **kwargs: super(LLM, self).stop_profile(*args, **kwargs),
    }
)
