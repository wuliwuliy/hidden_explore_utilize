"""
Extended Async LLM Engine for Origin vLLM.
This module provides enhanced async LLM engine by inheriting from base vLLM classes.
"""

from ..config import safe_inherit

# Define async method wrappers
async def _enhanced_generate(self, *args, **kwargs):
    """Enhanced generate method."""
    result = super(AsyncLLMEngine, self).generate(*args, **kwargs)
    async for item in result:
        yield item

async def _enhanced_add_request(self, *args, **kwargs):
    """Enhanced add request method."""
    return await super(AsyncLLMEngine, self).add_request(*args, **kwargs)

async def _enhanced_abort(self, *args, **kwargs):
    """Enhanced abort method."""
    return await super(AsyncLLMEngine, self).abort(*args, **kwargs)

async def _enhanced_get_model_config(self, *args, **kwargs):
    """Enhanced get model config method."""
    return await super(AsyncLLMEngine, self).get_model_config(*args, **kwargs)

async def _enhanced_get_tokenizer(self, *args, **kwargs):
    """Enhanced get tokenizer method."""
    return await super(AsyncLLMEngine, self).get_tokenizer(*args, **kwargs)

async def _enhanced_start_background_loop(self, *args, **kwargs):
    """Enhanced start background loop method."""
    return await super(AsyncLLMEngine, self).start_background_loop(*args, **kwargs)

async def _enhanced_check_health(self, *args, **kwargs):
    """Enhanced check health method."""
    return await super(AsyncLLMEngine, self).check_health(*args, **kwargs)

# Safely create inherited AsyncLLMEngine class
AsyncLLMEngine = safe_inherit(
    'vllm.engine.async_llm_engine',
    'AsyncLLMEngine',
    'AsyncLLMEngine',
    {
        '_custom_init': lambda self: setattr(self, '_custom_initialized', True),
        'generate': _enhanced_generate,
        'add_request': _enhanced_add_request,
        'abort': _enhanced_abort,
        'get_model_config': _enhanced_get_model_config,
        'get_tokenizer': _enhanced_get_tokenizer,
        'start_background_loop': _enhanced_start_background_loop,
        'check_health': _enhanced_check_health,
    }
)
