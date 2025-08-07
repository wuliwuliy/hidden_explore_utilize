"""Origin vLLM: Extended vLLM implementation using inheritance

This module provides extended functionality by inheriting from the base vLLM implementation.
It maintains compatibility with the original vLLM API while adding custom features.
"""

import sys
import os
import warnings

# Add the vllm-0.5.4 directory to sys.path for importing
# vllm_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'vllm-0.5.4')
# if vllm_path not in sys.path:
#     sys.path.insert(0, vllm_path)

# Core Engine Components
from vllm.engine.arg_utils import AsyncEngineArgs as BaseAsyncEngineArgs, EngineArgs as BaseEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine as BaseAsyncLLMEngine
from .engine.llm_engine import LLMEngine as BaseLLMEngine, _load_generation_config_dict
from vllm.entrypoints.llm import LLM as BaseLLM

# Configuration Classes
from vllm.config import (
    CacheConfig, DecodingConfig, DeviceConfig, EngineConfig, LoadConfig, LoRAConfig, 
    ModelConfig, MultiModalConfig, ObservabilityConfig, ParallelConfig, PromptAdapterConfig,
    SchedulerConfig, SpeculativeConfig, TokenizerPoolConfig, VisionLanguageConfig,
    _get_and_verify_dtype, _get_and_verify_max_len, get_served_model_name
)

# Core Processing Components
from vllm.core.scheduler import Scheduler, SchedulerOutputs
from vllm.engine.output_processor.interfaces import SequenceGroupOutputProcessor
from vllm.engine.output_processor.stop_checker import StopChecker
from vllm.engine.metrics import LoggingStatLogger, PrometheusStatLogger, StatLogger, Stats
# from vllm.engine.metrics_types import StatLoggerBase

# Executor and Worker Components
from vllm.executor.executor_base import ExecutorBase, ExecutorAsyncBase
from vllm.executor.ray_utils import initialize_ray_cluster
from .worker.worker import Worker, _check_if_gpu_supports_dtype
from vllm.worker.worker_base import WorkerInput
from .worker.model_runner import ModelRunner, CUDAGraphRunner, GPUModelRunnerBase, _async_h2d
from vllm.worker.model_runner_base import ModelRunnerBase, ModelRunnerInputBase
from vllm.worker.embedding_model_runner import EmbeddingModelRunner
from vllm.worker.cache_engine import CacheEngine

# Input/Output and Sequence Components
from vllm.inputs import INPUT_REGISTRY, InputRegistry, LLMInputs, PromptInputs, TextPrompt, TokensPrompt, parse_and_batch_prompt
# from vllm.inputs.preprocess import InputPreprocessor
from .outputs import CompletionOutput, EmbeddingOutput, EmbeddingRequestOutput, RequestOutput
from .sequence import (
    SamplerOutput, Sequence, SequenceData, SequenceGroup, 
    SequenceGroupMetadata, SequenceGroupOutput, SequenceOutput, SequenceStatus,
    ExecuteModelRequest, IntermediateTensors
)
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams, SamplingType

# Model Executor Components
from vllm.model_executor import SamplingMetadata, set_random_seed
from vllm.model_executor.models import ModelRegistry as BaseModelRegistry
from vllm.model_executor.model_loader import BaseModelLoader, get_architecture_class_name
from vllm.model_executor.model_loader.loader import _initialize_model
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from vllm.model_executor.model_loader.weight_utils import default_weight_loader, get_quant_config, initialize_dummy_weights
from vllm.model_executor.models.utils import is_pp_missing_parameter
from vllm.model_executor.models.interfaces import supports_lora, supports_vision
from .model_executor.models.gemma import GemmaForCausalLM

# Model Executor Layers
from vllm.model_executor.layers.linear import *
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
from vllm.model_executor.layers.activation import ScaledActivation
from vllm.model_executor.layers.sampler import (
    Sampler, _apply_penalties, 
    _apply_top_k_top_p, _apply_min_p, _sample, _get_logprobs, _build_sampler_output
)
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS, get_quantization_config
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.sampling_metadata import SamplingMetadata, SamplingTensors

# Attention Components
from vllm.attention import AttentionMetadata, get_attn_backend

# Distributed and Parallel Components
# import vllm.model_executor.parallel_utils.parallel_state as ps
import vllm.distributed.parallel_state as ps
from vllm.distributed.parallel_state import (
    get_pp_group, get_world_group, init_distributed_environment, 
    init_model_parallel_group, get_tensor_model_parallel_group,
    get_tensor_model_parallel_cpu_group
)
from vllm.distributed.communication_op import tensor_model_parallel_all_gather
from vllm.distributed.device_communicators import pynccl_utils
from vllm.distributed.device_communicators.custom_all_reduce import init_custom_ar
from vllm.distributed import (
    init_distributed_environment, set_custom_all_reduce, 
    get_tensor_model_parallel_group
)
# from vllm.model_executor.parallel_utils.custom_all_reduce import init_custom_ar as init_custom_ar_old
# from vllm.model_executor.parallel_utils.parallel_state import (
#     initialize_model_parallel, get_tensor_model_parallel_group as get_tensor_model_parallel_group_old
# )

# LoRA Components
from vllm.lora.request import LoRARequest
from vllm.lora.layers import LoRAMapping
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager

# Prompt Adapter Components
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.prompt_adapter.worker_manager import LRUCacheWorkerPromptAdapterManager

# Tokenizer Components
from vllm.transformers_utils.tokenizer import detokenize_incrementally, get_cached_tokenizer, AnyTokenizer
# from vllm.transformers_utils.tokenizers import *
from vllm.transformers_utils.tokenizer_group import TokenizerGroup
from vllm.transformers_utils.tokenizer_group.base_tokenizer_group import BaseTokenizerGroup
from vllm.transformers_utils.detokenizer import Detokenizer
from vllm.transformers_utils.config import get_config, get_hf_text_config

# Guided Decoding Components
from vllm.model_executor.guided_decoding import (
    GuidedDecodingRequest, get_local_guided_decoding_logits_processor
)
from vllm.model_executor.guided_decoding.guided_fields import LLMGuidedOptions

# Utility Components
from vllm.logger import init_logger
from vllm.utils import (
    Counter, LRUCache, make_async, is_hip, get_cpu_memory, get_nvcc_cuda_version,
    in_wsl, str_to_int_tuple, CudaMemoryProfiler, DeviceMemoryProfiler,
    is_pin_memory_available, init_cached_hf_modules, FlexibleArgumentParser,
    deprecate_kwargs, weak_bind, print_warning_once, supports_dynamo
)
from vllm.usage.usage_lib import UsageContext, is_usage_stats_enabled, usage_message
from vllm.version import __version__ as VLLM_VERSION
from vllm.tracing import SpanAttributes, SpanKind, extract_trace_context, init_tracer
from vllm.envs import envs

# Compilation Components
# from vllm.compilation.levels import CompilationLevel

# Multimodal Components
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry

# Plugin Components
# from vllm.plugins import get_torch_compile_backend

# # Import our extended classes
# from .engine.arg_utils import AsyncEngineArgs, EngineArgs
# from .engine.async_llm_engine import AsyncLLMEngine
# from .engine.llm_engine import LLMEngine, SchedulerContext, SchedulerOutputState
# from .entrypoints.llm import LLM
# from .model_executor.models import ModelRegistry
# from .version import __commit__, __version__

from .engine.output_processor.single_step import SingleStepOutputProcessor
# from .engine.llm_engine import LLMEngine
from .engine.output_processor.utils import create_output_by_sequence_group
# from .model_executor.models.gemma import GemmaMLP, GemmaAttention, GemmaDecoderLayer, GemmaModel, GemmaForCausalLM
# from .model_executor.models.gemma2 import Gemma2MLP, Gemma2Attention, Gemma2DecoderLayer, Gemma2Model, Gemma2ForCausalLM
# from .model_executor.models.llama import LlamaMLP, LlamaAttention, LlamaDecoderLayer, LlamaModel, LlamaForCausalLM
# from .model_executor.models.qwen2 import Qwen2MLP, Qwen2Attention, Qwen2DecoderLayer, Qwen2Model, Qwen2ForCausalLM
# from .outputs import CompletionOutput, EmbeddingOutput, EmbeddingRequestOutput, RequestOutput, RequestOutputFactory
# from .sequence import Logprob, SequenceStatus, SequenceStage, RequestMetrics,SequenceData,Sequence, \
#       SequenceGroup, SequenceGroupMetadata, SequenceOutput, SequenceGroupOutput, \
#         CompletionSequenceGroupOutput,EmbeddingSequenceGroupOutput , IntermediateTensors, \
#             SamplerOutput, PoolerOutput, HiddenStates, ExecuteModelRequest
# from .worker.model_runner import ModelInputForGPU, ModelInputForGPUWithSamplingMetadata, ModelInputForGPUBuilder,\
#     GPUModelRunnerBase , ModelRunner, CUDAGraphRunner
# from .worker.worker import Worker, init_worker_distributed_environment, raise_if_cache_size_invalid

# except ImportError as e:
#     warnings.warn(f"Failed to import some vLLM components: {e}. "
#                   "Make sure vLLM is properly installed and accessible.",
#                   RuntimeWarning, stacklevel=2)
    
#     # Fallback imports - just import version info
#     from .version import __commit__, __version__
    
#     # Create placeholder classes if base imports fail
#     class LLM:
#         """Placeholder LLM class when base vLLM is not available."""
#         def __init__(self, *args, **kwargs):
#             raise ImportError("Base vLLM library is not available. Please install vLLM first.")
    
#     class LLMEngine:
#         """Placeholder LLMEngine class when base vLLM is not available."""
#         def __init__(self, *args, **kwargs):
#             raise ImportError("Base vLLM library is not available. Please install vLLM first.")
    
#     class AsyncLLMEngine:
#         """Placeholder AsyncLLMEngine class when base vLLM is not available."""
#         def __init__(self, *args, **kwargs):
#             raise ImportError("Base vLLM library is not available. Please install vLLM first.")
    
#     class EngineArgs:
#         """Placeholder EngineArgs class when base vLLM is not available."""
#         def __init__(self, *args, **kwargs):
#             raise ImportError("Base vLLM library is not available. Please install vLLM first.")
    
#     class AsyncEngineArgs:
#         """Placeholder AsyncEngineArgs class when base vLLM is not available."""
#         def __init__(self, *args, **kwargs):
#             raise ImportError("Base vLLM library is not available. Please install vLLM first.")
    
#     class ModelRegistry:
#         """Placeholder ModelRegistry class when base vLLM is not available."""
#         def __init__(self, *args, **kwargs):
#             raise ImportError("Base vLLM library is not available. Please install vLLM first.")

#     # Set undefined variables as None for other imports
#     PromptInputs = None
#     TextPrompt = None
#     TokensPrompt = None
#     SamplingParams = None
#     RequestOutput = None
#     CompletionOutput = None
#     EmbeddingOutput = None
#     EmbeddingRequestOutput = None
#     PoolingParams = None
#     initialize_ray_cluster = None
#     CacheConfig = None
#     DeviceConfig = None
#     ModelConfig = None
#     ParallelConfig = None
#     SchedulerConfig = None
#     LoRAConfig = None
#     DecodingConfig = None
#     EngineConfig = None
#     LoadConfig = None
#     MultiModalConfig = None
#     ObservabilityConfig = None
#     PromptAdapterConfig = None
#     SpeculativeConfig = None
#     TokenizerPoolConfig = None
#     VisionLanguageConfig = None

__all__ = [
    # Core Engine Components (Base Classes)
    "BaseAsyncEngineArgs",
    "BaseEngineArgs",
    "BaseAsyncLLMEngine", 
    "BaseLLMEngine",
    "BaseLLM",
    "_load_generation_config_dict",
    
    # Configuration Classes
    "CacheConfig",
    "DecodingConfig",
    "DeviceConfig", 
    "EngineConfig",
    "LoadConfig",
    "LoRAConfig",
    "ModelConfig",
    "MultiModalConfig",
    "ObservabilityConfig",
    "ParallelConfig",
    "PromptAdapterConfig",
    "SchedulerConfig",
    "SpeculativeConfig",
    "TokenizerPoolConfig",
    "VisionLanguageConfig",
    "_get_and_verify_dtype",
    "_get_and_verify_max_len",
    "get_served_model_name",
    
    # Core Processing Components
    "Scheduler",
    "SchedulerOutputs",
    "SequenceGroupOutputProcessor",
    "StopChecker",
    "LoggingStatLogger",
    "PrometheusStatLogger",
    "StatLogger",
    "Stats",
    "StatLoggerBase",
    
    # Executor and Worker Components
    "ExecutorBase",
    "ExecutorAsyncBase",
    "initialize_ray_cluster",
    "Worker",
    "_check_if_gpu_supports_dtype",
    "WorkerInput",
    "ModelRunner",
    "CUDAGraphRunner",
    "GPUModelRunnerBase",
    "_async_h2d",
    "ModelRunnerBase", 
    "ModelRunnerInputBase",
    "EmbeddingModelRunner",
    "CacheEngine",
    
    # Input/Output and Sequence Components
    "INPUT_REGISTRY",
    "InputRegistry",
    "LLMInputs",
    "PromptInputs",
    "TextPrompt",
    "TokensPrompt",
    "parse_and_batch_prompt",
    "InputPreprocessor",
    "CompletionOutput",
    "EmbeddingOutput",
    "EmbeddingRequestOutput",
    "RequestOutput",
    "MultiModalData",
    "SamplerOutput",
    "Sequence",
    "SequenceData",
    "SequenceGroup",
    "SequenceGroupMetadata",
    "SequenceGroupOutput",
    "SequenceOutput",
    "SequenceStatus",
    "ExecuteModelRequest",
    "IntermediateTensors",
    "PoolingParams",
    "SamplingParams",
    "SamplingType",
    
    # Model Executor Components
    "InputMetadata",
    "SamplingMetadata", 
    "set_random_seed",
    "BaseModelRegistry",
    "BaseModelLoader",
    "get_architecture_class_name",
    "_initialize_model",
    "set_default_torch_dtype",
    "default_weight_loader",
    "get_quant_config",
    "initialize_dummy_weights",
    "is_pp_missing_parameter",
    "supports_lora",
    "supports_vision",
    "GemmaForCausalLM",
    
    # Model Executor Layers
    "VocabParallelEmbedding",
    "ParallelLMHead",
    "ScaledActivation",
    "Sampler",
    "_prune_hidden_states",
    "_apply_logits_processors",
    "_apply_penalties",
    "_apply_top_k_top_p",
    "_apply_min_p",
    "_sample",
    "_get_logprobs",
    "_build_sampler_output",
    "QUANTIZATION_METHODS",
    "get_quantization_config",
    "LogitsProcessor",
    "FusedMoE",
    "SamplingTensors",
    
    # Attention Components
    "AttentionMetadata",
    "get_attn_backend",
    
    # Distributed and Parallel Components
    "ps",
    "get_pp_group",
    "get_world_group", 
    "init_distributed_environment",
    "init_model_parallel_group",
    "get_tensor_model_parallel_group",
    "get_tensor_model_parallel_cpu_group",
    "tensor_model_parallel_all_gather",
    "pynccl_utils",
    "init_custom_ar",
    "set_custom_all_reduce",
    "init_custom_ar_old",
    "initialize_model_parallel",
    "get_tensor_model_parallel_group_old",
    
    # LoRA Components
    "LoRARequest",
    "LoRAMapping",
    "LRUCacheWorkerLoRAManager",
    
    # Prompt Adapter Components
    "PromptAdapterRequest",
    "LRUCacheWorkerPromptAdapterManager",
    
    # Tokenizer Components
    "detokenize_incrementally",
    "get_cached_tokenizer",
    "AnyTokenizer",
    "TokenizerGroup",
    "BaseTokenizerGroup",
    "Detokenizer",
    "get_config",
    "get_hf_text_config",
    
    # Guided Decoding Components
    "GuidedDecodingRequest",
    "get_local_guided_decoding_logits_processor",
    "LLMGuidedOptions",
    
    # Utility Components
    "init_logger",
    "Counter",
    "LRUCache",
    "make_async",
    "is_hip",
    "get_cpu_memory",
    "get_nvcc_cuda_version",
    "in_wsl",
    "str_to_int_tuple",
    "CudaMemoryProfiler",
    "DeviceMemoryProfiler",
    "is_pin_memory_available",
    "init_cached_hf_modules",
    "FlexibleArgumentParser",
    "deprecate_kwargs",
    "weak_bind",
    "print_warning_once",
    "supports_dynamo",
    "UsageContext",
    "is_usage_stats_enabled",
    "usage_message",
    "VLLM_VERSION",
    "SpanAttributes",
    "SpanKind",
    "extract_trace_context",
    "init_tracer",
    "envs",
    
    # Compilation Components
    "CompilationLevel",
    
    # Multimodal Components
    "MULTIMODAL_REGISTRY",
    "MultiModalRegistry",
    
    # Plugin Components
    "get_torch_compile_backend",
    
    # Custom Extended Components
    "SingleStepOutputProcessor",
    "create_output_by_sequence_group",
]
