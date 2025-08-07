"""Origin vLLM: Extended vLLM implementation using inheritance

This module provides extended functionality by inheriting from the base vLLM implementation.
It maintains compatibility with the original vLLM API while adding custom features.
"""

import sys
import os
import warnings

# Add the vllm-0.5.4 directory to sys.path for importing
vllm_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'vllm-0.5.4')
if vllm_path not in sys.path:
    sys.path.insert(0, vllm_path)

try:
    # Import base vLLM classes
    from vllm.engine.arg_utils import AsyncEngineArgs as BaseAsyncEngineArgs
    from vllm.engine.arg_utils import EngineArgs as BaseEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine as BaseAsyncLLMEngine
    from vllm.engine.llm_engine import LLMEngine as BaseLLMEngine
    from vllm.entrypoints.llm import LLM as BaseLLM
    from vllm.executor.ray_utils import initialize_ray_cluster
    from vllm.inputs import PromptInputs, TextPrompt, TokensPrompt
    from vllm.model_executor.models import ModelRegistry as BaseModelRegistry
    from vllm.outputs import (CompletionOutput, EmbeddingOutput,
                              EmbeddingRequestOutput, RequestOutput)
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams

    from vllm.config import (CacheConfig, DeviceConfig, ModelConfig, ParallelConfig, SchedulerConfig, LoRAConfig)
    from vllm.logger import init_logger
    from vllm.transformers_utils.config import get_config
    from vllm.utils import get_cpu_memory, is_hip, get_nvcc_cuda_version

    from vllm.lora.request import LoRARequest
    from vllm.config import (CacheConfig, DeviceConfig, ModelConfig, ParallelConfig, SchedulerConfig, LoRAConfig)
    from vllm.core.scheduler import Scheduler, SchedulerOutputs
    from vllm.logger import init_logger
    from vllm.outputs import RequestOutput
    from vllm.sampling_params import SamplingParams
    from vllm.sequence import (SamplerOutput, Sequence, SequenceGroup, SequenceGroupMetadata, SequenceGroupOutput,
                            SequenceOutput, SequenceStatus)
    from vllm.transformers_utils.tokenizer import detokenize_incrementally
    from vllm.engine.metrics import StatLogger, Stats
    from vllm.utils import Counter

    from vllm.lora.request import LoRARequest
    from vllm.outputs import RequestOutput
    from vllm.sampling_params import SamplingParams
    from vllm.utils import Counter

    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.weight_utils import (get_quant_config, initialize_dummy_weights)
    from vllm.config import DeviceConfig, LoRAConfig
    from vllm.model_executor.sampling_metadata import SamplingMetadata, SamplingTensors
    from vllm.sequence import SamplerOutput
    from vllm.model_executor.layers.sampler import Sampler
    from vllm.model_executor.layers.sampler import _prune_hidden_states, _apply_logits_processors, _apply_penalties, _apply_top_k_top_p, _apply_min_p, _apply_penalties, _sample, _get_logprobs, _build_sampler_output
    from vllm.model_executor.layers.linear import *
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
    from vllm.model_executor.layers.activation import ScaledActivation
    from vllm.model_executor.layers.sampler import Sampler

    from vllm.config import (DeviceConfig, ModelConfig, LoRAConfig, ParallelConfig, SchedulerConfig)
    from vllm.logger import init_logger
    from vllm.model_executor import InputMetadata, SamplingMetadata
    from vllm.sampling_params import SamplingParams, SamplingType
    from vllm.sequence import SamplerOutput, SequenceData, SequenceGroupMetadata
    from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
    from vllm.lora.layers import LoRAMapping
    from vllm.lora.request import LoRARequest
    from vllm.utils import in_wsl
    from vllm.worker.model_runner import ModelRunner, CUDAGraphRunner, _async_h2d
    
    import vllm.model_executor.parallel_utils.parallel_state as ps
    
    from vllm.lora.request import LoRARequest
    from vllm.utils import make_async, LRUCache
    from vllm.transformers_utils.tokenizers import *

    from vllm.config import (CacheConfig, DeviceConfig, ModelConfig, ParallelConfig, SchedulerConfig, LoRAConfig)
    from vllm.model_executor import InputMetadata, set_random_seed
    from vllm.model_executor.parallel_utils.parallel_state import (initialize_model_parallel)
    from vllm.sampling_params import SamplingParams, SamplingType
    from vllm.sequence import SamplerOutput, SequenceData, SequenceGroupMetadata
    from vllm.worker.cache_engine import CacheEngine
    from vllm.model_executor.parallel_utils.custom_all_reduce import init_custom_ar
    from vllm.model_executor.parallel_utils.parallel_state import get_tensor_model_parallel_group
    from vllm.lora.request import LoRARequest

    from vllm.config import (CacheConfig, DecodingConfig, DeviceConfig, EngineConfig, LoRAConfig, ParallelConfig,
                            SchedulerConfig, SpeculativeConfig, TokenizerPoolConfig, VisionLanguageConfig)
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
    from vllm.utils import str_to_int_tuple
    
    from vllm.logger import init_logger
    from vllm.model_executor.layers.quantization import get_quantization_config
    from vllm.transformers_utils.config import get_hf_text_config
    from vllm.utils import is_hip
    # Add for verl
    from vllm.config import ModelConfig, _get_and_verify_dtype, _get_and_verify_max_len

    from vllm.model_executor.layers.linear import *
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    from vllm.model_executor.model_loader.utils import set_default_torch_dtype
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm.model_executor.models.gemma import GemmaForCausalLM

    import vllm
    from vllm.config import (CacheConfig, DecodingConfig, DeviceConfig, LoRAConfig, ParallelConfig, SchedulerConfig,
                            SpeculativeConfig, VisionLanguageConfig)
    from vllm.core.scheduler import Scheduler
    from vllm.engine.output_processor.interfaces import (SequenceGroupOutputProcessor)
    from vllm.engine.output_processor.stop_checker import StopChecker
    from vllm.executor.executor_base import ExecutorBase
    from vllm.logger import init_logger
    from vllm.transformers_utils.detokenizer import Detokenizer
    from vllm.engine.metrics import StatLogger
    from vllm.usage.usage_lib import (UsageContext, is_usage_stats_enabled, usage_message)
    from vllm.utils import Counter
    from vllm.engine.llm_engine import _load_generation_config_dict
    from vllm.engine.llm_engine import LLMEngine
    from vllm.model_executor.model_loader import (get_architecture_class_name)

    from vllm.lora.request import LoRARequest
    from vllm.outputs import RequestOutput
    from vllm.sampling_params import SamplingParams
    from vllm.sequence import MultiModalData
    from vllm.usage.usage_lib import UsageContext
    from vllm.utils import Counter

    from vllm.model_executor.layers.linear import *
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
    from vllm.model_executor.layers.activation import ScaledActivation
    from vllm.model_executor.models import ModelRegistry

    from vllm.config import (DeviceConfig, LoRAConfig, ParallelConfig, SchedulerConfig, VisionLanguageConfig)
    from vllm.model_executor.model_loader import BaseModelLoader
    from vllm.model_executor.model_loader.loader import _initialize_model
    from vllm.model_executor.model_loader.utils import set_default_torch_dtype
    from vllm.distributed.communication_op import tensor_model_parallel_all_gather
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    from vllm.attention import (AttentionMetadata, get_attn_backend)
    from vllm.config import (DeviceConfig, LoRAConfig, ParallelConfig, SchedulerConfig, VisionLanguageConfig)
    from vllm.logger import init_logger
    from vllm.lora.layers import LoRAMapping
    from vllm.lora.request import LoRARequest
    from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
    from vllm.model_executor import SamplingMetadata
    from vllm.sequence import (MultiModalData, SamplerOutput, SequenceData, SequenceGroupMetadata)
    from vllm.utils import (CudaMemoryProfiler, is_hip, is_pin_memory_available)
    from vllm.worker.model_runner import ModelRunner, CUDAGraphRunner

    import vllm.distributed.parallel_state as ps
    import vllm.envs as envs
    from vllm.logger import init_logger

    import vllm.envs as envs
    from vllm.executor.executor_base import ExecutorBase, ExecutorAsyncBase
    from vllm.logger import init_logger
    from vllm.lora.request import LoRARequest
    from vllm.sequence import SamplerOutput, ExecuteModelRequest
    from vllm.config import (CacheConfig, DeviceConfig, LoRAConfig, ParallelConfig, SchedulerConfig, SpeculativeConfig,
                            VisionLanguageConfig)
    
    from vllm.lora.request import LoRARequest
    from vllm.utils import make_async, LRUCache
    from vllm.transformers_utils.tokenizers import *

    from vllm.config import (CacheConfig, DeviceConfig, LoRAConfig, ParallelConfig, SchedulerConfig, VisionLanguageConfig)
    from vllm.model_executor import set_random_seed
    from vllm.sequence import SamplerOutput, ExecuteModelRequest
    from vllm.worker.cache_engine import CacheEngine
    from vllm.distributed.device_communicators import pynccl_utils
    from vllm.distributed.device_communicators.custom_all_reduce import (init_custom_ar)
    # TODO(sgm): check why vllm has similar file in vllm.model_executor.parallel_utils.parallel_state
    from vllm.distributed import get_tensor_model_parallel_cpu_group, init_distributed_environment, get_tensor_model_parallel_group
    from vllm.worker.worker import Worker, _check_if_gpu_supports_dtype

    from vllm.config import (CacheConfig, DecodingConfig, DeviceConfig, EngineConfig, LoRAConfig, MultiModalConfig,
                            ObservabilityConfig, ParallelConfig, PromptAdapterConfig, SchedulerConfig, SpeculativeConfig,
                            TokenizerPoolConfig)
    from vllm.executor.executor_base import ExecutorBase
    from vllm.logger import init_logger
    from vllm.utils import FlexibleArgumentParser
    from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS
    from vllm.utils import str_to_int_tuple
    from vllm.transformers_utils.tokenizer_group.base_tokenizer_group import (BaseTokenizerGroup)

    from vllm.logger import init_logger
    from vllm.model_executor.layers.quantization import get_quantization_config
    from vllm.transformers_utils.config import get_hf_text_config
    from vllm.utils import is_hip, print_warning_once
    # Add for verl
    from vllm.config import ModelConfig, _get_and_verify_dtype, _get_and_verify_max_len, get_served_model_name

    from vllm.model_executor.layers.linear import *
    from vllm.model_executor.models import ModelRegistry
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm.model_executor.models.utils import is_pp_missing_parameter
    from vllm.model_executor.layers.fused_moe import FusedMoE

    from vllm.model_executor.model_loader.utils import set_default_torch_dtype
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader  

    import vllm.envs as envs
    from vllm.config import (CacheConfig, DecodingConfig, DeviceConfig, EngineConfig, LoRAConfig, MultiModalConfig,
                            ObservabilityConfig, ParallelConfig, PromptAdapterConfig, SchedulerConfig, SpeculativeConfig)
    from vllm.core.scheduler import Scheduler
    from vllm.engine.output_processor.interfaces import (SequenceGroupOutputProcessor)
    from vllm.engine.output_processor.stop_checker import StopChecker
    from vllm.executor.executor_base import ExecutorBase
    from vllm.inputs import INPUT_REGISTRY, LLMInputs, PromptInputs
    from vllm.logger import init_logger
    from vllm.transformers_utils.detokenizer import Detokenizer
    from vllm.engine.metrics import (LoggingStatLogger, PrometheusStatLogger, StatLoggerBase, Stats)
    from vllm.tracing import (SpanAttributes, SpanKind, extract_trace_context, init_tracer)
    from vllm.usage.usage_lib import (UsageContext, is_usage_stats_enabled, usage_message)
    from vllm.utils import Counter
    from vllm.engine.llm_engine import _load_generation_config_dict
    from vllm.engine.llm_engine import LLMEngine
    from vllm.version import __version__ as VLLM_VERSION
    from vllm.model_executor.model_loader import (get_architecture_class_name)

    from vllm import LLM
    from vllm.inputs import (PromptInputs, TextPrompt, TokensPrompt, parse_and_batch_prompt)
    from vllm.logger import init_logger
    from vllm.lora.request import LoRARequest
    from vllm.model_executor.guided_decoding import (GuidedDecodingRequest, get_local_guided_decoding_logits_processor)
    from vllm.model_executor.guided_decoding.guided_fields import LLMGuidedOptions
    from vllm.outputs import EmbeddingRequestOutput, RequestOutput
    from vllm.pooling_params import PoolingParams
    from vllm.prompt_adapter.request import PromptAdapterRequest
    from vllm.sampling_params import SamplingParams
    from vllm.transformers_utils.tokenizer import get_cached_tokenizer
    from vllm.usage.usage_lib import UsageContext
    from vllm.utils import Counter, deprecate_kwargs

    from vllm.model_executor.layers.linear import *
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
    from vllm.model_executor.layers.activation import ScaledActivation
    from vllm.model_executor.models import ModelRegistry

    from vllm.config import (CacheConfig, DeviceConfig, LoadConfig, LoRAConfig, ModelConfig, MultiModalConfig,
                            ParallelConfig, SchedulerConfig)
    from vllm.model_executor.model_loader import BaseModelLoader
    from vllm.model_executor.model_loader.loader import _initialize_model
    from vllm.model_executor.model_loader.utils import set_default_torch_dtype
    from vllm.distributed.communication_op import tensor_model_parallel_all_gather
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    import vllm.envs as envs
    from vllm.attention import (AttentionMetadata, get_attn_backend)
    from vllm.config import (CacheConfig, DeviceConfig, LoRAConfig, MultiModalConfig, ParallelConfig, PromptAdapterConfig,
                            SchedulerConfig)
    from vllm.logger import init_logger
    from vllm.lora.layers import LoRAMapping
    from vllm.lora.request import LoRARequest
    from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
    from vllm.model_executor import SamplingMetadata
    from vllm.model_executor.models.interfaces import (supports_lora, supports_vision)
    from vllm.utils import (CudaMemoryProfiler, is_hip, is_pin_memory_available)
    from vllm.worker.model_runner import ModelRunner, CUDAGraphRunner
    from vllm.prompt_adapter.worker_manager import (LRUCacheWorkerPromptAdapterManager)

    import vllm.distributed.parallel_state as ps
    from vllm.distributed.parallel_state import get_pp_group, get_world_group, init_distributed_environment, init_model_parallel_group
    import vllm.envs as envs
    from vllm.logger import init_logger

    import vllm.envs as envs
    from vllm.executor.executor_base import ExecutorBase, ExecutorAsyncBase
    from vllm.logger import init_logger
    from vllm.lora.request import LoRARequest
    from vllm.sequence import SamplerOutput, ExecuteModelRequest

    from vllm.config import (CacheConfig, DeviceConfig, LoRAConfig, MultiModalConfig, ParallelConfig, PromptAdapterConfig,
                            SchedulerConfig, SpeculativeConfig)
    from vllm.prompt_adapter.request import PromptAdapterRequest

    from vllm.lora.request import LoRARequest
    from vllm.utils import make_async, LRUCache
    from vllm.transformers_utils.tokenizers import *

    from vllm.config import (CacheConfig, DeviceConfig, LoRAConfig, MultiModalConfig, ParallelConfig, PromptAdapterConfig,
                            SchedulerConfig, SpeculativeConfig)
    from vllm.model_executor import set_random_seed
    from vllm.sequence import (ExecuteModelRequest, IntermediateTensors, SamplerOutput)
    from vllm.worker.cache_engine import CacheEngine
    # TODO(sgm): check why vllm has similar file in vllm.model_executor.parallel_utils.parallel_state
    from vllm.distributed import (init_distributed_environment, set_custom_all_reduce, get_tensor_model_parallel_group)
    from vllm.worker.worker_base import WorkerInput
    from vllm.worker.worker import Worker, _check_if_gpu_supports_dtype
    from vllm.worker.model_runner_base import ModelRunnerBase, ModelRunnerInputBase
    from vllm.worker.embedding_model_runner import EmbeddingModelRunner
    from vllm.worker.model_runner import GPUModelRunnerBase
    from vllm.utils import init_cached_hf_modules

    from vllm.config import EngineConfig
    from vllm.engine.arg_utils import EngineArgs

    from vllm.config import ModelConfig
    from vllm.logger import init_logger
    from vllm.utils import is_hip
    from vllm.model_executor.model_loader.loader import BaseModelLoader

    from vllm.model_executor.model_loader.weight_utils import default_weight_loader
    from vllm.model_executor.models.utils import is_pp_missing_parameter
    from vllm.model_executor.layers.fused_moe import FusedMoE

    from vllm.model_executor.model_loader.utils import set_default_torch_dtype
    
    from vllm.config import (
        CacheConfig,
        DecodingConfig,
        DeviceConfig,
        EngineConfig,
        LoadConfig,
        LoRAConfig,
        ModelConfig,
        ObservabilityConfig,
        ParallelConfig,
        PromptAdapterConfig,
        SchedulerConfig,
        SpeculativeConfig,
    )
    from vllm.core.scheduler import Scheduler
    from vllm.engine.arg_utils import EngineArgs
    from vllm.engine.llm_engine import LLMEngine, SchedulerContext, SchedulerOutputState, _load_generation_config_dict
    from vllm.engine.metrics_types import StatLoggerBase
    from vllm.engine.output_processor.interfaces import SequenceGroupOutputProcessor
    from vllm.engine.output_processor.stop_checker import StopChecker
    from vllm.executor.executor_base import ExecutorBase
    from vllm.inputs import INPUT_REGISTRY, InputRegistry
    from vllm.inputs.preprocess import InputPreprocessor
    from vllm.logger import init_logger
    from vllm.sequence import Sequence
    from vllm.tracing import init_tracer
    from vllm.transformers_utils.detokenizer import Detokenizer
    from vllm.transformers_utils.tokenizer import AnyTokenizer
    from vllm.usage.usage_lib import UsageContext, is_usage_stats_enabled, usage_message
    from vllm.utils import Counter, weak_bind
    from vllm.version import __version__ as VLLM_VERSION
    from vllm.model_executor.model_loader import get_architecture_class_name
    from vllm.engine.metrics import LoggingStatLogger, PrometheusStatLogger

    from vllm import LLM
    from vllm.outputs import EmbeddingRequestOutput, RequestOutput
    from vllm.utils import Counter

    from vllm.model_executor.layers.linear import *
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
    from vllm.model_executor.models import ModelRegistry

    from vllm.config import CacheConfig, DeviceConfig, LoadConfig, LoRAConfig, ModelConfig, ParallelConfig, SchedulerConfig
    from vllm.distributed.communication_op import tensor_model_parallel_all_gather
    from vllm.model_executor.model_loader import BaseModelLoader
    from vllm.model_executor.model_loader.loader import _initialize_model
    from vllm.model_executor.model_loader.utils import set_default_torch_dtype
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    import vllm.envs as envs
    from vllm.compilation.levels import CompilationLevel
    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        LoadConfig,
        LoRAConfig,
        ModelConfig,
        ObservabilityConfig,
        ParallelConfig,
        PromptAdapterConfig,
        SchedulerConfig,
    )
    from vllm.inputs import INPUT_REGISTRY, InputRegistry
    from vllm.logger import init_logger
    from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
    from vllm.model_executor.models.interfaces import supports_lora
    from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
    from vllm.prompt_adapter.worker_manager import LRUCacheWorkerPromptAdapterManager
    from vllm.utils import DeviceMemoryProfiler, is_hip, supports_dynamo
    from vllm.worker.model_runner import ModelRunner
    from vllm.plugins import get_torch_compile_backend

    import vllm.distributed.parallel_state as ps
    from vllm.distributed.parallel_state import (
        get_pp_group,
        get_world_group,
        init_distributed_environment,
        init_model_parallel_group,
    )
    from vllm.logger import init_logger

    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        LoRAConfig,
        ObservabilityConfig,
        ParallelConfig,
        PromptAdapterConfig,
        SchedulerConfig,
        SpeculativeConfig,
    )
    from vllm.executor.executor_base import ExecutorAsyncBase, ExecutorBase
    from vllm.logger import init_logger
    from vllm.lora.request import LoRARequest
    from vllm.model_executor.layers.sampler import SamplerOutput
    from vllm.sequence import ExecuteModelRequest
    from vllm.prompt_adapter.request import PromptAdapterRequest

    from vllm.transformers_utils.tokenizer_group import TokenizerGroup
    from vllm.utils import LRUCache

    from vllm.config import (
        CacheConfig,
        DeviceConfig,
        LoRAConfig,
        ParallelConfig,
        PromptAdapterConfig,
        SchedulerConfig,
        SpeculativeConfig,
    )

    # TODO(sgm): check why vllm has similar file in vllm.model_executor.parallel_utils.parallel_state
    from vllm.distributed import get_tensor_model_parallel_group, init_distributed_environment, set_custom_all_reduce
    from vllm.model_executor import set_random_seed
    from vllm.model_executor.layers.sampler import SamplerOutput
    from vllm.sequence import ExecuteModelRequest, IntermediateTensors
    from vllm.worker.cache_engine import CacheEngine
    from vllm.worker.embedding_model_runner import EmbeddingModelRunner
    from vllm.worker.model_runner import GPUModelRunnerBase
    from vllm.worker.model_runner_base import ModelRunnerInputBase
    from vllm.worker.worker import Worker, _check_if_gpu_supports_dtype
    from vllm.worker.worker_base import WorkerInput
    from vllm.utils import init_cached_hf_modules

    from vllm import SamplingParams

    from vllm import LLM, SamplingParams
    from vllm import SamplingParams


    # Import our extended classes
    from .engine.arg_utils import AsyncEngineArgs, EngineArgs
    from .engine.async_llm_engine import AsyncLLMEngine
    from .engine.llm_engine import LLMEngine
    from .entrypoints.llm import LLM
    from .model_executor.models import ModelRegistry
    from .version import __commit__, __version__
    
except ImportError as e:
    warnings.warn(f"Failed to import some vLLM components: {e}. "
                  "Make sure vLLM is properly installed and accessible.",
                  RuntimeWarning, stacklevel=2)
    
    # Fallback imports - just import version info
    from .version import __commit__, __version__
    
    # Create placeholder classes if base imports fail
    class LLM:
        """Placeholder LLM class when base vLLM is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("Base vLLM library is not available. Please install vLLM first.")
    
    class LLMEngine:
        """Placeholder LLMEngine class when base vLLM is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("Base vLLM library is not available. Please install vLLM first.")
    
    class AsyncLLMEngine:
        """Placeholder AsyncLLMEngine class when base vLLM is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("Base vLLM library is not available. Please install vLLM first.")
    
    class EngineArgs:
        """Placeholder EngineArgs class when base vLLM is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("Base vLLM library is not available. Please install vLLM first.")
    
    class AsyncEngineArgs:
        """Placeholder AsyncEngineArgs class when base vLLM is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("Base vLLM library is not available. Please install vLLM first.")
    
    class ModelRegistry:
        """Placeholder ModelRegistry class when base vLLM is not available."""
        def __init__(self, *args, **kwargs):
            raise ImportError("Base vLLM library is not available. Please install vLLM first.")

    # Set undefined variables as None for other imports
    PromptInputs = None
    TextPrompt = None
    TokensPrompt = None
    SamplingParams = None
    RequestOutput = None
    CompletionOutput = None
    EmbeddingOutput = None
    EmbeddingRequestOutput = None
    PoolingParams = None
    initialize_ray_cluster = None

__all__ = [
    "__commit__",
    "__version__",
    "LLM",
    "ModelRegistry",
    "PromptInputs",
    "TextPrompt",
    "TokensPrompt",
    "SamplingParams",
    "RequestOutput",
    "CompletionOutput",
    "EmbeddingOutput",
    "EmbeddingRequestOutput",
    "LLMEngine",
    "EngineArgs",
    "AsyncLLMEngine",
    "AsyncEngineArgs",
    "initialize_ray_cluster",
    "PoolingParams",

    "CacheConfig", 
    "DeviceConfig", 
    "ModelConfig", 
    "ParallelConfig", 
    "SchedulerConfig", 
    "LoRAConfig",


]
