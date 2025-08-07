from typing import Dict, Type

from our_vllm.model_executor.layers.quantization.aqlm import AQLMConfig
from our_vllm.model_executor.layers.quantization.awq import AWQConfig
from our_vllm.model_executor.layers.quantization.awq_marlin import AWQMarlinConfig
from our_vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from our_vllm.model_executor.layers.quantization.bitsandbytes import (
    BitsAndBytesConfig)
from our_vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsConfig)
from our_vllm.model_executor.layers.quantization.deepspeedfp import (
    DeepSpeedFPConfig)
from our_vllm.model_executor.layers.quantization.fbgemm_fp8 import FBGEMMFp8Config
from our_vllm.model_executor.layers.quantization.fp8 import Fp8Config
from our_vllm.model_executor.layers.quantization.gptq import GPTQConfig
from our_vllm.model_executor.layers.quantization.gptq_marlin import (
    GPTQMarlinConfig)
from our_vllm.model_executor.layers.quantization.gptq_marlin_24 import (
    GPTQMarlin24Config)
from our_vllm.model_executor.layers.quantization.marlin import MarlinConfig
from our_vllm.model_executor.layers.quantization.qqq import QQQConfig
from our_vllm.model_executor.layers.quantization.squeezellm import SqueezeLLMConfig

QUANTIZATION_METHODS: Dict[str, Type[QuantizationConfig]] = {
    "aqlm": AQLMConfig,
    "awq": AWQConfig,
    "deepspeedfp": DeepSpeedFPConfig,
    "fp8": Fp8Config,
    "fbgemm_fp8": FBGEMMFp8Config,
    # The order of gptq methods is important for config.py iteration over
    # override_quantization_method(..)
    "marlin": MarlinConfig,
    "gptq_marlin_24": GPTQMarlin24Config,
    "gptq_marlin": GPTQMarlinConfig,
    "awq_marlin": AWQMarlinConfig,
    "gptq": GPTQConfig,
    "squeezellm": SqueezeLLMConfig,
    "compressed-tensors": CompressedTensorsConfig,
    "bitsandbytes": BitsAndBytesConfig,
    "qqq": QQQConfig,
}


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")
    return QUANTIZATION_METHODS[quantization]


__all__ = [
    "QuantizationConfig",
    "get_quantization_config",
    "QUANTIZATION_METHODS",
]
