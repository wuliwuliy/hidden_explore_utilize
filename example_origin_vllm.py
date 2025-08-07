#!/usr/bin/env python3
"""
Example script showing how to use Origin vLLM.
This demonstrates the inheritance-based approach.
"""

import sys
import os
from pathlib import Path

# Add paths for imports
base_dir = Path(__file__).parent
vllm_path = base_dir / "vllm-0.5.4"
sys.path.insert(0, str(vllm_path))
sys.path.insert(0, str(base_dir))

try:
    # Import Origin vLLM (which inherits from base vLLM)
    import hidden_vllm
    from hidden_vllm import LLM
    
    print("Origin vLLM imported successfully!")
    print(f"Version: {hidden_vllm.__version__}")
    print(f"Commit: {hidden_vllm.__commit__}")
    
    # Example usage (commented out to avoid requiring actual model)
    # from origin_vllm import SamplingParams
    # model = LLM(model="facebook/opt-125m")
    # sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    # outputs = model.generate(["Hello, my name is"], sampling_params)
    # for output in outputs:
    #     print(output.outputs[0].text)
    
    print("Origin vLLM is ready to use!")
    
except ImportError as e:
    print(f"Import failed: {e}")
    print("Please run setup.py first to configure the environment.")
    print("Or make sure vLLM base installation is properly configured.")
