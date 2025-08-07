#!/usr/bin/env python3
"""
Setup script for Origin vLLM.
This script helps configure the environment for using Origin vLLM with inheritance.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def setup_environment():
    """Setup the environment for Origin vLLM."""
    print("Setting up Origin vLLM environment...")
    
    # Get the base directory
    base_dir = Path(__file__).parent.parent
    vllm_base_dir = base_dir / "vllm-0.5.4"
    origin_vllm_dir = base_dir / "origin_vllm"
    
    if not vllm_base_dir.exists():
        print(f"Error: Base vLLM directory not found at {vllm_base_dir}")
        print("Please ensure vLLM base installation is available.")
        return False
    
    if not origin_vllm_dir.exists():
        print(f"Error: Origin vLLM directory not found at {origin_vllm_dir}")
        return False
    
    print(f"Base vLLM directory: {vllm_base_dir}")
    print(f"Origin vLLM directory: {origin_vllm_dir}")
    
    # Add to Python path
    python_path = os.environ.get('PYTHONPATH', '')
    vllm_path_str = str(vllm_base_dir)
    origin_path_str = str(base_dir)
    
    if vllm_path_str not in python_path:
        if python_path:
            new_python_path = f"{vllm_path_str}:{python_path}"
        else:
            new_python_path = vllm_path_str
        os.environ['PYTHONPATH'] = new_python_path
        print(f"Added to PYTHONPATH: {vllm_path_str}")
    
    if origin_path_str not in python_path:
        current_python_path = os.environ.get('PYTHONPATH', '')
        if current_python_path:
            new_python_path = f"{origin_path_str}:{current_python_path}"
        else:
            new_python_path = origin_path_str
        os.environ['PYTHONPATH'] = new_python_path
        print(f"Added to PYTHONPATH: {origin_path_str}")
    
    print("Environment setup complete!")
    return True


def test_imports():
    """Test that the imports work correctly."""
    print("Testing imports...")
    
    try:
        # Test base vLLM imports
        sys.path.insert(0, str(Path(__file__).parent.parent / "vllm-0.5.4"))
        import vllm
        print("✓ Base vLLM import successful")
        
        # Test origin vLLM imports
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import hidden_vllm
        print("✓ Origin vLLM import successful")
        
        print("All imports successful!")
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def create_example_script():
    """Create an example script showing how to use Origin vLLM."""
    example_content = '''#!/usr/bin/env python3
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
    from origin_vllm import LLM, SamplingParams
    
    print("Origin vLLM imported successfully!")
    print(f"Version: {origin_vllm.__version__}")
    print(f"Commit: {origin_vllm.__commit__}")
    
    # Example usage (commented out to avoid requiring actual model)
    # model = LLM(model="facebook/opt-125m")
    # sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    # outputs = model.generate(["Hello, my name is"], sampling_params)
    # for output in outputs:
    #     print(output.outputs[0].text)
    
except ImportError as e:
    print(f"Import failed: {e}")
    print("Please run setup.py first to configure the environment.")
'''
    
    example_path = Path(__file__).parent.parent / "example_origin_vllm.py"
    with open(example_path, 'w') as f:
        f.write(example_content)
    
    print(f"Created example script: {example_path}")


def main():
    """Main setup function."""
    print("Origin vLLM Setup")
    print("=" * 50)
    
    if not setup_environment():
        print("Setup failed!")
        return 1
    
    if not test_imports():
        print("Import test failed!")
        return 1
    
    create_example_script()
    
    print("=" * 50)
    print("Setup completed successfully!")
    print("You can now use Origin vLLM with inheritance-based extensions.")
    print("Run example_origin_vllm.py to see usage examples.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
