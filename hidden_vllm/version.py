import warnings

try:
    # Try to import the base vllm commit id, fallback to our own if needed
    try:
        import vllm.commit_id
        __commit__ = vllm.commit_id.__commit__
    except ImportError:
        import hidden_vllm.commit_id
        __commit__ = hidden_vllm.commit_id.__commit__
except Exception as e:
    warnings.warn(f"Failed to read commit hash:\n{e}",
                  RuntimeWarning,
                  stacklevel=2)
    __commit__ = "COMMIT_HASH_PLACEHOLDER"

__version__ = "0.5.4+origin"
