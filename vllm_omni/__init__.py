"""
vLLM-Omni: Multi-modality models inference and serving with
non-autoregressive structures.

This package extends vLLM beyond traditional text-based, autoregressive
generation to support multi-modality models with non-autoregressive
structures and non-textual outputs.

Architecture:
- 🟡 Modified: vLLM components modified for multimodal support
- 🔴 Added: New components for multimodal and non-autoregressive
  processing
"""

from __future__ import annotations

import os
from contextlib import contextmanager

# We import version early, because it will warn if vLLM / vLLM Omni
# are not using the same major + minor version (if vLLM is installed).
# We should do this before applying patch, because vLLM imports might
# throw in patch if the versions differ.
from .version import __version__, __version_tuple__  # isort:skip # noqa: F401


@contextmanager
def _temporary_env(key: str, value: str):
    old = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


try:
    # vLLM may import modules that call `torch.compile` at import time on some
    # versions/configurations. That can fail in environments where Inductor
    # backends are not fully usable (e.g. `duplicate template name`).
    #
    # We only disable Dynamo during *import* to keep `import vllm_omni` safe,
    # without permanently turning off compilation for the running process.
    if os.environ.get("TORCHDYNAMO_DISABLE") is None:
        with _temporary_env("TORCHDYNAMO_DISABLE", "1"):
            from . import patch  # noqa: F401
    else:
        from . import patch  # noqa: F401
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    if exc.name != "vllm":
        raise
    # Allow importing vllm_omni without vllm (e.g., documentation builds)
    patch = None  # type: ignore

# Register custom configs (AutoConfig, AutoTokenizer) as early as possible.
from vllm_omni.transformers_utils import configs as _configs  # noqa: F401, E402
from vllm_omni.transformers_utils import parsers as _parsers  # noqa: F401, E402


def __getattr__(name: str):
    # Lazy import for AsyncOmni and Omni to avoid pulling in heavy
    # dependencies (vllm model_loader → fused_moe → pynvml) at package
    # import time.  This prevents crashes in lightweight subprocesses
    # (e.g. model-architecture inspection) that lack a CUDA context.
    # See: https://github.com/vllm-project/vllm-omni/issues/1793
    if name == "AsyncOmni":
        if os.environ.get("TORCHDYNAMO_DISABLE") is None:
            with _temporary_env("TORCHDYNAMO_DISABLE", "1"):
                from .entrypoints.async_omni import AsyncOmni
        else:
            from .entrypoints.async_omni import AsyncOmni
        return AsyncOmni
    if name == "Omni":
        if os.environ.get("TORCHDYNAMO_DISABLE") is None:
            with _temporary_env("TORCHDYNAMO_DISABLE", "1"):
                from .entrypoints.omni import Omni
        else:
            from .entrypoints.omni import Omni
        return Omni
    if name == "OmniModelConfig":
        if os.environ.get("TORCHDYNAMO_DISABLE") is None:
            with _temporary_env("TORCHDYNAMO_DISABLE", "1"):
                from .config import OmniModelConfig
        else:
            from .config import OmniModelConfig
        return OmniModelConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "__version_tuple__",
    # Main components
    "Omni",
    "AsyncOmni",
    # Configuration
    "OmniModelConfig",
    # All other components are available through their respective modules
    # processors.*, schedulers.*, executors.*, etc.
]
