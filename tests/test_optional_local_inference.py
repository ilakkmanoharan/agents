"""Test that livekit-local-inference is optional and fails gracefully when missing."""

import sys
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


def test_import_livekit_agents_without_local_inference():
    """Test that livekit.agents can be imported when livekit-local-inference is unavailable."""
    # Save the original module
    original_module = sys.modules.get("livekit.local_inference")

    try:
        # Block livekit.local_inference
        sys.modules["livekit.local_inference"] = None  # type: ignore

        # Force reimport of the modules that use local_inference
        for module in [
            "livekit.agents.inference.vad",
            "livekit.agents.inference.eot.transports",
            "livekit.agents.inference.eot",
            "livekit.agents.inference",
        ]:
            if module in sys.modules:
                del sys.modules[module]

        # This should succeed even without livekit.local_inference
        import livekit.agents
        from livekit.agents import llm, stt, tts, utils

        # Verify key modules are available
        assert livekit.agents is not None
        assert llm is not None
        assert stt is not None
        assert tts is not None
        assert utils is not None

    finally:
        # Restore the original module
        if original_module is None and "livekit.local_inference" in sys.modules:
            del sys.modules["livekit.local_inference"]
        elif original_module is not None:
            sys.modules["livekit.local_inference"] = original_module


def test_vad_fails_gracefully_without_local_inference():
    """Test that VAD raises clear ImportError when livekit-local-inference is missing."""
    with patch.dict(sys.modules, {"livekit.local_inference": None}):
        # Clear cached imports
        from livekit.agents.inference import vad as vad_module

        vad_module._cached_native_vad_cls = None
        vad_module._VAD_WINDOW_SAMPLES = None

        # Test that the lazy import helper raises the expected error
        with pytest.raises(ImportError) as exc_info:
            vad_module._ensure_local_inference()

        assert "livekit-local-inference" in str(exc_info.value).lower()
        assert "pip install livekit-agents[local-inference]" in str(exc_info.value)


def test_local_eot_fails_gracefully_without_local_inference():
    """Test that local EOT raises clear ImportError when livekit-local-inference is missing."""
    with patch.dict(sys.modules, {"livekit.local_inference": None}):
        # Clear cached imports
        from livekit.agents.inference.eot import transports

        transports._cached_native_eot_cls = None

        # Test that the lazy import helper raises the expected error
        with pytest.raises(ImportError) as exc_info:
            transports._ensure_local_inference_eot()

        assert "livekit-local-inference" in str(exc_info.value).lower()
        assert "pip install livekit-agents[local-inference]" in str(exc_info.value)


def test_preload_skips_gracefully_without_local_inference():
    """Test that preload gracefully skips when livekit-local-inference is unavailable."""
    with patch.dict(sys.modules, {"livekit.local_inference": None}):
        # Force reimport of preload module
        if "livekit.agents.ipc._preload" in sys.modules:
            del sys.modules["livekit.agents.ipc._preload"]

        # This should not raise, just skip the local inference preload
        from livekit.agents.ipc import _preload

        # The module should import successfully
        assert _preload is not None


def test_vad_works_with_local_inference():
    """Test that VAD works normally when livekit-local-inference is available."""
    try:
        from livekit.agents.inference import vad as vad_module

        # This should succeed when the package is available
        NativeVAD, VAD_WINDOW_SAMPLES = vad_module._ensure_local_inference()

        assert NativeVAD is not None
        assert VAD_WINDOW_SAMPLES is not None
        assert VAD_WINDOW_SAMPLES > 0
    except ImportError:
        pytest.skip("livekit-local-inference not installed")


def test_local_eot_works_with_local_inference():
    """Test that local EOT works normally when livekit-local-inference is available."""
    try:
        from livekit.agents.inference.eot import transports

        # This should succeed when the package is available
        EOT = transports._ensure_local_inference_eot()

        assert EOT is not None
    except ImportError:
        pytest.skip("livekit-local-inference not installed")
