"""Expose the configured receiver loader without editing its model package."""
import os

_receiver_models = os.environ.get("RECEIVER_MODELS_DIR")
if _receiver_models:
    import models

    if _receiver_models not in models.__path__:
        models.__path__.append(_receiver_models)
