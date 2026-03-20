"""
Model Parser
============
Inspects Keras (TensorFlow) or custom NumPy-based models and extracts
layer configurations and weights into a unified intermediate representation
(IR) that the optimizer pipeline can process.

Supported model types:
  1. Keras Sequential / Functional models (tf.keras or standalone keras)
  2. OpenCL optimizer's own OptimizedModel (pass-through)
  3. dict-based model spec ({"layers": [...]}) — for testing / custom models
"""

import numpy as np
from typing import List, Dict, Any, Optional
from optimizer.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Intermediate Representation
# ---------------------------------------------------------------------------

class LayerSpec:
    """
    Unified description of a single model layer extracted by the parser.

    Attributes:
        layer_type  : 'dense', 'conv2d', 'activation', 'pooling', 'flatten', 'unknown'
        name        : Layer name string
        weights     : List of numpy weight arrays (may be empty)
        config      : Dict of hyperparameters (units, filters, activation, etc.)
        original    : Reference to the original layer object (Keras etc.)
    """

    __slots__ = ("layer_type", "name", "weights", "config", "original")

    def __init__(
        self,
        layer_type: str,
        name: str,
        weights: List[np.ndarray],
        config: Dict[str, Any],
        original: Any = None,
    ):
        self.layer_type = layer_type
        self.name       = name
        self.weights    = weights
        self.config     = config
        self.original   = original

    def __repr__(self) -> str:
        return f"LayerSpec(type='{self.layer_type}', name='{self.name}', config={self.config})"


class ModelParser:
    """
    Parses a model object and returns a list of LayerSpec objects.

    Usage:
        parser = ModelParser()
        specs  = parser.parse(model)
    """

    def parse(self, model: Any) -> List[LayerSpec]:
        """
        Auto-detect model type and extract layer specifications.

        Args:
            model : A Keras model, dict spec, or OptimizedModel.

        Returns:
            List of LayerSpec objects describing each layer.
        """
        # Already optimized — return as-is marker
        if hasattr(model, "_is_opencl_optimized"):
            logger.info("Model is already optimized — skipping parse.")
            return []

        # Keras / TensorFlow model
        if self._is_keras_model(model):
            logger.info("Detected Keras model.")
            return self._parse_keras(model)

        # Dict-based spec (testing / custom usage)
        if isinstance(model, dict) and "layers" in model:
            logger.info("Detected dict-based model spec.")
            return self._parse_dict(model)

        logger.warning(
            f"Unknown model type: {type(model)}. Cannot extract layers automatically.\n"
            f"Consider passing a Keras model or a dict spec."
        )
        return []

    # ------------------------------------------------------------------
    # Keras parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _is_keras_model(obj: Any) -> bool:
        type_name = type(obj).__module__ + "." + type(obj).__qualname__
        return "keras" in type_name.lower() or hasattr(obj, "layers")

    def _parse_keras(self, model: Any) -> List[LayerSpec]:
        specs = []
        for layer in model.layers:
            spec = self._parse_keras_layer(layer)
            specs.append(spec)
        return specs

    def _parse_keras_layer(self, layer: Any) -> LayerSpec:
        try:
            cfg = layer.get_config()
        except Exception:
            cfg = {}

        type_name = type(layer).__name__.lower()
        raw_weights = [w.numpy() for w in layer.weights] if layer.weights else []

        # ---- Dense ----
        if "dense" in type_name:
            w, b = (raw_weights[0], raw_weights[1]) if len(raw_weights) >= 2 else (None, None)
            return LayerSpec(
                layer_type="dense",
                name=layer.name,
                weights=[w, b] if w is not None else [],
                config={
                    "units":      cfg.get("units", w.shape[1] if w is not None else 0),
                    "activation": cfg.get("activation", "linear"),
                    "use_bias":   cfg.get("use_bias", True),
                },
                original=layer,
            )

        # ---- Conv2D ----
        if "conv2d" in type_name or "conv" in type_name:
            w, b = (raw_weights[0], raw_weights[1]) if len(raw_weights) >= 2 else (None, None)
            # Keras uses NHWC; we need NCHW — transpose weights
            w_nchw = np.transpose(w, (3, 2, 0, 1)) if w is not None else None
            return LayerSpec(
                layer_type="conv2d",
                name=layer.name,
                weights=[w_nchw, b] if w_nchw is not None else [],
                config={
                    "filters":    cfg.get("filters", 0),
                    "kernel_size": cfg.get("kernel_size", (3, 3)),
                    "strides":    cfg.get("strides", (1, 1)),
                    "padding":    cfg.get("padding", "valid"),
                    "activation": cfg.get("activation", "linear"),
                },
                original=layer,
            )

        # ---- Activation ----
        if "activation" in type_name:
            return LayerSpec(
                layer_type="activation",
                name=layer.name,
                weights=[],
                config={"activation": cfg.get("activation", "linear")},
                original=layer,
            )

        # ---- Pooling ----
        if "pool" in type_name:
            mode = "global_avg" if "global" in type_name else (
                "max" if "max" in type_name else "avg"
            )
            return LayerSpec(
                layer_type="pooling",
                name=layer.name,
                weights=[],
                config={
                    "mode":      mode,
                    "pool_size": cfg.get("pool_size", (2, 2)),
                    "strides":   cfg.get("strides", (2, 2)),
                },
                original=layer,
            )

        # ---- Flatten ----
        if "flatten" in type_name:
            return LayerSpec(
                layer_type="flatten",
                name=layer.name,
                weights=[],
                config={},
                original=layer,
            )

        # ---- Unknown ----
        logger.debug(f"Layer '{layer.name}' ({type_name}) mapped to 'unknown'.")
        return LayerSpec(
            layer_type="unknown",
            name=layer.name,
            weights=raw_weights,
            config=cfg,
            original=layer,
        )

    # ------------------------------------------------------------------
    # Dict spec parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_dict(spec: Dict) -> List[LayerSpec]:
        """
        Parse a simple dict model definition. Example format::

            {
                "layers": [
                    {
                        "type": "dense",
                        "name": "fc1",
                        "weights": [W_array, b_array],
                        "config": {"activation": "relu"}
                    },
                    ...
                ]
            }
        """
        specs = []
        for layer_def in spec.get("layers", []):
            specs.append(LayerSpec(
                layer_type = layer_def.get("type", "unknown"),
                name       = layer_def.get("name", f"layer_{len(specs)}"),
                weights    = layer_def.get("weights", []),
                config     = layer_def.get("config", {}),
                original   = layer_def,
            ))
        return specs
