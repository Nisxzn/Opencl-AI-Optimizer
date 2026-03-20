"""
Graph Optimizer
===============
Applies graph-level optimizations to the parsed LayerSpec list before
the optimized model is assembled. Current passes:

  1. **Activation Fusion** — folds a standalone ActivationLayer into the
     preceding Dense or Conv2D layer (avoids an extra kernel launch).
  2. **Constant Folding** — removes no-op identity layers.
  3. **Layer Summary** — prints a human-readable optimization report.

Each pass is a separate method for easy extension.
"""

from typing import List
from optimizer.compiler.model_parser import LayerSpec
from optimizer.utils.logger import get_logger

logger = get_logger(__name__)


class GraphOptimizer:
    """
    Applies a series of optimization passes on a LayerSpec list.

    Usage:
        opt   = GraphOptimizer()
        specs = opt.optimize(specs)
    """

    def optimize(self, specs: List[LayerSpec]) -> List[LayerSpec]:
        """
        Run all registered optimization passes.

        Args:
            specs : List of LayerSpec objects from ModelParser.

        Returns:
            Optimized list of LayerSpec objects.
        """
        original_count = len(specs)
        logger.info(f"Graph optimizer: {original_count} layers input.")

        specs = self._fuse_activations(specs)
        specs = self._remove_identity_layers(specs)

        fused_count = len(specs)
        logger.info(
            f"Graph optimizer: {fused_count} layers after optimization "
            f"({original_count - fused_count} layers eliminated)."
        )

        self._print_graph(specs)
        return specs

    # ------------------------------------------------------------------
    # Pass 1: Activation Fusion
    # ------------------------------------------------------------------

    def _fuse_activations(self, specs: List[LayerSpec]) -> List[LayerSpec]:
        """
        Fold standalone activation layers into the preceding Dense / Conv2D
        layer. Example:
            Dense(linear) → Activation(relu)
        becomes:
            Dense(relu)
        """
        fused = []
        i = 0
        while i < len(specs):
            current = specs[i]

            # Look ahead: is the next layer a standalone activation?
            if (
                i + 1 < len(specs)
                and specs[i + 1].layer_type == "activation"
                and current.layer_type in ("dense", "conv2d")
                and current.config.get("activation", "linear") == "linear"
            ):
                next_spec = specs[i + 1]
                activation = next_spec.config.get("activation", "linear")

                logger.debug(
                    f"  Fusing activation '{activation}' "
                    f"into layer '{current.name}'"
                )

                # Mutate the current spec to include the activation
                current.config["activation"] = activation
                fused.append(current)
                i += 2  # skip the now-redundant activation layer
            else:
                fused.append(current)
                i += 1

        return fused

    # ------------------------------------------------------------------
    # Pass 2: Remove identity / no-op layers
    # ------------------------------------------------------------------

    @staticmethod
    def _remove_identity_layers(specs: List[LayerSpec]) -> List[LayerSpec]:
        """
        Remove layers that do nothing:
          - Activation 'linear' standalone layers
          - Input / Lambda layers that are known passthroughs
        """
        filtered = []
        for spec in specs:
            if (
                spec.layer_type == "activation"
                and spec.config.get("activation", "linear") == "linear"
            ):
                logger.debug(f"  Removing no-op linear activation: '{spec.name}'")
                continue
            if spec.layer_type in ("input_layer", "input"):
                logger.debug(f"  Removing input layer '{spec.name}' (not needed for inference).")
                continue
            filtered.append(spec)
        return filtered

    # ------------------------------------------------------------------
    # Utility: print optimized graph
    # ------------------------------------------------------------------

    @staticmethod
    def _print_graph(specs: List[LayerSpec]) -> None:
        logger.info("Optimized layer graph:")
        for i, spec in enumerate(specs):
            act = spec.config.get("activation", "")
            act_str = f" → {act}" if act and act != "linear" else ""
            logger.info(f"  [{i:02d}] {spec.layer_type:12s} | {spec.name}{act_str}")
