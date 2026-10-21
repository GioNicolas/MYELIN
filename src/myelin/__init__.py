"""MYELIN: Memory-Yielded Early-Layer Inference Network.

Consolidação sináptica guiada por estresse, das camadas profundas para as rasas,
em tempo de inferência.
"""

from myelin.fast_synapse import FastSynapse
from myelin.knn_cache import KNNCache
from myelin.lens import TunedLens
from myelin.neural_cache import NeuralCache
from myelin.stacked_memory import OnlineBias, StackedMemory
from myelin.stress_gate import StressGate, entropy_stress, margin_stress
from myelin.wrapper import LayerStates, SimulationLog, capture, simulate

__all__ = [
    "FastSynapse",
    "KNNCache",
    "LayerStates",
    "NeuralCache",
    "OnlineBias",
    "SimulationLog",
    "StackedMemory",
    "StressGate",
    "TunedLens",
    "capture",
    "entropy_stress",
    "margin_stress",
    "simulate",
]
__version__ = "0.1.0"
