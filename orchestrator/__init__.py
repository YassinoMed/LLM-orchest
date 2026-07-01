"""Orchestrateur intelligent de modèles de langage (routage déterministe + neuronal).

API publique :
    - :class:`Orchestrator` : classe principale (routeur heuristique ou neuronal).
    - :class:`RoutingResult` : résultat du routage.
    - :class:`Router` : routeur heuristique (paramétrable).
    - :class:`NeuralRouter` : routeur neuronal (nécessite PyTorch + entraînement).
    - :data:`MODEL_NAMES` : catalogue des modèles spécialisés.
    - :func:`reduce` : reformulation d'une requête.
    - :func:`normalize` : normalisation de texte.

Exemple (heuristique, par défaut)::

    from orchestrator import Orchestrator
    orch = Orchestrator()
    print(orch.to_json("Combien font 345 * 678 ?"))
    # {"model": "math", "reduced_prompt": "Calcule 345 * 678."}

Exemple (neuronal, après entraînement)::

    orch = Orchestrator(router_mode="neural")
    print(orch.to_json("Combien font 345 * 678 ?"))
"""

from .models import MODEL_NAMES, RoutingResult
from .orchestrator import Orchestrator
from .reducer import reduce
from .router import Router, normalize

# Export NeuralRouter uniquement si PyTorch est disponible (import lazy).
_neural_router_available = False
try:
    from .neural_router import NeuralRouter
    _neural_router_available = True
except ImportError:
    pass

__all__ = [
    "Orchestrator",
    "Router",
    "NeuralRouter",
    "RoutingResult",
    "MODEL_NAMES",
    "normalize",
    "reduce",
]

# Ne pas inclure NeuralRouter dans __all__ si PyTorch n'est pas installé.
if not _neural_router_available:
    __all__.remove("NeuralRouter")

__version__ = "0.1.0"
