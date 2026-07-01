"""Orchestrateur intelligent de modèles de langage (routage déterministe, 100% local).

API publique :
    - :class:`Orchestrator` : classe principale.
    - :class:`RoutingResult` : résultat du routage.
    - :class:`Router` : moteur de classification (paramétrable).
    - :data:`MODEL_NAMES` : catalogue des modèles spécialisés.
    - :func:`reduce` : reformulation d'une requête (rarement utilisée seule).

Exemple::

    from orchestrator import Orchestrator
    orch = Orchestrator()
    print(orch.to_json("Combien font 345 * 678 ?"))
    # {"model": "math", "reduced_prompt": "Calcule 345 * 678."}
"""

from .models import MODEL_NAMES, RoutingResult
from .orchestrator import Orchestrator
from .reducer import reduce
from .router import Router, normalize

__all__ = [
    "Orchestrator",
    "Router",
    "RoutingResult",
    "MODEL_NAMES",
    "normalize",
    "reduce",
]

__version__ = "0.1.0"
