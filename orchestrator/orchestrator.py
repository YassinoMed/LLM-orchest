"""Point d'entrée principal de l'orchestrateur.

Assemble le :class:`~orchestrator.router.Router` (classification) et le module
:mod:`orchestrator.reducer` (reformulation) pour produire un
:class:`~orchestrator.models.RoutingResult`, sérialisable en JSON conforme à la spec :

    {"model": "<catégorie>", "reduced_prompt": "<prompt épuré>"}
"""

from __future__ import annotations

from .models import RoutingResult
from .reducer import reduce as reduce_prompt
from .router import Router


class Orchestrator:
    """Orchestrateur de modèles : analyse une requête et produit un routage.

    Usage type::

        orch = Orchestrator()
        result = orch.route("Combien font 345 * 678 ?")
        print(result.to_json())   # {"model": "math", "reduced_prompt": "Calcule 345 * 678."}

    Args:
        router: instance de :class:`Router` personnalisée (seuils, signaux).
            Si ``None``, un routeur par défaut est créé.
    """

    def __init__(self, router: Router | None = None) -> None:
        self.router = router if router is not None else Router()

    def route(self, query: str) -> RoutingResult:
        """Analyse la requête et renvoie le résultat du routage.

        Args:
            query: requête en langage naturel de l'utilisateur.

        Returns:
            Un :class:`RoutingResult` (``model`` + ``reduced_prompt``).
        """
        model = self.router.classify(query)
        reduced = reduce_prompt(query, model)
        return RoutingResult(model=model, reduced_prompt=reduced)

    def to_json(self, query: str, *, indent: int | None = None) -> str:
        """Raccourci : route la requête puis renvoie directement le JSON.

        Args:
            query: requête de l'utilisateur.
            indent: indentation optionnelle du JSON (``None`` = compact).
        """
        return self.route(query).to_json(indent=indent)
