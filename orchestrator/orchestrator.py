"""Point d'entrée principal de l'orchestrateur.

Assemble un routeur (heuristique ou neuronal) et le module
:mod:`orchestrator.reducer` (reformulation) pour produire un
:class:`~orchestrator.models.RoutingResult`, sérialisable en JSON conforme à la spec :

    {"model": "<catégorie>", "reduced_prompt": "<prompt épuré>"}
"""

from __future__ import annotations

import logging

from .models import RoutingResult
from .reducer import reduce as reduce_prompt
from .router import Router

logger = logging.getLogger(__name__)


class Orchestrator:
    """Orchestrateur de modèles : analyse une requête et produit un routage.

    Usage type (routeur heuristique, par défaut)::

        orch = Orchestrator()
        result = orch.route("Combien font 345 * 678 ?")
        print(result.to_json())   # {"model": "math", "reduced_prompt": "Calcule 345 * 678."}

    Usage avec le routeur neuronal::

        orch = Orchestrator(router_mode="neural")
        result = orch.route("Combien font 345 * 678 ?")

    Args:
        router: instance de routeur personnalisée (contrat : méthode ``classify``
            + ``score``). Prioritaire sur ``router_mode``. Permet l'injection
            de dépendance pour les tests.
        router_mode: ``"heuristic"`` (défaut) ou ``"neural"``. Ignoré si
            ``router`` est fourni explicitement.
        **neural_kwargs: arguments passés au :class:`~orchestrator.neural_router.NeuralRouter`
            (``model_path``, ``tfidf_path``, ``confidence_threshold``, ``device``).
    """

    def __init__(
        self,
        router: object | None = None,
        router_mode: str = "heuristic",
        **neural_kwargs: object,
    ) -> None:
        if router is not None:
            # Injection de dépendance directe (prioritaire, pour les tests).
            self.router = router
        elif router_mode == "neural":
            try:
                from .neural_router import NeuralRouter
                self.router = NeuralRouter(**neural_kwargs)
                logger.info("Routeur neuronal actif (device=%s).", self.router.device)
            except (ImportError, FileNotFoundError) as exc:
                logger.warning(
                    "Impossible de charger le routeur neuronal : %s. "
                    "Bascule sur le routeur heuristique.",
                    exc,
                )
                self.router = Router()
        elif router_mode == "heuristic":
            self.router = Router()
        else:
            raise ValueError(
                f"router_mode invalide : {router_mode!r}. "
                f"Choisis parmi 'heuristic' ou 'neural'."
            )

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
