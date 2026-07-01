"""Routeur neuronal : classification de requêtes via un MLP entraîné.

Ce module expose un :class:`NeuralRouter` avec la **même interface publique**
que :class:`orchestrator.router.Router` :

    - ``classify(query) -> str`` : retourne un identifiant de catégorie.
    - ``score(query) -> dict[str, float]`` : retourne les probabilités par catégorie.

Le routeur charge un modèle PyTorch entraîné et un vectoriseur TF-IDF depuis
des fichiers de checkpoint. Si les checkpoints sont absents, une erreur claire
est levée avec les instructions pour lancer l'entraînement.

Le device (CPU/MPS/CUDA) est détecté automatiquement.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .models import MODEL_NAMES
from .tfidf import TfidfVectorizer

logger = logging.getLogger(__name__)

# Import lazy de PyTorch (le core du package n'en dépend pas).
try:
    import torch
    from .neural_model import MLPClassifier12
except ImportError as exc:
    raise ImportError(
        "PyTorch est requis pour le mode neuronal. "
        "Installe-le avec : pip install torch"
    ) from exc


def _auto_device() -> torch.device:
    """Détecte le meilleur device disponible."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class NeuralRouter:
    """Routeur basé sur un MLP entraîné + TF-IDF.

    Même contrat que :class:`orchestrator.router.Router` : l'orchestrateur
    peut utiliser l'un ou l'autre indifféremment grâce à l'interface
    ``classify``/``score``.

    Args:
        model_path: chemin vers le checkpoint PyTorch (.pt).
        tfidf_path: chemin vers le vectoriseur TF-IDF sauvegardé (.json).
        confidence_threshold: probabilité minimale pour accepter la prédiction.
            En dessous, la requête est considérée comme ambiguë et classée
            en ``"general"`` (même logique que le seuil du routeur heuristique).
        device: device PyTorch (``"auto"`` = détection automatique).

    Raises:
        FileNotFoundError: si les checkpoints n'existent pas (message clair
            avec instructions pour lancer l'entraînement).
    """

    def __init__(
        self,
        model_path: str = "checkpoints/model.pt",
        tfidf_path: str = "checkpoints/tfidf.json",
        confidence_threshold: float = 0.5,
        device: str = "auto",
    ) -> None:
        self.model_path = Path(model_path)
        self.tfidf_path = Path(tfidf_path)
        self.confidence_threshold = confidence_threshold

        # Vérification de l'existence des checkpoints.
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Checkpoint modèle introuvable : {self.model_path}\n"
                f"Pour entraîner le modèle, lancez :\n"
                f"  python -m orchestrator.train --epochs 30"
            )
        if not self.tfidf_path.exists():
            raise FileNotFoundError(
                f"Checkpoint TF-IDF introuvable : {self.tfidf_path}\n"
                f"Pour entraîner le modèle, lancez :\n"
                f"  python -m orchestrator.train --epochs 30"
            )

        # Device.
        self.device = _auto_device() if device == "auto" else torch.device(device)

        # Chargement du vectoriseur TF-IDF.
        self.vectorizer = TfidfVectorizer.load(self.tfidf_path)
        logger.debug(
            "TF-IDF chargé : %d termes, device=%s",
            self.vectorizer.vocab_size,
            self.device,
        )

        # Chargement du modèle PyTorch.
        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=True,
        )
        self.model = MLPClassifier12(
            input_dim=self.vectorizer.vocab_size,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        n_params = self.model.count_parameters()
        logger.info(
            "NeuralRouter initialisé : %d paramètres, %d classes, device=%s",
            n_params,
            len(MODEL_NAMES),
            self.device,
        )

    def classify(self, query: str) -> str:
        """Identifie la catégorie de la requête via le MLP.

        Contract identique à :meth:`orchestrator.router.Router.classify`.

        Args:
            query: requête en langage naturel.

        Returns:
            Identifiant de catégorie (l'un des :data:`MODEL_NAMES`).
            Si la confiance maximale est inférieure au seuil, retourne
            ``"general"`` (ambiguïté).
        """
        scores = self.score(query)
        best_cat = max(scores, key=scores.get)
        if scores[best_cat] < self.confidence_threshold:
            return "general"
        return best_cat

    def score(self, query: str) -> dict[str, float]:
        """Retourne les probabilités par catégorie pour la requête.

        Contract identique à :meth:`orchestrator.router.Router.score`.

        Args:
            query: requête en langage naturel.

        Returns:
            Dictionnaire ``{nom_catégorie: probabilité}`` pour toutes les
            catégories de :data:`MODEL_NAMES`.
        """
        # Vectorisation TF-IDF.
        tfidf_vec = self.vectorizer.transform_one(query)  # shape: (vocab_size,)
        x = torch.from_numpy(tfidf_vec).unsqueeze(0).to(self.device)  # (1, vocab_size)

        # Inférence sans gradient.
        with torch.no_grad():
            probs = self.model.predict_proba(x)  # (1, num_classes)
        probs_np = probs.cpu().numpy().flatten()

        # Construction du dictionnaire dans l'ordre canonique de MODEL_NAMES.
        return {
            name: float(probs_np[i])
            for i, name in enumerate(MODEL_NAMES)
        }
