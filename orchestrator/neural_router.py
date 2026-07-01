"""Routeur neuronal : classification via un Transformer 12 couches entraîné.

Ce module expose un :class:`NeuralRouter` avec la **même interface publique**
que :class:`orchestrator.router.Router` :

    - ``classify(query) -> str`` : retourne un identifiant de catégorie.
    - ``score(query) -> dict[str, float]`` : retourne les probabilités par catégorie.

Le routeur charge un Transformer PyTorch et un tokenizer word-level depuis
des fichiers de checkpoint. Si les checkpoints sont absents, une erreur claire
est levée avec les instructions pour lancer l'entraînement.

Le device (CPU/MPS/CUDA) est détecté automatiquement.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .models import MODEL_NAMES
from .tokenizer import WordTokenizer

logger = logging.getLogger(__name__)

# Import lazy de PyTorch (le core du package n'en dépend pas).
try:
    import torch
    from .neural_model import TransformerClassifier12
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
    """Routeur basé sur un Transformer 12 couches entraîné + tokenizer word-level.

    Même contrat que :class:`orchestrator.router.Router` : l'orchestrateur
    peut utiliser l'un ou l'autre indifféremment grâce à l'interface
    ``classify``/``score``.

    Args:
        model_path: chemin vers le checkpoint PyTorch (.pt).
        tokenizer_path: chemin vers le tokenizer sauvegardé (.json).
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
        tokenizer_path: str = "checkpoints/tokenizer.json",
        confidence_threshold: float = 0.5,
        device: str = "auto",
    ) -> None:
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.confidence_threshold = confidence_threshold

        # Vérification de l'existence des checkpoints.
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Checkpoint modèle introuvable : {self.model_path}\n"
                f"Pour entraîner le modèle, lancez :\n"
                f"  python -m orchestrator.train --epochs 30"
            )
        if not self.tokenizer_path.exists():
            raise FileNotFoundError(
                f"Checkpoint tokenizer introuvable : {self.tokenizer_path}\n"
                f"Pour entraîner le modèle, lancez :\n"
                f"  python -m orchestrator.train --epochs 30"
            )

        # Device.
        self.device = _auto_device() if device == "auto" else torch.device(device)

        # Chargement du tokenizer.
        self.tokenizer = WordTokenizer.load(self.tokenizer_path)
        logger.debug(
            "Tokenizer chargé : %d termes, max_seq_len=%d, device=%s",
            self.tokenizer.vocab_size,
            self.tokenizer.max_seq_len,
            self.device,
        )

        # Chargement du modèle PyTorch.
        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=True,
        )

        # Reconstruction du modèle à partir des hyperparamètres sauvegardés.
        vocab_size = checkpoint.get("vocab_size", self.tokenizer.vocab_size)
        d_model = checkpoint.get("d_model", 256)
        nhead = checkpoint.get("nhead", 8)
        num_layers = checkpoint.get("num_layers", 12)
        dim_ff = checkpoint.get("dim_feedforward", 1024)
        max_seq_len = checkpoint.get("max_seq_len", self.tokenizer.max_seq_len)

        self.model = TransformerClassifier12(
            vocab_size=vocab_size,
            num_classes=len(MODEL_NAMES),
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_ff,
            max_seq_len=max_seq_len,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        n_params = self.model.count_parameters()
        logger.info(
            "NeuralRouter initialisé : %s, %d params, %d classes, device=%s",
            self.model.__class__.__name__,
            n_params,
            len(MODEL_NAMES),
            self.device,
        )

    def classify(self, query: str) -> str:
        """Identifie la catégorie de la requête via le Transformer.

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
        # Tokenisation → input_ids + attention_mask.
        input_ids, attention_mask = self.tokenizer.encode(query)

        # Conversion en tenseurs et ajout batch dim.
        x_ids = torch.from_numpy(input_ids).unsqueeze(0).to(self.device)
        x_mask = torch.from_numpy(attention_mask).unsqueeze(0).to(self.device)

        # Inférence sans gradient.
        with torch.no_grad():
            probs = self.model.predict_proba(x_ids, x_mask)  # (1, num_classes)
        probs_np = probs.cpu().numpy().flatten()

        # Construction du dictionnaire dans l'ordre canonique de MODEL_NAMES.
        return {
            name: float(probs_np[i])
            for i, name in enumerate(MODEL_NAMES)
        }
